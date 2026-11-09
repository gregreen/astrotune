"""Train the bijector that maps every catalog onto the harmonized space.

    python -m astrotune.train --run-name my_run
    astrotune-train --run-name my_run --n-epochs 4

Outputs go to ``models/<run-name>/``::

    config.json            hyperparameters + provenance
    progress/progress_0001.png, progress_0002.png, ...   periodic figures
    checkpoints/           latest few resumable checkpoints (model + optimizer
                           + metrics); see ``--resume``
    bijector/              SavedModel with to_harmonized / from_harmonized
    metrics_history.json   per-step loss and penalties
    agreement_history.json catalog agreement (held-out pairs) at each figure
    evaluation/            final before/after agreement tables and figures

A run interrupted by Ctrl-C or SIGTERM saves a checkpoint first; continue it
with ``astrotune-train --run-name my_run --resume``.

The loss is the pair term (chi^2 of the star's position in catalog ``k`` when
mapped there from catalog ``j`` through the harmonized space) plus
regularization terms on control points. See ``TrainConfig`` for the
hyperparameters, which default to the values of the reference notebook.

The notebook's learned error model (``ln_err_model``), which fitted a
correction to the catalogs' reported errors, is deliberately not ported. The
pair term uses the reported ``sigma_x`` directly.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import signal
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tqdm.auto import tqdm

from . import config, registry
from .evaluate import (
    PARAM_SHORT,
    evaluate_bijector,
    format_summary,
    quick_agreement,
    validation_pairs,
)
from .model import ODEBijector
from .training_data import load_training_data, read_attrs
from .training_plots import catalog_colors, catalog_sample_valid, catalog_samples, save_progress_figure

log = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    # Network
    hidden_size: int = 64
    n_hidden: int = 3
    ode_rtol: float = 1e-5
    ode_atol: float = 1e-5

    # Optimization
    batch_size: int = 1024
    n_epochs: int = 32
    val_frac: float = 1 / 8
    learning_rate: float = 1e-3  # halved at each of 7 evenly spaced steps
    clip_norm: float = 100.0
    seed: int = 3
    val_every: int = 1  # evaluate one validation batch every this many steps
    max_steps: Optional[int] = None  # stop early (mainly for testing)

    # Pair loss
    # chi^2 above this is softened to ~log growth. The notebook used 15, but
    # with the reported uncertainties chi^2 averages ~16, so that would flatten
    # a large part of the distribution.
    chi2_turnover: float = 50.0
    # The reference notebook's active loss leaves out the pair path-length
    # penalty (it is reported as a metric, but only the control-point path
    # length enters the loss). Set to True to add it (cf. the paper).
    pair_path_in_loss: bool = False

    # Path-length penalty
    l1_path: float = 1e-1
    l2_path: float = 1e-1

    # Distance of control points from their images
    l1_total: float = 1e-1
    l2_total: float = 1e-1
    sigma_ctrl: float = 0.3  # scatter added to control points

    # Stretching / contraction of space
    eps_div: float = 0.1
    weight_div: float = 3.0

    # Weight decay
    l2_weights: float = 1e-4

    # Freeze variables whose names contain these strings once this fraction of
    # training is done. The notebook intended to freeze the embedding and FiLM
    # layers at 3/4 of training, but a variable-name typo meant it never did;
    # the published run therefore trained all variables throughout. Default:
    # no freezing.
    freeze_after_frac: Optional[float] = None
    freeze_vars: Tuple[str, ...] = ("embedding", "film")

    # Progress figures
    plot_every_seconds: Optional[float] = 5 * 60.0
    plot_every_steps: Optional[int] = None
    n_plot: int = 2048  # stars per catalog in the progress figure
    # Held-out pairs used to track catalog agreement at each progress figure
    agreement_n_pairs: int = 32768

    # Resumable state (model, optimizer, metrics) is written at each progress
    # figure and, in addition, at this interval
    checkpoint_every_seconds: Optional[float] = 10 * 60.0


#: Settings that may differ when resuming a run; everything else must match
#: the saved configuration, because it shapes the learning-rate schedule, the
#: data order or the loss.
RESUME_MUTABLE = (
    "plot_every_seconds",
    "plot_every_steps",
    "checkpoint_every_seconds",
    "agreement_n_pairs",
    "n_plot",
)


def arcsinh_of_exp(lnx):
    """``arcsinh(x)`` given ``ln(x)``, in a numerically stable way."""
    return tf.keras.ops.logaddexp(lnx, 0.5 * tf.math.softplus(2 * lnx))


def _make_dataset(arrays, batch_size: int, n_repeat: int, seed: int):
    ds = tf.data.Dataset.from_tensor_slices(tuple(arrays))
    ds = ds.shuffle(buffer_size=batch_size * 64, seed=seed)
    ds = ds.repeat(n_repeat)
    return ds.batch(batch_size, drop_remainder=True)


class Trainer:
    """Holds the model, the data pipelines and the training step."""

    def __init__(self, pair_dict, ctrl_dict, cfg: TrainConfig):
        self.cfg = cfg
        self.n_dim = pair_dict["x0"].shape[1]
        self.n_cat = int(ctrl_dict["c_control"].max()) + 1

        tf.random.set_seed(cfg.seed)

        self._make_data(pair_dict, ctrl_dict)
        self.samples = catalog_samples(ctrl_dict, self.n_cat, cfg.n_plot)
        self.sample_valid = catalog_sample_valid(ctrl_dict, self.n_cat, cfg.n_plot)
        self.eval_pairs = validation_pairs(pair_dict, cfg.val_frac, cfg.agreement_n_pairs)

        self.bij = ODEBijector(
            self.n_dim,
            self.n_cat,
            cfg.hidden_size,
            cfg.n_hidden,
            rtol=cfg.ode_rtol,
            atol=cfg.ode_atol,
            vector_network_kwargs=dict(use_residual=True, use_film=True, use_layer_norm=True),
        )
        self.variables = list(self.bij.trainable_variables)
        n_vars = sum(int(np.prod(v.shape)) for v in self.variables)
        log.info("%d variables in bijector model.", n_vars)

        self.learning_rate = keras.optimizers.schedules.PiecewiseConstantDecay(
            [i * self.n_steps // 8 for i in range(1, 8)],
            [cfg.learning_rate / 2**i for i in range(8)],
        )
        self.opt = keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.opt.build(self.variables)

        self.grad_step = tf.function(self._grad_step)

        # Kept off the bijector so that it is not tracked or exported
        self.to_harmonized = tf.function(
            self.bij._forward,
            input_signature=(
                tf.TensorSpec([None, self.n_dim], tf.float32),
                tf.TensorSpec([None], tf.int32),
            ),
        )

    # -- data ---------------------------------------------------------------
    def _make_data(self, pair_dict, ctrl_dict) -> None:
        cfg = self.cfg
        n_val = int(len(pair_dict["x0"]) * cfg.val_frac)
        val_epochs = int(np.ceil(cfg.n_epochs / cfg.val_frac))
        pair_keys = ("x0", "x1", "xe0", "xe1", "xs0", "xs1", "c0", "c1")
        ctrl_keys = ("x_control", "xe_control", "c_control")

        # Validation data are the first n_val rows
        args = (cfg.batch_size,)
        self.data_pairs = _make_dataset(
            [pair_dict[k][n_val:] for k in pair_keys], *args, cfg.n_epochs, cfg.seed
        )
        self.val_pairs = _make_dataset(
            [pair_dict[k][:n_val] for k in pair_keys], *args, val_epochs, cfg.seed
        )

        n_val = int(len(ctrl_dict["x_control"]) * cfg.val_frac)
        self.data_ctrl = _make_dataset(
            [ctrl_dict[k][n_val:] for k in ctrl_keys], *args, cfg.n_epochs, cfg.seed
        )
        self.val_ctrl = _make_dataset(
            [ctrl_dict[k][:n_val] for k in ctrl_keys], *args, val_epochs, cfg.seed
        )

        self.n_steps = min(
            int(ds.cardinality()) for ds in (self.data_pairs, self.data_ctrl)
        )
        if cfg.max_steps is not None:
            self.n_steps = min(self.n_steps, cfg.max_steps)
        log.info("%d steps expected.", self.n_steps)

    # -- loss ---------------------------------------------------------------
    def _grad_step(
        self,
        x0, x1, xe0, xe1, xs0, xs1, c0, c1,
        x_ctrl, xe_ctrl, c_ctrl,
        apply_grads=True,
        freeze_list=(),
    ):
        cfg = self.cfg
        bij = self.bij

        vars_filtered = [
            v for v in self.variables if not any(s in v.name for s in freeze_list)
        ]

        # Scatter control points, and create an offset copy of each, which is
        # used to measure stretching/contraction of space
        x_ctrl = x_ctrl + cfg.sigma_ctrl * tf.random.normal(tf.shape(x_ctrl))
        x1_ctrl = x_ctrl + cfg.eps_div * tf.random.normal(tf.shape(x_ctrl))

        # Where a parameter is unknown for x0 (e.g., [alpha/Fe]), it must not
        # contribute to the loss: make it also unknown for x1.
        xs1 = tf.where(xs0 > 1e6, xs0, xs1)

        with tf.GradientTape(watch_accessed_variables=False) as g:
            g.watch(vars_filtered)

            # Draw x0 from its uncertainty distribution
            xs0_clipped = tf.clip_by_value(xs0, 1e-5, 1.0)
            x0 = x0 + xs0_clipped * tf.random.normal(tf.shape(x0))

            # Map x0 to the harmonized space and into the space of x1's catalog
            x0_pred, s0 = bij.forward_with_s(x0, c=c0)
            x1_pred, s1 = bij.inverse_with_s(x0_pred, c=c1)

            # Map control points (and their offset copies) to the common space
            x_ctrl_pred, s_ctrl = bij.forward_with_s(x_ctrl, c=c_ctrl)
            x1_ctrl_pred, _ = bij.forward_with_s(x1_ctrl, c=c_ctrl)

            # chi^2 of the mapped x0 relative to x1, evaluated in log space for
            # stability, then softened: turnover * arcsinh(chi^2 / turnover)
            ln_x1_diffsq = tf.math.log((x1_pred - x1) ** 2 + 1e-30)
            ln_chi2 = tf.math.reduce_logsumexp(ln_x1_diffsq - 2 * tf.math.log(xs1), axis=1)
            chi2 = cfg.chi2_turnover * arcsinh_of_exp(ln_chi2 - tf.math.log(cfg.chi2_turnover))
            chi2 = tf.reduce_mean(chi2)

            # Path-length penalty of pair transformations
            length2 = s0**2 + s1**2
            penalty_pair = cfg.l1_path * tf.reduce_mean(
                tf.math.sqrt(length2)
            ) + cfg.l2_path * tf.reduce_mean(length2)

            # Penalties on control-point transformations
            length1_ctrl = tf.reduce_mean(tf.abs(x_ctrl_pred - x_ctrl) / xe_ctrl)
            length2_ctrl = tf.reduce_mean(((x_ctrl_pred - x_ctrl) / xe_ctrl) ** 2)
            penalty_ctrl = (
                cfg.l1_total * length1_ctrl
                + cfg.l2_total * length2_ctrl
                + cfg.l1_path * tf.reduce_mean(tf.abs(s_ctrl))
                + cfg.l2_path * tf.reduce_mean(s_ctrl**2)
            )

            # Penalty on stretching/contraction of space
            length_div = tf.reduce_sum((x1_ctrl - x_ctrl) ** 2, axis=1)
            length_div_p = tf.reduce_sum((x1_ctrl_pred - x_ctrl_pred) ** 2, axis=1)
            eps = 1e-2 * cfg.eps_div
            penalty_div = cfg.weight_div * tf.reduce_mean(
                tf.math.log((length_div + eps) / (length_div_p + eps)) ** 2
            )

            # Weight decay on the dense-layer weights
            penalty_l2 = cfg.l2_weights * tf.add_n(
                [tf.nn.l2_loss(v) for v in vars_filtered if "w:" in v.name]
            )

            loss = chi2 + penalty_ctrl + penalty_div + penalty_l2
            if cfg.pair_path_in_loss:
                loss = loss + penalty_pair

        res = {
            "loss": loss,
            "chi2": chi2,
            "penalty_pair": penalty_pair,
            "penalty_ctrl": penalty_ctrl,
            "penalty_div": penalty_div,
            "penalty_l2": penalty_l2,
        }

        if apply_grads:
            grads = g.gradient(loss, vars_filtered)
            grads, norm = tf.clip_by_global_norm(grads, cfg.clip_norm)
            self.opt.apply_gradients(zip(grads, vars_filtered))
            res["norm"] = norm

        return res

    # -- resumable state ----------------------------------------------------
    def _manager(self, run_dir: Path) -> tf.train.CheckpointManager:
        ckpt = tf.train.Checkpoint(bij=self.bij)
        return tf.train.CheckpointManager(ckpt, str(Path(run_dir) / "checkpoints"), max_to_keep=3)

    def save_state(
        self, run_dir: Path, step: int, metrics, n_figs: int, elapsed: float,
        agreement, complete: bool = False,
    ) -> None:
        """Write everything needed to resume: model, optimizer, metrics, progress.

        The model goes into a ``tf.train.Checkpoint`` numbered by step, and
        the rest into ``state-<step>.npz`` next to it. The state file is
        written first (atomically), so a checkpoint never lacks its state.
        """
        ckpt_dir = Path(run_dir) / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        meta = dict(
            step=step, n_figs=n_figs, elapsed=elapsed, complete=complete,
            metrics=metrics, agreement=agreement,
        )
        arrays = {f"opt_{i}": np.asarray(v) for i, v in enumerate(self.opt.variables)}
        tmp = ckpt_dir / "state.tmp"
        with open(tmp, "wb") as f:
            np.savez(f, meta=np.array(json.dumps(meta)), **arrays)
        os.replace(tmp, ckpt_dir / f"state-{step}.npz")

        manager = self._manager(run_dir)
        if step != self._last_saved_step:
            manager.save(checkpoint_number=step)
            self._last_saved_step = step
        keep = {int(p.rsplit("-", 1)[1]) for p in manager.checkpoints}
        for f in ckpt_dir.glob("state-*.npz"):
            if int(f.stem.split("-")[1]) not in keep:
                f.unlink()

    def load_state(self, run_dir: Path) -> Optional[dict]:
        """Restore the latest complete checkpoint; return its metadata (or None)."""
        manager = self._manager(run_dir)
        for path in reversed(manager.checkpoints):
            step = int(path.rsplit("-", 1)[1])
            state_path = Path(run_dir) / "checkpoints" / f"state-{step}.npz"
            if not state_path.exists():
                continue
            tf.train.Checkpoint(bij=self.bij).restore(path).assert_existing_objects_matched()
            data = np.load(state_path)
            meta = json.loads(str(data["meta"]))
            opt_vars = self.opt.variables
            n_saved = sum(k.startswith("opt_") for k in data.files)
            if n_saved != len(opt_vars):
                raise ValueError(
                    f"Checkpoint has {n_saved} optimizer variables, expected {len(opt_vars)}."
                )
            for i, v in enumerate(opt_vars):
                v.assign(data[f"opt_{i}"])
            self._last_saved_step = step
            log.info("Resumed from %s (step %d)", path, step)
            return meta
        return None

    # -- training loop ------------------------------------------------------
    def train(self, run_dir: Path, resume: bool = False) -> Dict[str, list]:
        cfg = self.cfg
        run_dir = Path(run_dir)
        fig_dir = run_dir / "progress"
        fig_dir.mkdir(parents=True, exist_ok=True)

        labels = registry.labels()[: self.n_cat]
        colors = catalog_colors(self.n_cat)
        self._last_saved_step = -1

        metrics: Dict[str, list] = {"step_val": []}
        agreement: list = []
        self.agreement = agreement
        n_figs = 0
        elapsed_prev = 0.0
        start = 0
        if resume:
            meta = self.load_state(run_dir)
            if meta is None:
                raise FileNotFoundError(f"No checkpoint to resume from in {run_dir}/checkpoints")
            metrics, agreement = meta["metrics"], meta["agreement"]
            self.agreement = agreement
            n_figs, elapsed_prev, start = meta["n_figs"], meta["elapsed"], meta["step"]
            # Fresh noise after the restart (datasets were already seeded)
            tf.random.set_seed(cfg.seed + start)

        t_start = time.monotonic() - elapsed_prev
        t_last_plot = t_last_ckpt = time.monotonic()
        step_last_plot = start

        def checkpoint(step, complete=False):
            nonlocal t_last_ckpt
            self.save_state(
                run_dir, step, metrics, n_figs, time.monotonic() - t_start, agreement, complete
            )
            t_last_ckpt = time.monotonic()

        def make_figure(step):
            nonlocal n_figs, t_last_plot, step_last_plot
            elapsed = time.monotonic() - t_start
            path = fig_dir / f"progress_{n_figs + 1:04d}.png"
            summary = quick_agreement(self.to_harmonized, self.eval_pairs)
            agreement.append({"step": step, "summary": summary})
            log.info("Catalog agreement at step %d:\n%s", step, format_summary(summary))
            ok = [s for s in summary if s["n_pairs"]]
            title = (
                f"step {step:,} / {self.n_steps:,}  ({elapsed/60:.1f} min)\n"
                "mean over catalog pairs, catalog -> harmonized\n"
                "|offset|: " + ", ".join(
                    f"{PARAM_SHORT[s['param']]} {s['abs_bias_orig']:.3f}->{s['abs_bias_harm']:.3f}"
                    for s in ok
                )
                + "\nscatter: " + ", ".join(
                    f"{PARAM_SHORT[s['param']]} {s['sigma_orig']:.3f}->{s['sigma_harm']:.3f}"
                    for s in ok
                )
            )
            save_progress_figure(path, metrics, self.bij, self.samples, labels, colors,
                                 title=title, valid=self.sample_valid)
            n_figs += 1
            checkpoint(step)
            t_last_plot = time.monotonic()
            step_last_plot = step
            return path

        freeze_step = None
        if cfg.freeze_after_frac is not None:
            freeze_step = int(cfg.freeze_after_frac * self.n_steps)
        freeze_list: Tuple[str, ...] = ()
        if freeze_step is not None and start > freeze_step:
            freeze_list = cfg.freeze_vars

        # Continue the (seeded, hence reproducible) data order where it stopped
        n_val_done = -(-start // cfg.val_every)
        data_iter = zip(self.data_pairs.skip(start).prefetch(2),
                        self.data_ctrl.skip(start).prefetch(2))
        val_iter = zip(self.val_pairs.skip(n_val_done).prefetch(2),
                       self.val_ctrl.skip(n_val_done).prefetch(2))

        # Treat SIGTERM (e.g., from a job scheduler) like Ctrl-C: save, then exit
        def _on_sigterm(signum, frame):
            raise KeyboardInterrupt

        old_handler = None
        if threading.current_thread() is threading.main_thread():
            old_handler = signal.signal(signal.SIGTERM, _on_sigterm)

        pbar = tqdm(data_iter, total=self.n_steps, initial=start, desc="training")
        try:
            for i, (d_pair, d_ctrl) in enumerate(pbar, start=start):
                if i >= self.n_steps:
                    break

                if freeze_step is not None and i == freeze_step:
                    freeze_list = cfg.freeze_vars
                    log.info("Step %d: freezing variables matching %s", i, freeze_list)

                res = self.grad_step(*d_pair, *d_ctrl, apply_grads=True, freeze_list=freeze_list)
                for k, v in res.items():
                    metrics.setdefault(k, []).append(float(v.numpy()))

                if i % cfg.val_every == 0:
                    v_pair, v_ctrl = next(val_iter)
                    res = self.grad_step(
                        *v_pair, *v_ctrl, apply_grads=False, freeze_list=freeze_list
                    )
                    for k, v in res.items():
                        metrics.setdefault(k + "_val", []).append(float(v.numpy()))
                    metrics["step_val"].append(i)

                avg = {k: np.mean(v[-16:]) for k, v in metrics.items() if k != "step_val"}
                pbar.set_postfix(
                    {
                        "loss": avg["loss"],
                        "chi2": avg["chi2"],
                        "plty_p": avg["penalty_pair"],
                        "plty_c": avg["penalty_ctrl"],
                        "plty_d": avg["penalty_div"],
                        "plty_l2": avg["penalty_l2"],
                        "norm": avg["norm"],
                        "lr": float(self.learning_rate(i)),
                    },
                    refresh=False,
                )

                due = False
                if cfg.plot_every_seconds is not None:
                    due |= time.monotonic() > t_last_plot + cfg.plot_every_seconds
                if cfg.plot_every_steps is not None:
                    due |= (i + 1) - step_last_plot >= cfg.plot_every_steps
                if due and len(metrics["loss"]) > 1:
                    pbar.write(f"Wrote {make_figure(i + 1)}")
                elif (
                    cfg.checkpoint_every_seconds is not None
                    and time.monotonic() > t_last_ckpt + cfg.checkpoint_every_seconds
                ):
                    checkpoint(i + 1)
        except KeyboardInterrupt:
            done = len(metrics["loss"])
            pbar.close()
            checkpoint(done)
            log.warning(
                "Interrupted at step %d; resume with --resume --run-dir %s", done, run_dir
            )
            raise
        finally:
            if old_handler is not None:
                signal.signal(signal.SIGTERM, old_handler)

        pbar.close()
        done = len(metrics["loss"])
        if done > 1 and step_last_plot != done:
            log.info("Wrote %s", make_figure(done))
        checkpoint(done, complete=True)
        return metrics

    def save(self, run_dir: Path, metrics) -> None:
        """Write the SavedModel (with ``to_harmonized``/``from_harmonized``) and metrics."""
        run_dir = Path(run_dir)
        bij, n_dim = self.bij, self.n_dim

        bij.to_harmonized = tf.function(
            func=bij._forward,
            input_signature=(
                tf.TensorSpec([None, n_dim], tf.float32),  # x
                tf.TensorSpec([None], tf.int32),  # c
            ),
        )
        bij.from_harmonized = tf.function(
            func=bij._inverse,
            input_signature=(
                tf.TensorSpec([None, n_dim], tf.float32),
                tf.TensorSpec([None], tf.int32),
            ),
        )
        tf.saved_model.save(bij, str(run_dir / "bijector"))

        with open(run_dir / "metrics_history.json", "w") as f:
            json.dump(metrics, f)


def _set_memory_growth() -> None:
    for device in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(device, True)
        except RuntimeError:  # already initialized
            pass


def _check_resumable(saved: dict, cfg: TrainConfig, n_pairs: int, n_control: int) -> None:
    """Refuse to resume with settings that would change the run."""
    if (saved["n_pairs"], saved["n_control"]) != (n_pairs, n_control):
        raise ValueError(
            "The pairs/control-point file differs from the one the run was started with "
            f"({saved['n_pairs']} pairs, {saved['n_control']} control points)."
        )
    old = dataclasses.asdict(TrainConfig(**_config_from_json(saved["config"])))
    new = dataclasses.asdict(cfg)
    changed = [k for k in new if k not in RESUME_MUTABLE and new[k] != old[k]]
    if changed:
        raise ValueError(
            "Cannot change these settings when resuming: "
            + ", ".join(f"{k} ({old[k]!r} -> {new[k]!r})" for k in changed)
        )


def _config_from_json(d: dict) -> dict:
    d = dict(d)
    if "freeze_vars" in d:
        d["freeze_vars"] = tuple(d["freeze_vars"])
    return d


def run_training(
    cfg: TrainConfig,
    run_dir: Path,
    pairs_path: Optional[Path] = None,
    resume: bool = False,
) -> Dict[str, list]:
    """Train a bijector and write everything to ``run_dir``.

    With ``resume=True`` the run continues from the latest checkpoint in
    ``run_dir`` (model, optimizer state, metrics, figure numbering and data
    order are all restored).
    """
    run_dir = Path(run_dir)
    cfg_path = run_dir / "config.json"
    if not resume and (run_dir / "checkpoints").exists():
        raise FileExistsError(
            f"{run_dir} already contains a run. Use --resume to continue it, "
            "or choose a different --run-name."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    _set_memory_growth()

    pair_dict, ctrl_dict = load_training_data(pairs_path)
    log.info(
        "%d pairs, %d control points", len(pair_dict["x0"]), len(ctrl_dict["x_control"])
    )

    if resume:
        with open(cfg_path) as f:
            snapshot = json.load(f)
        _check_resumable(snapshot, cfg, len(pair_dict["x0"]), len(ctrl_dict["x_control"]))
        snapshot["config"] = dataclasses.asdict(cfg)
        snapshot.setdefault("resumed", []).append(datetime.now().isoformat(timespec="seconds"))
    else:
        snapshot = {
            "config": dataclasses.asdict(cfg),
            "pairs_path": str(pairs_path or config.PAIRS_PATH),
        "pairs_attrs": read_attrs(pairs_path),
            "n_pairs": len(pair_dict["x0"]),
            "n_control": len(ctrl_dict["x_control"]),
            "catalogs": list(registry.keys()[: int(ctrl_dict["c_control"].max()) + 1]),
            "tensorflow": tf.__version__,
            "started": datetime.now().isoformat(timespec="seconds"),
        }
    with open(cfg_path, "w") as f:
        json.dump(snapshot, f, indent=2)

    trainer = Trainer(pair_dict, ctrl_dict, cfg)
    metrics = trainer.train(run_dir, resume=resume)
    trainer.save(run_dir, metrics)
    log.info("Saved model to %s", run_dir)

    with open(run_dir / "agreement_history.json", "w") as f:
        json.dump(trainer.agreement, f)
    evaluate_bijector(
        trainer.to_harmonized, pair_dict, ctrl_dict, run_dir / "evaluation",
        val_frac=cfg.val_frac,
    )
    return metrics


def _add_config_args(parser: argparse.ArgumentParser) -> None:
    defaults = TrainConfig()
    for f in dataclasses.fields(TrainConfig):
        default = getattr(defaults, f.name)
        flag = "--" + f.name.replace("_", "-")
        if isinstance(default, bool):
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=None)
        elif isinstance(default, tuple):
            parser.add_argument(flag, nargs="*", default=None)
        elif f.name in ("max_steps", "plot_every_steps"):
            parser.add_argument(flag, type=int, default=None)
        elif f.name in ("freeze_after_frac", "plot_every_seconds", "checkpoint_every_seconds"):
            parser.add_argument(flag, type=float, default=None)
        else:
            parser.add_argument(flag, type=type(default), default=None)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="astrotune-train",
        description="Train the Astrotune bijector. Every TrainConfig field is "
        "available as an option; defaults reproduce the reference notebook.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="output directory name under models/ (default: timestamp)",
    )
    parser.add_argument("--run-dir", type=Path, default=None, help="explicit output directory")
    parser.add_argument("--pairs-path", type=Path, default=None, help="pairs_and_ctrl_points.h5")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue the run in --run-dir/--run-name from its latest checkpoint. The "
        "saved configuration is reused; only plotting/checkpoint intervals may be changed.",
    )
    _add_config_args(parser)
    args = parser.parse_args(argv)

    overrides = {
        f.name: getattr(args, f.name)
        for f in dataclasses.fields(TrainConfig)
        if getattr(args, f.name) is not None
    }
    if "freeze_vars" in overrides:
        overrides["freeze_vars"] = tuple(overrides["freeze_vars"])

    if args.run_dir is None and args.run_name is None:
        if args.resume:
            parser.error("--resume needs --run-dir or --run-name")
        args.run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = args.run_dir or config.MODELS_DIR / args.run_name

    if args.resume:
        with open(run_dir / "config.json") as f:
            saved = _config_from_json(json.load(f)["config"])
        cfg = TrainConfig(**{**saved, **overrides})
    else:
        cfg = TrainConfig(**overrides)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_training(cfg, run_dir, args.pairs_path, resume=args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
