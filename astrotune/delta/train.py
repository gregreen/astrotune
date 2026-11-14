"""Train and evaluate the delta model on stars.

    astrotune-delta-train --run-dir models/my_run

needs the cache written by ``astrotune-apply --run-dir models/my_run``. It

1. builds (or loads) the star table (:mod:`astrotune.delta.stars`) and splits it by star,
2. evaluates the baselines on the validation stars: no added variance, the 0.03
   floor that the merge uses, and one constant ``ln_delta`` per catalog and parameter,
3. trains the network with the star likelihood of :mod:`astrotune.delta.model`
   (the latent-``y`` likelihood, with each star's common value integrated out),
4. measures calibration with and without the learned variances, and writes
   everything to ``<run-dir>/delta/``:

    star_table.npz               the entries of every star (cache)
    diagnostics/                 calibration of the quoted errors (no delta)
    delta_model.*, delta_model_config.json   the trained network
    evaluation/                  baselines, calibration after the fit, figures

Every number in the loss and in the validation metric is **per star**, not per
pair, so a star in nine catalogs counts once.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
from tqdm.auto import tqdm

from .. import registry
from .diagnostics import PARAM_NAMES, SF_THRESHOLDS, diagnose, plot_diagnostics, to_json
from .stars import N_PARAMS, get_star_table, split

log = logging.getLogger(__name__)


@dataclass
class DeltaConfig:
    hidden_size: int = 64
    n_hidden: int = 3
    epochs: int = 20
    #: Entries per training block. Blocks are cut on star boundaries, so the number
    #: of entries varies by the entries of the last star in each block.
    batch_size: int = 8192
    learning_rate: float = 2e-3
    prior_weight: float = 0.01  # weight of the prior on ln_delta, per measured parameter
    l2_weights: float = 1e-5
    clip_norm: float = 10.0
    seed: int = 0
    # The likelihood is very flat along the direction that trades variance between two
    # catalogs (a star in two catalogs constrains only the sum), so this baseline
    # needs many steps.
    const_epochs: int = 20
    const_learning_rate: float = 0.1
    #: Validation stars used for the NLL (all of them are used for the diagnostics)
    val_max_stars: int = 100_000


def _spec(tf):
    """Input signature: one trace for every block, whatever its length."""
    return {
        "x": tf.TensorSpec([None, N_PARAMS], tf.float32),
        "ls": tf.TensorSpec([None, N_PARAMS], tf.float32),
        "c": tf.TensorSpec([None], tf.int32),
        "y": tf.TensorSpec([None, N_PARAMS], tf.float32),
        "C": tf.TensorSpec([None, N_PARAMS, N_PARAMS], tf.float32),
        "m": tf.TensorSpec([None, N_PARAMS], tf.bool),
        "seg": tf.TensorSpec([None], tf.int32),
    }


def mean_star_nll(stars, delta_fn, batch_size: int = 131072, max_stars: Optional[int] = None) -> float:
    """Mean negative log likelihood per star over ``stars`` (a proper likelihood)."""
    import tensorflow as tf

    from .model import make_batch, star_nll
    from .stars import block_indices

    if max_stars is not None and len(stars["id"]) > max_stars:
        stars = _first_stars(stars, max_stars)
    total, n = 0.0, 0
    for idx, seg, n_star in block_indices(stars, None, batch_size):
        nll, _ = star_nll(make_batch(stars, idx, seg), delta_fn)
        total += float(tf.reduce_sum(nll))
        n += n_star
    return total / max(n, 1)


def _first_stars(stars, n_star: int):
    keep = np.zeros(len(stars["id"]), dtype=bool)
    keep[:n_star] = True
    from .stars import subset

    return subset(stars, keep)


def fit(model, stars, cfg: DeltaConfig, epochs: int, lr: float, val=None,
        desc: str = "training") -> list:
    """Minimise the star NLL (+ prior) with Adam and a cosine schedule; returns the history."""
    import tensorflow as tf
    from tensorflow import keras

    from .model import l2_penalty, make_batch, prior_penalty, star_nll
    from .stars import block_indices

    rng = np.random.default_rng(cfg.seed)
    n_star = len(stars["id"])
    steps = max(1, int(np.ceil(stars["off"][-1] / cfg.batch_size)))
    sched = keras.optimizers.schedules.CosineDecay(lr, steps * epochs, alpha=0.01)
    opt = keras.optimizers.Adam(learning_rate=sched)
    variables = list(model.trainable_variables)

    @tf.function(input_signature=[_spec(tf)])
    def step(batch):
        with tf.GradientTape() as tape:
            nll, la = star_nll(batch, model)
            data = tf.reduce_mean(nll)
            prior = cfg.prior_weight * prior_penalty(batch, la)
            loss = data + tf.cast(prior, tf.float64) + tf.cast(cfg.l2_weights * l2_penalty(model), tf.float64)
        grads = tape.gradient(loss, variables)
        grads, _ = tf.clip_by_global_norm(grads, cfg.clip_norm)
        opt.apply_gradients(zip(grads, variables))
        return data

    history = []
    for epoch in range(epochs):
        perm = rng.permutation(n_star)
        losses = []
        for idx, seg, n in tqdm(block_indices(stars, perm, cfg.batch_size),
                                total=steps, desc=f"{desc} {epoch + 1}/{epochs}", leave=False):
            losses.append(float(step(make_batch(stars, idx, seg))))
        rec = dict(epoch=epoch + 1, train_nll=float(np.mean(losses)))
        if val is not None:
            rec["val_nll"] = mean_star_nll(val, model, max_stars=cfg.val_max_stars)
        history.append(rec)
        log.info("%s epoch %d/%d: %s", desc, epoch + 1, epochs,
                 ", ".join(f"{k} {v:.4f}" for k, v in rec.items() if k != "epoch"))
    return history


def plot_learned_delta(model, stars, labels: Sequence[str], path: Path) -> Path:
    """Median sigma_delta (harmonized units) in bins of the catalog's own Teff coordinate."""
    import matplotlib

    if not os.environ.get("DISPLAY"):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ..training_plots import catalog_colors
    from .model import numpy_delta_fn

    fn = numpy_delta_fn(model)
    x, ls, c, m = stars["x"], stars["ls"], stars["c"], stars["m"]
    sig = np.exp(0.5 * fn(x, ls, c))
    colors = catalog_colors(len(labels))
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained", sharex=True)
    edges = np.quantile(x[:, 0], np.linspace(0, 1, 13))
    which = np.clip(np.digitize(x[:, 0], edges[1:-1]), 0, 11)
    for p, ax in enumerate(axes.flat):
        for k, lab in enumerate(labels):
            sel = (c == k) & m[:, p]
            if sel.sum() < 2000:
                continue
            xc, yc = [], []
            for b in range(12):  # twelve Teff bins, not one pass per entry
                s = which[sel] == b
                if s.sum() >= 100:
                    xc.append(np.median(x[sel][s, 0]))
                    yc.append(np.median(sig[sel][s, p]))
            ax.plot(xc, yc, "-", color=colors[k], label=lab, lw=1.2)
        ax.set_yscale("log")
        ax.set(title=PARAM_NAMES[p], ylabel=r"median $\sigma_\delta$ (harmonized units)")
        ax.grid(True, alpha=0.2)
    for ax in axes[1]:
        ax.set_xlabel(r"catalog's own $5\ln(T_{\rm eff}/5000\,{\rm K})$")
        ax.invert_xaxis()
    axes[0, 0].legend(fontsize=5, ncol=2)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def compare(results: Dict[str, dict], nll: Dict[str, float]) -> str:
    """A table of the headline numbers for several covariance models."""
    head = (f"{'model':22s}{'val NLL/star':>13s}{'median PIT':>11s}"
            + "".join(f"{'sf<' + format(t, 'g'):>10s}" for t in SF_THRESHOLDS)
            + "".join(f"{n:>9s}" for n in PARAM_NAMES) + "   (scale ratio)")
    lines = [head]
    for name, r in results.items():
        p = r["pooled"]
        lines.append(
            f"{name:22s}{nll[name]:13.4f}{p['median_pit']:11.3f}"
            + "".join(f"{f:10.2%}" for f in p["frac_sf"])
            + "".join(f"{s:9.2f}" for s in p["scale_ratio"])
        )
    lines.append(f"{'(nominal)':22s}{'':13s}{0.5:11.3f}" + "".join(f"{t:10.2%}" for t in SF_THRESHOLDS))
    return "\n".join(lines)


def run(run_dir: Path, out_dir: Optional[Path] = None, cfg: Optional[DeltaConfig] = None,
        rebuild: bool = False, min_entries: int = 1000) -> dict:
    import tensorflow as tf

    from .model import ConstDelta, DeltaNetwork, fixed_delta, numpy_delta_fn, save_delta_model

    cfg = cfg or DeltaConfig()
    run_dir = Path(run_dir)
    out_dir = Path(out_dir) if out_dir else run_dir / "delta"
    ev_dir = out_dir / "evaluation"
    ev_dir.mkdir(parents=True, exist_ok=True)
    tf.random.set_seed(cfg.seed)

    stars, keys = get_star_table(run_dir, out_dir, rebuild)
    labels = [registry.get_spec(k).label for k in keys]
    n_cat = len(keys)
    train, val = split(stars)
    del stars
    log.info("%d training stars (%d entries), %d validation stars (%d entries)",
             len(train["id"]), int(train["off"][-1]), len(val["id"]), int(val["off"][-1]))

    # Baselines on the validation stars
    models = {
        "none (propagated only)": fixed_delta(0.0),
        "floor 0.03 (as merge)": fixed_delta(0.03),
    }
    const = ConstDelta(n_cat)
    hist_const = fit(const, train, cfg, cfg.const_epochs, cfg.const_learning_rate, val, desc="const")
    models["constant per catalog"] = const

    net = DeltaNetwork(n_cat, cfg.hidden_size, cfg.n_hidden)
    hist = fit(net, train, cfg, cfg.epochs, cfg.learning_rate, val, desc="network")
    models["network"] = net

    save_delta_model(out_dir, net, dict(
        n_cat=n_cat, hidden_size=cfg.hidden_size, n_hidden=cfg.n_hidden, keys=list(keys),
        config=dataclasses.asdict(cfg), val_history=hist,
    ))

    nll, results = {}, {}
    for name, fn in models.items():
        nll[name] = mean_star_nll(val, fn, max_stars=cfg.val_max_stars)
        np_fn = numpy_delta_fn(fn) if hasattr(fn, "trainable_variables") else _fixed_np(fn)
        results[name] = diagnose(val, n_cat, np_fn, min_entries)
        log.info("%s: validation NLL per star %.4f", name, nll[name])
    table = compare(results, nll)
    log.info("\n%s", table)
    (ev_dir / "comparison.txt").write_text(table + "\n")
    # The constant model is the maximum-likelihood three-cornered hat: one excess
    # variance per catalog and parameter that makes the entries agree.
    const_sigma = np.exp(0.5 * const.raw.numpy())
    np.savez(ev_dir / "const_delta.npz", sigma=const_sigma, keys=list(keys))
    with open(ev_dir / "comparison.json", "w") as f:
        json.dump(to_json(dict(nll=nll, results={k: {kk: vv for kk, vv in v.items() if kk != "entries"}
                                                   for k, v in results.items()},
                               const_sigma=const_sigma, history=hist, const_history=hist_const)), f, indent=1)
    plot_diagnostics(results["network"], labels, ev_dir, "_network")
    plot_diagnostics(results["none (propagated only)"], labels, ev_dir, "_none")
    plot_learned_delta(net, val, labels, ev_dir / "learned_delta.png")
    return dict(nll=nll, table=table)


def _fixed_np(fn):
    """A numpy delta function for the ``fixed_delta`` baselines."""

    def wrap(x, ls, c):
        import tensorflow as tf

        return fn(tf.constant(x), tf.constant(ls), tf.constant(c, tf.int32)).numpy()

    return wrap


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="astrotune-delta-train",
        description="Learn extra uncertainties for the harmonized parameters (post hoc).",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None, help="default: <run-dir>/delta")
    parser.add_argument("--rebuild", action="store_true", help="rebuild the star table")
    parser.add_argument("--min-entries", type=int, default=1000)
    parser.add_argument("--cpu", action="store_true", help="do not use the GPU")
    for f in dataclasses.fields(DeltaConfig):
        parser.add_argument("--" + f.name.replace("_", "-"), type=type(f.default), default=f.default)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    import tensorflow as tf

    if args.cpu:
        tf.config.set_visible_devices([], "GPU")
    else:
        from ..model.io import enable_memory_growth

        enable_memory_growth()
    cfg = DeltaConfig(**{f.name: getattr(args, f.name) for f in dataclasses.fields(DeltaConfig)})
    res = run(args.run_dir, args.out_dir, cfg, args.rebuild, args.min_entries)
    print(res["table"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
