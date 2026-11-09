"""Train and evaluate the delta model on pairs of harmonized estimates.

    astrotune-delta-train --run-dir models/my_run

needs the cache written by ``astrotune-apply --run-dir models/my_run``. It

1. builds (or loads) the pair table (:mod:`astrotune.delta.pairs`) and splits it by star,
2. evaluates the baselines on the validation stars: no added variance, the 0.03
   floor that the merge uses, and one constant ``ln_delta`` per catalog and parameter,
3. trains the network with the Gaussian pair likelihood (+ a weak prior),
4. measures calibration with and without the learned variances, and writes
   everything to ``<run-dir>/delta/``:

    pair_table.npz               the pairs (cache)
    diagnostics/                 calibration of the quoted errors (no delta)
    delta_model.*, delta_model_config.json   the trained network
    evaluation/                  baselines, calibration after the fit, figures
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
from .pairs import PairTable, get_pair_table, split, subset

log = logging.getLogger(__name__)

BATCH_KEYS = ("xa", "xb", "lsa", "lsb", "ca", "cb", "ya", "yb", "Ca", "Cb", "m")


@dataclass
class DeltaConfig:
    hidden_size: int = 64
    n_hidden: int = 3
    epochs: int = 20
    batch_size: int = 8192
    learning_rate: float = 2e-3
    prior_weight: float = 0.01  # weight of the prior on ln_delta, per measured parameter
    l2_weights: float = 1e-5
    clip_norm: float = 10.0
    seed: int = 0
    # The likelihood is very flat along the direction that trades variance between two
    # catalogs (a pair constrains only the sum), so this baseline needs many steps
    const_epochs: int = 20
    const_learning_rate: float = 0.1
    val_max: int = 300_000  # validation pairs used for the NLL (all are used for diagnostics)


def _batch(pairs: PairTable, idx) -> Dict[str, "tf.Tensor"]:
    import tensorflow as tf

    return {k: tf.constant(pairs[k][idx]) for k in BATCH_KEYS}


def mean_nll(pairs: PairTable, delta_fn, batch_size: int = 32768, max_pairs: Optional[int] = None) -> float:
    """Mean negative log likelihood per pair (all measured parameters) of ``pairs``."""
    from .model import pair_nll

    n = len(pairs["id"]) if max_pairs is None else min(max_pairs, len(pairs["id"]))
    tot = 0.0
    for i in range(0, n, batch_size):
        nll, _, _ = pair_nll(_batch(pairs, slice(i, min(i + batch_size, n))), delta_fn)
        tot += float(np.sum(nll.numpy()))
    return tot / n


def fit(model, train: PairTable, cfg: DeltaConfig, epochs: int, lr: float, val: Optional[PairTable] = None,
        desc: str = "training") -> list:
    """Minimize the pair NLL (+ prior) with Adam and a cosine schedule; returns the history."""
    import tensorflow as tf
    from tensorflow import keras

    from .model import l2_penalty, pair_nll, prior_penalty

    rng = np.random.default_rng(cfg.seed)
    n = len(train["id"])
    steps = n // cfg.batch_size
    sched = keras.optimizers.schedules.CosineDecay(lr, steps * epochs, alpha=0.01)
    opt = keras.optimizers.Adam(learning_rate=sched)
    variables = list(model.trainable_variables)

    @tf.function
    def step(batch):
        with tf.GradientTape() as tape:
            nll, la, lb = pair_nll(batch, model)
            data = tf.cast(tf.reduce_mean(nll), tf.float32)
            loss = data + cfg.prior_weight * prior_penalty(batch, la, lb) + cfg.l2_weights * l2_penalty(model)
        grads = tape.gradient(loss, variables)
        grads, _ = tf.clip_by_global_norm(grads, cfg.clip_norm)
        opt.apply_gradients(zip(grads, variables))
        return data

    history = []
    for epoch in range(epochs):
        perm = rng.permutation(n)
        losses = []
        for s in tqdm(range(steps), desc=f"{desc} {epoch + 1}/{epochs}", leave=False):
            losses.append(float(step(_batch(train, np.sort(perm[s * cfg.batch_size : (s + 1) * cfg.batch_size])))))
        rec = dict(epoch=epoch + 1, train_nll=float(np.mean(losses)))
        if val is not None:
            rec["val_nll"] = mean_nll(val, model, max_pairs=cfg.val_max)
        history.append(rec)
        log.info("%s epoch %d/%d: %s", desc, epoch + 1, epochs,
                 ", ".join(f"{k} {v:.4f}" for k, v in rec.items() if k != "epoch"))
    return history


def plot_learned_delta(model, val: PairTable, labels: Sequence[str], path: Path) -> Path:
    """Median sigma_delta (harmonized units) in bins of the catalog's own Teff coordinate."""
    import matplotlib

    if not os.environ.get("DISPLAY"):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .model import numpy_delta_fn
    from ..training_plots import catalog_colors

    fn = numpy_delta_fn(model)
    x = np.concatenate([val["xa"], val["xb"]])
    ls = np.concatenate([val["lsa"], val["lsb"]])
    c = np.concatenate([val["ca"], val["cb"]])
    m = np.concatenate([val["m"], val["m"]])
    sig = np.exp(0.5 * fn(x, ls, c))
    colors = catalog_colors(len(labels))
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained", sharex=True)
    edges = np.quantile(x[:, 0], np.linspace(0, 1, 13))
    for p, ax in enumerate(axes.flat):
        for k, lab in enumerate(labels):
            sel = (c == k) & m[:, p]
            if sel.sum() < 2000:
                continue
            which = np.clip(np.digitize(x[sel, 0], edges[1:-1]), 0, 11)
            xc, yc = [], []
            for b in range(12):
                s = which == b
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
    head = (f"{'model':22s}{'val NLL/pair':>13s}{'median PIT':>11s}"
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
        rebuild: bool = False, min_pairs: int = 1000) -> dict:
    import tensorflow as tf

    from .model import ConstDelta, DeltaNetwork, fixed_delta, numpy_delta_fn, save_delta_model

    cfg = cfg or DeltaConfig()
    run_dir = Path(run_dir)
    out_dir = Path(out_dir) if out_dir else run_dir / "delta"
    ev_dir = out_dir / "evaluation"
    ev_dir.mkdir(parents=True, exist_ok=True)
    tf.random.set_seed(cfg.seed)

    pairs, keys = get_pair_table(run_dir, out_dir, rebuild)
    labels = [registry.get_spec(k).label for k in keys]
    n_cat = len(keys)
    train, val = split(pairs)
    del pairs
    log.info("%d training pairs, %d validation pairs", len(train["id"]), len(val["id"]))

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
        nll[name] = mean_nll(val, fn)
        if hasattr(fn, "trainable_variables"):
            np_fn = numpy_delta_fn(fn)
        elif name.startswith("none"):
            np_fn = None
        else:
            sig = 0.03
            np_fn = (lambda s: (lambda x, ls, c: np.full(x.shape, 2 * np.log(s), dtype="f4")))(sig)
        results[name] = diagnose(val, n_cat, np_fn, min_pairs)
        log.info("%s: validation NLL %.4f", name, nll[name])
    table = compare(results, nll)
    log.info("\n%s", table)
    (ev_dir / "comparison.txt").write_text(table + "\n")
    with open(ev_dir / "comparison.json", "w") as f:
        json.dump(to_json(dict(nll=nll, results={k: {kk: vv for kk, vv in v.items() if kk != "pairs"}
                                                   for k, v in results.items()},
                               history=hist, const_history=hist_const)), f, indent=1)
    plot_diagnostics(results["network"], labels, ev_dir, "_network")
    plot_diagnostics(results["none (propagated only)"], labels, ev_dir, "_none")
    plot_learned_delta(net, val, labels, ev_dir / "learned_delta.png")
    return dict(nll=nll, table=table)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="astrotune-delta-train",
        description="Learn extra uncertainties for the harmonized parameters (post hoc).",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None, help="default: <run-dir>/delta")
    parser.add_argument("--rebuild", action="store_true", help="rebuild the pair table")
    parser.add_argument("--min-pairs", type=int, default=1000)
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
    res = run(args.run_dir, args.out_dir, cfg, args.rebuild, args.min_pairs)
    print(res["table"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
