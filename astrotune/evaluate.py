"""Do the catalogs agree better after mapping to the harmonized space?

For stars observed in two catalogs ``j < k`` we compare, parameter by
parameter, the difference between the catalogs' values before the
transformation, ``x_j - x_k``, and after it, ``f_j(x_j) - f_k(x_k)``. Only
held-out (validation) pairs are used. For each catalog pair and parameter we
report the median offset (bias) and a robust scatter (1.4826 x MAD). The
transformation has worked if both shrink.

    python -m astrotune.evaluate --run-dir models/my_run

Outputs go to ``<run-dir>/evaluation/``: ``agreement.json``,
``agreement_summary.png``, ``param_shifts.png`` and ``pairs/pair_JJ_KK.png``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from . import config, registry
from .training_plots import PARAM_LABELS, catalog_colors, catalog_samples

import os

if not os.environ.get("DISPLAY"):  # headless by default
    import matplotlib

    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

log = logging.getLogger(__name__)

PAIR_FIELDS = ("x0", "x1", "xs0", "xs1", "c0", "c1")
PARAM_SHORT = ("Teff", "[Fe/H]", "logg", "[a/Fe]")

#: Larger than any real uncertainty; marks a parameter that a catalog lacks
MISSING = 1e6


# --- Data ---------------------------------------------------------------------
def validation_pairs(pair_dict, val_frac: float, max_pairs: Optional[int] = None, seed: int = 0):
    """Held-out pairs: the first ``val_frac`` of the rows, as in training."""
    n_val = int(len(pair_dict["x0"]) * val_frac)
    idx = np.arange(n_val)
    if max_pairs is not None and n_val > max_pairs:
        idx = np.sort(np.random.default_rng(seed).choice(n_val, max_pairs, replace=False))
    return {k: pair_dict[k][idx] for k in PAIR_FIELDS}


def harmonize(to_harmonized: Callable, x, c, batch_size: int = 8192) -> np.ndarray:
    """Apply ``to_harmonized(x, c)`` to numpy arrays in batches."""
    import tensorflow as tf

    out = []
    for i in range(0, len(x), batch_size):
        xb = tf.constant(x[i : i + batch_size], dtype=tf.float32)
        cb = tf.constant(c[i : i + batch_size], dtype=tf.int32)
        out.append(np.asarray(to_harmonized(xb, cb)))
    return np.concatenate(out, axis=0) if out else np.zeros_like(x)


def orient_pairs(pairs, to_harmonized: Callable) -> Dict[str, np.ndarray]:
    """Harmonize both sides of each pair and order them so that ``j < k``."""
    y0 = harmonize(to_harmonized, pairs["x0"], pairs["c0"])
    y1 = harmonize(to_harmonized, pairs["x1"], pairs["c1"])
    swap = (pairs["c0"] > pairs["c1"])[:, None]
    pick = lambda a, b: np.where(swap, b, a)
    valid = (pairs["xs0"] < MISSING) & (pairs["xs1"] < MISSING)
    valid &= np.isfinite(y0) & np.isfinite(y1)
    return dict(
        j=np.minimum(pairs["c0"], pairs["c1"]),
        k=np.maximum(pairs["c0"], pairs["c1"]),
        xj=pick(pairs["x0"], pairs["x1"]),
        xk=pick(pairs["x1"], pairs["x0"]),
        yj=pick(y0, y1),
        yk=pick(y1, y0),
        valid=valid,
    )


def observation_pairs(
    obs_dict, *, max_stars: int = 16384, seed: int = 0, stars=None
) -> Dict[str, np.ndarray]:
    """A synthetic pair table built from the observations of a sample of stars.

    The latent-y training data have no pairs, but the agreement statistics are
    defined per catalog pair. For every star that at least two catalogs observe
    we take the *first* observation of each catalog and form all catalog pairs
    (``j < k``), producing the :data:`PAIR_FIELDS` layout that
    :func:`orient_pairs` and :func:`agreement_table` consume.

    ``stars`` restricts the sample (e.g. to the validation stars); at most
    ``max_stars`` of them are used, drawn with ``seed``.
    """
    star, cat = obs_dict["star"], obs_dict["c"]
    n_stars = int(obs_dict["n_stars"])
    pool = np.arange(n_stars) if stars is None else np.asarray(stars)
    if len(pool) > max_stars:
        pool = np.sort(
            np.random.default_rng(seed).choice(pool, max_stars, replace=False)
        )

    # The first observation of every (star, catalog). The rows are sorted by id
    # and then by catalog, so a catalog's repeats are adjacent.
    first = np.ones(len(star), dtype=bool)
    first[1:] = (star[1:] != star[:-1]) | (cat[1:] != cat[:-1])
    rows = np.flatnonzero(first)
    observed = star[rows]
    lo = np.searchsorted(observed, pool, side="left")
    hi = np.searchsorted(observed, pool, side="right")

    i0, i1 = [], []
    for a, b in zip(lo, hi):
        if b - a < 2:
            continue
        for p, q in itertools.combinations(range(a, b), 2):
            i0.append(rows[p])
            i1.append(rows[q])
    i0 = np.asarray(i0, dtype="i8")
    i1 = np.asarray(i1, dtype="i8")
    x, xs = obs_dict["x"], obs_dict["xs"]
    return dict(
        x0=x[i0], x1=x[i1], xs0=xs[i0], xs1=xs[i1],
        c0=cat[i0].astype("i4"), c1=cat[i1].astype("i4"),
    )


# --- Statistics -----------------------------------------------------------------
def _median_and_sigma(d):
    med = np.median(d)
    return float(med), float(1.4826 * np.median(np.abs(d - med)))


def agreement_table(o, min_pairs: int = 100) -> List[dict]:
    """Per catalog pair and parameter: offset and scatter, before and after."""
    n_cat = len(registry.CATALOGS)
    key = o["j"].astype("i8") * n_cat + o["k"]
    rows = []
    for g in np.unique(key):
        in_pair = key == g
        for p in range(o["xj"].shape[1]):
            m = in_pair & o["valid"][:, p]
            n = int(m.sum())
            if n < min_pairs:
                continue
            bias_o, sig_o = _median_and_sigma(o["xj"][m, p] - o["xk"][m, p])
            bias_h, sig_h = _median_and_sigma(o["yj"][m, p] - o["yk"][m, p])
            rows.append(
                dict(
                    j=int(g // n_cat), k=int(g % n_cat), param=p, n=n,
                    bias_orig=bias_o, sigma_orig=sig_o,
                    bias_harm=bias_h, sigma_harm=sig_h,
                )
            )
    return rows


def summarize(rows: Sequence[dict], n_params: int = 4) -> List[dict]:
    """Average over catalog pairs (each pair counts once), for each parameter."""
    out = []
    for p in range(n_params):
        r = [x for x in rows if x["param"] == p]
        if not r:
            out.append(dict(param=p, n_pairs=0))
            continue
        a = lambda key: np.array([x[key] for x in r])
        out.append(
            dict(
                param=p,
                n_pairs=len(r),
                abs_bias_orig=float(np.mean(np.abs(a("bias_orig")))),
                abs_bias_harm=float(np.mean(np.abs(a("bias_harm")))),
                sigma_orig=float(np.mean(a("sigma_orig"))),
                sigma_harm=float(np.mean(a("sigma_harm"))),
                frac_bias_improved=float(np.mean(np.abs(a("bias_harm")) < np.abs(a("bias_orig")))),
                frac_sigma_improved=float(np.mean(a("sigma_harm") < a("sigma_orig"))),
            )
        )
    return out


def format_summary(summary: Sequence[dict]) -> str:
    lines = [
        "param    pairs   |bias| orig->harm      sigma orig->harm     improved (bias/sigma)"
    ]
    for s in summary:
        if not s["n_pairs"]:
            continue
        lines.append(
            f"{PARAM_SHORT[s['param']]:<8} {s['n_pairs']:>5}   "
            f"{s['abs_bias_orig']:.4f} -> {s['abs_bias_harm']:.4f}    "
            f"{s['sigma_orig']:.4f} -> {s['sigma_harm']:.4f}      "
            f"{s['frac_bias_improved']:.0%} / {s['frac_sigma_improved']:.0%}"
        )
    return "\n".join(lines)


def quick_agreement(to_harmonized: Callable, pairs, min_pairs: int = 50) -> List[dict]:
    """Cheap summary used to monitor agreement during training."""
    rows = agreement_table(orient_pairs(pairs, to_harmonized), min_pairs=min_pairs)
    return summarize(rows)


def catalog_param_valid(pair_dict, n_cat: int) -> np.ndarray:
    """``(n_cat, n_params)`` boolean: does the catalog measure this parameter?"""
    n_par = pair_dict["x0"].shape[1]
    valid = np.zeros((n_cat, n_par), dtype=bool)
    for c in range(n_cat):
        for side in "01":
            rows = pair_dict["c" + side] == c
            if rows.any():
                valid[c] |= (pair_dict["xs" + side][rows] < MISSING).any(axis=0)
    return valid


# --- Plots ----------------------------------------------------------------------
def plot_pair(o, j: int, k: int, labels, path: Path) -> None:
    """For one catalog pair: x_j vs x_k (left) and y_j vs y_k (right), per parameter."""
    sel = (o["j"] == j) & (o["k"] == k)
    fig, axes = plt.subplots(4, 2, figsize=(7, 13), layout="constrained")
    for p in range(4):
        m = sel & o["valid"][:, p]
        for col, (a, b, title) in enumerate(
            ((o["xj"], o["xk"], "catalog"), (o["yj"], o["yk"], "harmonized"))
        ):
            ax = axes[p, col]
            if m.sum() < 20:
                ax.text(0.5, 0.5, "no data", ha="center", transform=ax.transAxes)
                continue
            u, v = a[m, p], b[m, p]
            lo, hi = np.percentile(np.concatenate([u, v]), [0.5, 99.5])
            pad = 0.05 * (hi - lo)
            lim = (lo - pad, hi + pad)
            ax.hexbin(u, v, gridsize=50, extent=lim + lim, bins="log", mincnt=1, cmap="viridis")
            ax.plot(lim, lim, "w--", lw=0.8, alpha=0.8)
            d = u - v
            med = np.median(d)
            sig = 1.4826 * np.median(np.abs(d - med))
            ax.text(
                0.04, 0.96, f"offset {med:+.3f}\nscatter {sig:.3f}",
                transform=ax.transAxes, va="top", fontsize=7, color="w",
            )
            ax.set_xlim(lim)
            ax.set_ylim(lim)
            ax.set_aspect("equal")
            ax.set_xlabel(f"{PARAM_LABELS[p]} ({labels[j]})", fontsize=7)
            ax.set_ylabel(f"{PARAM_LABELS[p]} ({labels[k]})", fontsize=7)
            ax.tick_params(labelsize=7)
            if p == 0:
                ax.set_title(title, fontsize=9)
    fig.suptitle(f"{labels[j]}  vs  {labels[k]}", fontsize=10)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_summary(rows: Sequence[dict], summary: Sequence[dict], path: Path) -> None:
    """Each point is one catalog pair; points below the diagonal improved."""
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.5), layout="constrained")
    for p in range(4):
        r = [x for x in rows if x["param"] == p]
        for row, (key, title) in enumerate((("bias", "|offset|"), ("sigma", "scatter"))):
            ax = axes[row, p]
            if not r:
                continue
            a = np.array([abs(x[key + "_orig"]) for x in r])
            b = np.array([abs(x[key + "_harm"]) for x in r])
            n = np.array([x["n"] for x in r])
            sc = ax.scatter(a, b, c=np.log10(n), s=14, cmap="plasma")
            lo = max(min(a.min(), b.min()), 1e-4)
            hi = max(a.max(), b.max()) * 1.5
            ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            s = summary[p]
            frac = s["frac_bias_improved" if key == "bias" else "frac_sigma_improved"]
            ax.set_title(f"{PARAM_SHORT[p]}: {title}  ({frac:.0%} improved)", fontsize=9)
            ax.set_xlabel("catalogs", fontsize=8)
            ax.set_ylabel("harmonized", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.2)
    fig.colorbar(sc, ax=axes, label="log10(# pairs)", shrink=0.6)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_param_shifts(
    to_harmonized: Callable, ctrl_dict, valid: np.ndarray, labels, path: Path, n_bins: int = 15
) -> None:
    """Shift applied to each parameter, ``f(x) - x``, as a function of ``x``."""
    n_cat = len(labels)
    colors = catalog_colors(n_cat)
    samples = catalog_samples(ctrl_dict, n_cat)
    fig, axes = plt.subplots(2, 2, figsize=(9, 8), layout="constrained")
    for c, x in enumerate(samples):
        if len(x) < 200:
            continue
        y = harmonize(to_harmonized, x, np.full(len(x), c, dtype="i4"))
        for p, ax in enumerate(axes.flat):
            if not valid[c, p]:
                continue
            edges = np.quantile(x[:, p], np.linspace(0, 1, n_bins + 1))
            which = np.clip(np.digitize(x[:, p], edges[1:-1]), 0, n_bins - 1)
            xc, dy = [], []
            for b in range(n_bins):
                s = which == b
                if s.sum() >= 10:
                    xc.append(np.median(x[s, p]))
                    dy.append(np.median(y[s, p] - x[s, p]))
            ax.plot(xc, dy, color=colors[c], label=labels[c], lw=1.2)
    for p, ax in enumerate(axes.flat):
        ax.axhline(0, c="k", lw=0.6, alpha=0.5)
        ax.set_xlabel(PARAM_LABELS[p])
        ax.set_ylabel("harmonized - catalog")
        ax.grid(True, alpha=0.2)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
    axes.flat[0].legend(fontsize=5, ncol=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


# --- Driver ---------------------------------------------------------------------
def evaluate_bijector(
    to_harmonized: Callable,
    pair_dict,
    ctrl_dict,
    out_dir: Path,
    val_frac: float = 1 / 8,
    max_pairs: int = 200_000,
    n_pair_plots: int = 8,
    min_pairs: int = 100,
) -> List[dict]:
    """Evaluate on held-out pairs; write tables and figures to ``out_dir``."""
    out_dir = Path(out_dir)
    labels = registry.labels()[: int(ctrl_dict["c_control"].max()) + 1]
    pairs = validation_pairs(pair_dict, val_frac, max_pairs)
    log.info("Evaluating agreement on %d held-out pairs", len(pairs["x0"]))

    o = orient_pairs(pairs, to_harmonized)
    rows = agreement_table(o, min_pairs=min_pairs)
    summary = summarize(rows)
    log.info("Catalog agreement, averaged over catalog pairs:\n%s", format_summary(summary))

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "agreement.json", "w") as f:
        json.dump({"summary": summary, "pairs": rows, "labels": list(labels)}, f, indent=1)

    plot_summary(rows, summary, out_dir / "agreement_summary.png")

    n_cat = len(labels)
    counts = {}
    for r in rows:
        counts[(r["j"], r["k"])] = max(counts.get((r["j"], r["k"]), 0), r["n"])
    for j, k in sorted(counts, key=counts.get, reverse=True)[:n_pair_plots]:
        plot_pair(o, j, k, labels, out_dir / "pairs" / f"pair_{j:02d}_{k:02d}.png")

    valid = catalog_param_valid(pair_dict, n_cat)
    plot_param_shifts(to_harmonized, ctrl_dict, valid, labels, out_dir / "param_shifts.png")
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    import tensorflow as tf

    from .model.io import enable_memory_growth
    from .training_data import load_pairs

    enable_memory_growth()

    parser = argparse.ArgumentParser(
        prog="astrotune-evaluate",
        description="Compare catalog agreement before and after harmonization.",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--pairs-path", type=Path, default=None)
    parser.add_argument("--max-pairs", type=int, default=200_000)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    val_frac = 1 / 8
    cfg_path = args.run_dir / "config.json"
    if cfg_path.exists():
        val_frac = json.load(open(cfg_path))["config"].get("val_frac", val_frac)

    pair_dict, ctrl_dict = load_pairs(args.pairs_path)
    model = tf.saved_model.load(str(args.run_dir / "bijector"))
    evaluate_bijector(
        model.to_harmonized, pair_dict, ctrl_dict, args.run_dir / "evaluation",
        val_frac=val_frac, max_pairs=args.max_pairs,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
