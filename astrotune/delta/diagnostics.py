"""How well calibrated are the quoted errors, in the harmonized space?

For two catalog entries of the same star, ``d = y_a - y_b`` should be zero-mean
Gaussian with covariance ``S = C_a + C_b`` (the propagated covariances; plus,
once a delta model exists, the learned diagonal terms). This module measures
how far the pairs in a :mod:`pair table <astrotune.delta.pairs>` are from that:

``scale_ratio``
    robust standard deviation of ``z_p = d_p / sqrt(S_pp)``; 1 if calibrated,
    > 1 if the quoted errors are too small.
``chi2 / PIT``
    ``chi2 = d^T S^-1 d`` over the parameters measured by both catalogs; its
    chi-squared CDF value (the PIT) is uniform on [0, 1] if calibrated.
tails
    the fraction of pairs with a chi-squared survival probability below
    thresholds, and of ``|z| > 3`` (0.27 % if Gaussian).
``excess``
    the "three-cornered hat": per catalog pair, the robust variance of ``d_p``
    minus the median ``S_pp`` estimates the sum of the two catalogs' excess
    variances; solving over all pairs gives one value per catalog. It is
    *negative* where the quoted errors are too large.
correlations
    of ``z`` among the four parameters (core of the distribution), against
    what ``S`` itself predicts.
Teff dependence
    the scale ratio in bins of harmonized Teff.

Run with ``astrotune-delta-diagnose --run-dir models/X``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence

import numpy as np
from scipy import stats

if not os.environ.get("DISPLAY"):  # headless by default
    import matplotlib

    matplotlib.use("Agg")

import matplotlib.pyplot as plt

from .pairs import N_PARAMS, PairTable, counts_per_pair

log = logging.getLogger(__name__)

PARAM_NAMES = ("Teff", "[Fe/H]", "logg", "[a/Fe]")
SF_THRESHOLDS = (0.05, 1e-3, 1e-6)

#: ``delta_fn(x, ln_sigma, catalog) -> (n, 4)`` ln of the added variance
DeltaFn = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


def robust_scale(v: np.ndarray) -> float:
    """1.4826 x the median absolute deviation, ignoring NaNs (nan if too few values)."""
    v = v[np.isfinite(v)]
    if len(v) < 5:
        return np.nan
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


def pair_stats(pairs: PairTable, delta_fn: Optional[DeltaFn] = None, chunk: int = 400_000) -> Dict[str, np.ndarray]:
    """Residuals of every pair against ``S = C_a + C_b (+ diag(delta_a + delta_b))``.

    Returns ``d`` and ``z`` (NaN where a parameter is not measured by both),
    ``chi2``, ``dof``, ``sf`` (chi-squared survival probability), ``pit`` and the
    summed predicted correlation matrix (``corr_pred``, over fully measured pairs).
    """
    n = len(pairs["id"])
    d = np.full((n, N_PARAMS), np.nan)
    sdiag = np.full((n, N_PARAMS), np.nan)
    chi2 = np.empty(n)
    ii = np.arange(N_PARAMS)
    corr_sum, corr_n = np.zeros((N_PARAMS, N_PARAMS)), 0
    for s in range(0, n, chunk):
        sl = slice(s, s + chunk)
        m = pairs["m"][sl]
        mf = m.astype("f8")
        S = pairs["Ca"][sl].astype("f8") + pairs["Cb"][sl].astype("f8")
        if delta_fn is not None:
            da = np.exp(delta_fn(pairs["xa"][sl], pairs["lsa"][sl], pairs["ca"][sl]))
            db = np.exp(delta_fn(pairs["xb"][sl], pairs["lsb"][sl], pairs["cb"][sl]))
            S[:, ii, ii] += da + db
        S = S * (mf[:, :, None] * mf[:, None, :])
        S[:, ii, ii] += 1.0 - mf  # unmeasured parameters: unit variance, zero residual
        dd = (pairs["ya"][sl].astype("f8") - pairs["yb"][sl].astype("f8")) * mf
        chi2[sl] = np.einsum("ni,ni->n", dd, np.linalg.solve(S, dd[..., None])[..., 0])
        sd = np.sqrt(S[:, ii, ii])
        d[sl] = np.where(m, dd, np.nan)
        sdiag[sl] = np.where(m, sd, np.nan)
        full = m.all(1)
        if full.any():
            corr_sum += (S[full] / sd[full][:, :, None] / sd[full][:, None, :]).sum(0)
            corr_n += int(full.sum())
    dof = pairs["m"].sum(1)
    return dict(
        d=d,
        sdiag=sdiag,
        z=d / sdiag,
        chi2=chi2,
        dof=dof,
        sf=stats.chi2.sf(chi2, dof),
        pit=stats.chi2.cdf(chi2, dof),
        corr_pred=corr_sum / max(corr_n, 1),
    )


def three_cornered_hat(rows: Sequence[dict], n_cat: int, param: int) -> np.ndarray:
    """Per-catalog excess variance from per-pair excesses (weighted least squares).

    Each pair ``(j, k)`` gives ``excess_jk = D_j + D_k``. Catalogs that are not
    connected to any pair are NaN. Values may be negative.
    """
    sel = [r for r in rows if np.isfinite(r["excess"][param])]
    out = np.full(n_cat, np.nan)
    if not sel:
        return out
    A = np.zeros((len(sel), n_cat))
    y = np.zeros(len(sel))
    w = np.zeros(len(sel))
    for i, r in enumerate(sel):
        A[i, r["j"]] = A[i, r["k"]] = 1.0
        y[i], w[i] = r["excess"][param], np.sqrt(r["n_meas"][param])
    used = A.sum(0) > 0
    sol, *_ = np.linalg.lstsq(A[:, used] * w[:, None], y * w, rcond=None)
    out[used] = sol
    return out


def summarize(pairs: PairTable, st: Dict[str, np.ndarray], n_cat: int, min_pairs: int = 1000) -> dict:
    """All the calibration statistics in one dictionary (JSON-friendly after ``to_json``)."""
    z, d, sd = st["z"], st["d"], st["sdiag"]
    key = pairs["ca"].astype("i8") * n_cat + pairs["cb"]
    order = np.argsort(key, kind="stable")
    ukey, start = np.unique(key[order], return_index=True)
    stop = np.r_[start[1:], len(order)]

    rows = []
    for k, a, b in zip(ukey, start, stop):
        idx = order[a:b]
        n_meas = np.isfinite(z[idx]).sum(0)
        row = dict(
            j=int(k // n_cat), k=int(k % n_cat), n=int(len(idx)), n_meas=n_meas,
            scale_ratio=np.full(N_PARAMS, np.nan), excess=np.full(N_PARAMS, np.nan),
            beyond3=np.full(N_PARAMS, np.nan),
            median_pit=float(np.median(st["pit"][idx])),
            frac_sf=[float(np.mean(st["sf"][idx] < t)) for t in SF_THRESHOLDS],
        )
        for p in range(N_PARAMS):
            if n_meas[p] >= min_pairs:
                row["scale_ratio"][p] = robust_scale(z[idx, p])
                row["excess"][p] = robust_scale(d[idx, p]) ** 2 - np.nanmedian(sd[idx, p] ** 2)
                row["beyond3"][p] = float(np.nanmean(np.abs(z[idx, p]) > 3))
        rows.append(row)

    rows_big = [r for r in rows if r["n"] >= min_pairs]
    excess = np.stack([three_cornered_hat(rows_big, n_cat, p) for p in range(N_PARAMS)], axis=1)

    core = np.all(np.abs(z) < 3, axis=1)  # NaN compares False, so only fully measured pairs
    corr_obs = np.corrcoef(z[core].T) if core.sum() > 100 else np.full((N_PARAMS,) * 2, np.nan)

    teff = 0.5 * (pairs["ya"][:, 0] + pairs["yb"][:, 0])
    edges = np.quantile(teff, np.linspace(0, 1, 9))
    which = np.clip(np.digitize(teff, edges[1:-1]), 0, 7)
    teff_bins = dict(
        edges=edges,
        center=np.array([np.median(teff[which == i]) for i in range(8)]),
        scale_ratio=np.array([[robust_scale(z[which == i, p]) for p in range(N_PARAMS)] for i in range(8)]),
    )
    pooled = dict(
        n_pairs=int(len(z)),
        frac_sf=[float(np.mean(st["sf"] < t)) for t in SF_THRESHOLDS],
        median_pit=float(np.median(st["pit"])),
        beyond3=[float(np.nanmean(np.abs(z[:, p]) > 3)) for p in range(N_PARAMS)],
        scale_ratio=[robust_scale(z[:, p]) for p in range(N_PARAMS)],
        pit_hist=np.histogram(st["pit"], bins=20, range=(0, 1))[0],
    )
    return dict(
        pairs=rows, pooled=pooled, excess_variance=excess, corr_observed=corr_obs,
        corr_predicted=st["corr_pred"], teff_bins=teff_bins, min_pairs=min_pairs,
    )


def to_json(obj):
    """Convert a result dictionary to something ``json.dump`` accepts."""
    if isinstance(obj, dict):
        return {k: to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_json(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def diagnose(pairs: PairTable, n_cat: int, delta_fn: Optional[DeltaFn] = None, min_pairs: int = 1000) -> dict:
    return summarize(pairs, pair_stats(pairs, delta_fn), n_cat, min_pairs)


def format_summary(res: dict, labels: Sequence[str], n_worst: int = 12) -> str:
    p = res["pooled"]
    lines = [
        f"{p['n_pairs']:,} pairs. Pooled: median PIT {p['median_pit']:.2f} (0.50 if calibrated); "
        + ", ".join(f"P(sf<{t:g}) = {f:.2%}" for t, f in zip(SF_THRESHOLDS, p["frac_sf"]))
        + f"  (nominal {', '.join(f'{t:.2%}' for t in SF_THRESHOLDS)})",
        "scale ratio (1 = calibrated):   " + "  ".join(f"{n} {r:.2f}" for n, r in zip(PARAM_NAMES, p["scale_ratio"])),
        "|z| > 3 (0.27% if Gaussian):   " + "  ".join(f"{n} {r:.2%}" for n, r in zip(PARAM_NAMES, p["beyond3"])),
        "",
        "Per-catalog excess sigma (signed sqrt of the three-cornered-hat variance; < 0: quoted errors too large)",
        f"{'catalog':26s}" + "".join(f"{n:>9s}" for n in PARAM_NAMES),
    ]
    ex = res["excess_variance"]
    for c, lab in enumerate(labels):
        if np.all(np.isnan(ex[c])):
            continue
        v = np.sign(ex[c]) * np.sqrt(np.abs(ex[c]))
        lines.append(f"{lab[:26]:26s}" + "".join(f"{x:+9.3f}" if np.isfinite(x) else f"{'':>9s}" for x in v))
    lines += ["", "Largest catalog pairs (n >= min_pairs): median PIT | scale ratio per parameter"]
    big = sorted([r for r in res["pairs"] if r["n"] >= res["min_pairs"]], key=lambda r: -r["n"])[:n_worst]
    for r in big:
        lines.append(
            f"{labels[r['j']][:20]:20s} x {labels[r['k']][:20]:20s} n={r['n']:7d}  {r['median_pit']:.2f} | "
            + " ".join(f"{x:5.2f}" if np.isfinite(x) else "  -  " for x in r["scale_ratio"])
        )
    lines += ["", "Correlation of z (core)   observed:", np.array2string(np.asarray(res["corr_observed"]), precision=2),
              "predicted by S:", np.array2string(np.asarray(res["corr_predicted"]), precision=2)]
    return "\n".join(str(x) for x in lines)


# --- Figures --------------------------------------------------------------------
def plot_diagnostics(res: dict, labels: Sequence[str], out_dir: Path, suffix: str = "") -> Sequence[Path]:
    from matplotlib.colors import TwoSlopeNorm

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_cat = len(labels)
    short = [l.replace("SDSS-V ", "S5 ").replace("LAMOST ", "L ")[:14] for l in labels]
    paths = []

    # 1. scale ratio per catalog pair and parameter
    M = np.full((N_PARAMS, n_cat, n_cat), np.nan)
    for r in res["pairs"]:
        M[:, r["j"], r["k"]] = M[:, r["k"], r["j"]] = r["scale_ratio"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 12), layout="constrained")
    for p, ax in enumerate(axes.flat):
        im = ax.imshow(np.log2(M[p]), cmap="RdBu_r", norm=TwoSlopeNorm(0, -2, 2))
        ax.set_xticks(range(n_cat), short, rotation=90, fontsize=6)
        ax.set_yticks(range(n_cat), short, fontsize=6)
        ax.set_title(f"{PARAM_NAMES[p]}: scale ratio (blue < 1: errors too large)", fontsize=9)
    cb = fig.colorbar(im, ax=axes, shrink=0.6)
    cb.set_ticks([-2, -1, 0, 1, 2], labels=["1/4", "1/2", "1", "2", "4"])
    paths.append(out_dir / f"scale_ratio{suffix}.png")
    fig.savefig(paths[-1], dpi=110)
    plt.close(fig)

    # 2. per-catalog excess
    ex = np.asarray(res["excess_variance"])
    sig = np.sign(ex) * np.sqrt(np.abs(ex))
    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True, layout="constrained")
    for p, ax in enumerate(axes):
        ax.bar(range(n_cat), sig[:, p], color=np.where(sig[:, p] < 0, "C0", "C3"))
        ax.axhline(0, c="k", lw=0.6)
        ax.set_ylabel(f"{PARAM_NAMES[p]}\nsigma_excess")
        ax.grid(True, alpha=0.2)
    axes[-1].set_xticks(range(n_cat), labels, rotation=75, ha="right", fontsize=7)
    axes[0].set_title("Excess error per catalog (three-cornered hat); blue: quoted errors too large")
    paths.append(out_dir / f"catalog_excess{suffix}.png")
    fig.savefig(paths[-1], dpi=110)
    plt.close(fig)

    # 3. tails, Teff dependence and correlations
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), layout="constrained")
    ax = axes[0, 0]
    h = np.asarray(res["pooled"]["pit_hist"], float)
    ax.bar(np.linspace(0.025, 0.975, 20), h / h.sum() * 20, width=0.05)
    ax.axhline(1, c="k", lw=0.8)
    ax.set(xlabel="PIT = chi2 CDF of d^T S^-1 d", ylabel="density (1 = calibrated)", title="all pairs")
    ax = axes[0, 1]
    tb = res["teff_bins"]
    for p in range(N_PARAMS):
        ax.plot(tb["center"], np.asarray(tb["scale_ratio"])[:, p], "o-", label=PARAM_NAMES[p])
    ax.axhline(1, c="k", lw=0.8)
    ax.set(xlabel=r"$5\ln(T_{\rm eff}/5000\,{\rm K})$ (harmonized)", ylabel="scale ratio")
    ax.invert_xaxis()
    ax.legend(fontsize=8)
    for ax, key, title in ((axes[1, 0], "corr_observed", "observed correlation of z (core)"),
                           (axes[1, 1], "corr_predicted", "correlation predicted by S")):
        im = ax.imshow(np.asarray(res[key]), vmin=-0.5, vmax=0.5, cmap="RdBu_r")
        ax.set_xticks(range(4), PARAM_NAMES, fontsize=8)
        ax.set_yticks(range(4), PARAM_NAMES, fontsize=8)
        ax.set_title(title, fontsize=9)
        for (i, j), v in np.ndenumerate(np.asarray(res[key])):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=axes[1, :], shrink=0.6)
    paths.append(out_dir / f"tails_teff_correlations{suffix}.png")
    fig.savefig(paths[-1], dpi=110)
    plt.close(fig)
    return paths


def write_diagnostics(res: dict, labels: Sequence[str], out_dir: Path, suffix: str = "") -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"diagnostics{suffix}.json", "w") as f:
        json.dump(to_json(dict(res, labels=list(labels))), f, indent=1)
    (out_dir / f"summary{suffix}.txt").write_text(format_summary(res, labels) + "\n")
    plot_diagnostics(res, labels, out_dir, suffix)


def main(argv: Optional[Sequence[str]] = None) -> int:
    from .. import registry
    from .pairs import get_pair_table

    parser = argparse.ArgumentParser(
        prog="astrotune-delta-diagnose",
        description="Calibration of the quoted errors, from pairs of harmonized estimates.",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None, help="default: <run-dir>/delta")
    parser.add_argument("--min-pairs", type=int, default=1000)
    parser.add_argument("--rebuild", action="store_true", help="rebuild the pair table")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    out_dir = args.out_dir or args.run_dir / "delta"
    pairs, keys = get_pair_table(args.run_dir, out_dir, args.rebuild)
    labels = [registry.get_spec(k).label for k in keys]
    res = diagnose(pairs, len(keys), None, args.min_pairs)
    write_diagnostics(res, labels, out_dir / "diagnostics")
    print(format_summary(res, labels))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
