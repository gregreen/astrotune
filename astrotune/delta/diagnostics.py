"""How well calibrated are the quoted errors, in the harmonized space?

Every catalog entry of a star is a measurement of the same unknown value ``z``, so
the residual to look at is the entry's offset from that star's combined value,
measured in units of the variance that offset should have:

    d_k = y_k - z,     Cov(d_k) = S_k - A^-1,     A = sum_k S_k^-1,

where ``S_k = C_k`` (the propagated covariance) or ``S_k = C_k + diag(delta_k)``
once a delta model exists. The ``-A^-1`` term is what the star framing adds: ``z``
is itself estimated from the entries, so an entry's residual is smaller than its
own covariance by the covariance of the combined value. This module measures how
far the entries of a :mod:`star table <astrotune.delta.stars>` are from that:

``scale_ratio``
    robust standard deviation of ``z_p = d_p / sqrt(Cov(d)_pp)``; 1 if calibrated,
    > 1 if the quoted errors are too small.
``PIT``
    the star-level ``chi2 = sum_k d_k^T S_k^-1 d_k`` on ``sum_k n_k - n_shared``
    degrees of freedom (``n_shared`` is the number of parameters that at least one
    entry measures); its chi-squared CDF value is uniform on [0, 1] if calibrated.
    One number per star, so the stars are independent, which the pair-based version
    could not offer: it had ``K (K-1) / 2`` correlated statistics per ``K - 1`` dof.
tails
    the fraction of stars with a chi-squared survival probability below the
    thresholds, and of ``|z| > 3`` (0.27 % if Gaussian).
``excess``
    per catalog and parameter, the variance the catalog needs on top of its quoted
    error. It is *negative* where the quoted errors are too large. Estimated from the
    stars in exactly two catalogs, where ``Var(y_1 - y_2) = S_1 + S_2`` holds
    exactly, so ``median(d^2) - median(S_1 + S_2)`` is an unbiased estimate of the
    sum of the two catalogs' excesses; solving those equations by weighted least
    squares is the classical three-cornered hat. Stars with more entries are left to
    the likelihood (:func:`three_cornered_hat` says why). The maximum-likelihood
    version of the same quantity is the ``constant per catalog`` model that
    ``astrotune-delta-train`` fits.
correlations
    of ``z`` among the four parameters (core of the distribution), against what the
    residual covariance itself predicts.
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

from .stars import N_PARAMS, StarTable, block_indices, get_star_table

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


def _segments(seg: np.ndarray) -> np.ndarray:
    """Start offset of every run of equal ``seg`` (a block holds whole stars)."""
    return np.r_[0, np.flatnonzero(np.diff(seg)) + 1]


def entry_stats(stars: StarTable, delta_fn: Optional[DeltaFn] = None,
                chunk: int = 200_000) -> Dict[str, np.ndarray]:
    """Residuals of every entry against its star's combined value.

    Returns, per entry, ``d``, ``sd`` (the residual standard deviation), ``z`` (NaN
    where a parameter is not measured), ``chi2``, ``dof``, ``c`` and ``x``; per
    star, ``star_chi2``, ``star_dof`` and ``star_n``; and the summed predicted
    correlation matrix (``corr_pred``) of the residual covariance.
    """
    n_ent = int(stars["off"][-1])
    n_star = len(stars["id"])
    ii = np.arange(N_PARAMS)
    d = np.full((n_ent, N_PARAMS), np.nan)
    sd = np.full((n_ent, N_PARAMS), np.nan)
    chi2 = np.zeros(n_ent)
    dof = np.zeros(n_ent, dtype="i4")
    star_chi2 = np.zeros(n_star)
    star_dof = np.zeros(n_star, dtype="i4")
    corr_sum, corr_n = np.zeros((N_PARAMS, N_PARAMS)), 0

    for idx, seg, n in block_indices(stars, None, chunk):
        m = stars["m"][idx]
        mf = m.astype("f8")
        S = stars["C"][idx].astype("f8")
        if delta_fn is not None:
            S[:, ii, ii] += np.exp(delta_fn(stars["x"][idx], stars["ls"][idx], stars["c"][idx]).astype("f8"))
        Sinv = np.linalg.inv(S)
        starts = _segments(seg)

        # Per-star precision sum A, precision-weighted mean y and its covariance.
        A = np.add.reduceat(Sinv.reshape(len(idx), N_PARAMS * N_PARAMS), starts,
                            axis=0).reshape(n, N_PARAMS, N_PARAMS)
        b = np.add.reduceat(np.einsum("nij,nj->ni", Sinv, stars["y"][idx].astype("f8")), starts, axis=0)
        Ainv = np.linalg.inv(A)
        z = np.einsum("nij,nj->ni", Ainv, b)

        dd = stars["y"][idx].astype("f8") - z[seg]
        resid_var = np.diagonal(S, axis1=1, axis2=2) - np.diagonal(Ainv, axis1=1, axis2=2)[seg]
        quad = np.einsum("ni,ni->n", dd, np.einsum("nij,nj->ni", Sinv, dd))

        d[idx] = np.where(m, dd, np.nan)
        sd[idx] = np.sqrt(np.where(m, resid_var, np.nan))
        chi2[idx] = quad
        dof[idx] = m.sum(1)
        star_chi2[seg[0] + np.arange(n)] = np.add.reduceat(quad, starts)
        shared = np.add.reduceat(mf, starts, axis=0) > 0  # parameter some entry measures
        star_dof[seg[0] + np.arange(n)] = np.add.reduceat(m.sum(1), starts) - shared.sum(1)

        full = m.all(1) & np.isfinite(resid_var).all(1)
        if full.any():
            resid_cov = S[full] - Ainv[seg[full]]
            s = np.sqrt(np.maximum(np.diagonal(resid_cov, axis1=1, axis2=2), 1e-300))
            corr_sum += (resid_cov / s[:, :, None] / s[:, None, :]).sum(0)
            corr_n += int(full.sum())

    return dict(
        d=d, sd=sd, z=d / sd, chi2=chi2, dof=dof,
        sf=stats.chi2.sf(np.maximum(star_chi2, 0), np.maximum(star_dof, 1)),
        c=stars["c"], x=stars["x"], m=stars["m"],
        star_chi2=star_chi2, star_dof=star_dof, star_n=stars["n_entry"].astype("i4"),
        corr_pred=corr_sum / max(corr_n, 1),
    )


def three_cornered_hat(stars: StarTable, n_cat: int, min_pairs: int = 200) -> np.ndarray:
    """Per-catalog excess variance from the stars in exactly two catalogs.

    For such a star ``Var(y_1 - y_2) = S_1 + S_2`` exactly -- the difference is a
    difference of two independent measurements, with no estimated common value in
    it -- so ``median(d^2) - median(S_1 + S_2)`` estimates ``D_1 + D_2`` per
    parameter, and the ``D`` follow by weighted least squares over the catalog
    pairs. Catalogs not in any such star are NaN, and values may be negative.

    Stars with ``K >= 3`` entries are deliberately *not* used: their entry residual
    carries ``-A^-1``, the covariance of the star's combined value, and subtracting a
    median of that makes the naive pair formula biased low (with three equally
    weighted entries and equal quoted errors it recovers a third of the true extra
    variance). Those stars are what the likelihood is for.
    """
    out = np.full((n_cat, N_PARAMS), np.nan)
    off = stars["off"][:-1][stars["n_entry"] == 2]
    if len(off) == 0:
        return out
    c1 = stars["c"][off].astype("i8")
    key = c1 * n_cat + stars["c"][off + 1]
    order = np.argsort(key, kind="stable")
    ukey, start = np.unique(key[order], return_index=True)
    stop = np.r_[start[1:], len(order)]
    for p in range(N_PARAMS):
        rows = []
        for k, a, b in zip(ukey, start, stop):
            idx = off[order[a:b]]
            ok = stars["m"][idx, p] & stars["m"][idx + 1, p]
            if ok.sum() < min_pairs:
                continue
            i = idx[ok]
            d = (stars["y"][i, p].astype("f8") - stars["y"][i + 1, p].astype("f8"))
            Spp = (stars["C"][i, p, p].astype("f8") + stars["C"][i + 1, p, p].astype("f8"))
            rows.append((int(k // n_cat), int(k % n_cat),
                         robust_scale(d) ** 2 - float(np.median(Spp)), int(ok.sum())))
        if not rows:
            continue
        A = np.zeros((len(rows), n_cat))
        y = np.zeros(len(rows))
        w = np.zeros(len(rows))
        for i, (j, k, e, n) in enumerate(rows):
            A[i, j] = A[i, k] = 1.0
            y[i], w[i] = e, np.sqrt(n)
        used = A.sum(0) > 0
        sol, *_ = np.linalg.lstsq(A[:, used] * w[:, None], y * w, rcond=None)
        out[used, p] = sol
    return out


def summarize(stars: StarTable, st: Dict[str, np.ndarray], n_cat: int,
              min_entries: int = 1000) -> dict:
    """All the calibration statistics in one dictionary (JSON-friendly after ``to_json``)."""
    z, d, sd = st["z"], st["d"], st["sd"]
    c, m = st["c"], st["m"]

    # Per catalog and parameter
    excess = three_cornered_hat(stars, n_cat, min_pairs=max(min_entries // 4, 20))
    per_catalog = []
    for k in range(n_cat):
        in_cat = c == k
        row = dict(catalog=k, n=int(in_cat.sum()), scale_ratio=np.full(N_PARAMS, np.nan),
                   beyond3=np.full(N_PARAMS, np.nan), excess=excess[k])
        for p in range(N_PARAMS):
            sel = in_cat & m[:, p]
            if sel.sum() < min_entries:
                continue
            row["scale_ratio"][p] = robust_scale(z[sel, p])
            row["beyond3"][p] = float(np.mean(np.abs(z[sel, p]) > 3))
        per_catalog.append(row)

    # Pooled, over stars (independent) and over entries (for the per-parameter widths)
    ok = st["star_dof"] > 0
    pit = stats.chi2.cdf(np.maximum(st["star_chi2"][ok], 0), st["star_dof"][ok])
    core = np.all(np.abs(z) < 3, axis=1) & m.all(1)
    corr_obs = np.corrcoef(z[core].T) if core.sum() > 100 else np.full((N_PARAMS,) * 2, np.nan)

    teff = st["x"][:, 0]
    edges = np.quantile(teff, np.linspace(0, 1, 9))
    which = np.clip(np.digitize(teff, edges[1:-1]), 0, 7)
    pooled = dict(
        n_stars=int(ok.sum()),
        n_entries=int(len(z)),
        median_pit=float(np.median(pit)),
        frac_sf=[float(np.mean(stats.chi2.sf(st["star_chi2"][ok], st["star_dof"][ok]) < t))
                 for t in SF_THRESHOLDS],
        beyond3=[float(np.nanmean(np.abs(z[m[:, p], p]) > 3)) for p in range(N_PARAMS)],
        scale_ratio=[robust_scale(z[m[:, p], p]) for p in range(N_PARAMS)],
        pit_hist=np.histogram(pit, bins=20, range=(0, 1))[0],
    )
    return dict(
        per_catalog=per_catalog, pooled=pooled,
        excess_variance=np.stack([r["excess"] for r in per_catalog], axis=0)
        if per_catalog else np.full((n_cat, N_PARAMS), np.nan),
        corr_observed=corr_obs, corr_predicted=st["corr_pred"],
        teff_bins=dict(
            edges=edges,
            center=np.array([np.median(teff[which == i]) for i in range(8)]),
            scale_ratio=np.array([[robust_scale(z[(which == i) & m[:, p], p]) for p in range(N_PARAMS)]
                                  for i in range(8)]),
        ),
        min_entries=min_entries,
    )


def to_json(obj):
    """Convert a result dictionary to something ``json.dump`` accepts."""
    if isinstance(obj, dict):
        return {k: to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_json(obj.tolist())
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def diagnose(stars: StarTable, n_cat: int, delta_fn: Optional[DeltaFn] = None,
             min_entries: int = 1000) -> dict:
    return summarize(stars, entry_stats(stars, delta_fn), n_cat, min_entries)


def format_summary(res: dict, labels: Sequence[str], n_worst: int = 12) -> str:
    p = res["pooled"]
    lines = [
        f"{p['n_stars']:,} stars, {p['n_entries']:,} entries. Pooled: median PIT "
        f"{p['median_pit']:.2f} (0.50 if calibrated); "
        + ", ".join(f"P(sf<{t:g}) = {f:.2%}" for t, f in zip(SF_THRESHOLDS, p["frac_sf"]))
        + f"  (nominal {', '.join(f'{t:.2%}' for t in SF_THRESHOLDS)})",
        "scale ratio (1 = calibrated):   " + "  ".join(f"{n} {r:.2f}" for n, r in zip(PARAM_NAMES, p["scale_ratio"])),
        "|z| > 3 (0.27% if Gaussian):   " + "  ".join(f"{n} {r:.2%}" for n, r in zip(PARAM_NAMES, p["beyond3"])),
        "",
        f"{'catalog':26s}" + "".join(f"{n:>9s}" for n in PARAM_NAMES)
        + "".join(f"{'ex ' + n:>12s}" for n in PARAM_NAMES),
    ]
    for row in res["per_catalog"]:
        lines.append(
            f"{labels[row['catalog']][:26]:26s}"
            + "".join(f"{r:9.2f}" for r in row["scale_ratio"])
            + "".join(f"{np.sqrt(max(e, 0.0)):12.3f}" for e in row["excess"])
        )
    worst = []
    for row in res["per_catalog"]:
        for q in range(N_PARAMS):
            r = row["scale_ratio"][q]
            if np.isfinite(r):
                worst.append((abs(np.log(r)), labels[row["catalog"]], PARAM_NAMES[q], r))
    worst.sort(reverse=True)
    lines.append("")
    lines.append(f"farthest from 1 ({n_worst} of {len(worst)}):")
    for _, lab, name, r in worst[:n_worst]:
        lines.append(f"  {lab[:26]:26s} {name:7s} {r:.2f}")
    return "\n".join(lines)


def _heatmap(ax, matrix, labels, title, fmt="{:.2f}"):
    """A catalog x parameter matrix as an image with the numbers written in."""
    v = np.asarray(matrix, dtype="f8")
    logv = np.log2(np.clip(v, 1e-3, 1e3))
    im = ax.imshow(logv, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
    ax.set_xticks(range(N_PARAMS), PARAM_NAMES, fontsize=8)
    ax.set_yticks(range(len(labels)), [lab[:20] for lab in labels], fontsize=6)
    ax.set_title(title, fontsize=9)
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            if np.isfinite(v[i, j]):
                ax.text(j, i, fmt.format(v[i, j]), ha="center", va="center", fontsize=6)
    return im


def plot_diagnostics(res: dict, labels: Sequence[str], out_dir: Path, suffix: str = "") -> Sequence[Path]:
    """Three figures: per catalog/parameter, distributions, and Teff dependence."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    ratios = np.array([r["scale_ratio"] for r in res["per_catalog"]])
    excess = np.array([np.sqrt(np.maximum(r["excess"], 0.0)) for r in res["per_catalog"]])
    p = res["pooled"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 0.22 * len(labels) + 2.4), layout="constrained")
    im = _heatmap(axes[0], ratios, labels, "scale ratio of the residual (1 = calibrated)")
    fig.colorbar(im, ax=axes[0], label=r"$\log_2$ (1/4 ... 4)")
    im = _heatmap(axes[1], excess, labels, r"excess $\sigma$ needed (harmonized units)")
    fig.colorbar(im, ax=axes[1], label=r"$\log_2$")
    axes[2].bar(np.arange(20) / 20 + 0.025, p["pit_hist"], width=0.05, color="C0")
    axes[2].axhline(p["n_stars"] * 0.05, color="C3", lw=1, label="uniform")
    axes[2].set(xlabel="PIT of the star chi2", ylabel="stars", title="star-level coverage")
    axes[2].legend(fontsize=7)
    fig.suptitle(f"calibration of the quoted errors{suffix}", fontsize=10)
    paths.append(out_dir / f"summary{suffix}.png")
    fig.savefig(paths[-1], dpi=110)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    st_teff = res["teff_bins"]
    for q, ax in enumerate(axes.flat):
        ax.plot(st_teff["center"], st_teff["scale_ratio"][:, q], "o-", color="C0")
        ax.axhline(1.0, color="C3", lw=1)
        ax.set(title=PARAM_NAMES[q], xlabel=r"harmonized $5\ln(T_{\rm eff}/5000\,{\rm K})$",
               ylabel="scale ratio")
        ax.invert_xaxis()
        ax.grid(alpha=0.2)
    fig.suptitle(f"scale ratio against Teff{suffix}", fontsize=10)
    paths.append(out_dir / f"teff{suffix}.png")
    fig.savefig(paths[-1], dpi=110)
    plt.close(fig)
    return paths


def write_diagnostics(res: dict, labels: Sequence[str], out_dir: Path, suffix: str = "") -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"summary{suffix}.json", "w") as f:
        json.dump(to_json(dict(res, labels=list(labels))), f, indent=1)
    (out_dir / f"summary{suffix}.txt").write_text(format_summary(res, labels) + "\n")
    plot_diagnostics(res, labels, out_dir, suffix)


def main(argv: Optional[Sequence[str]] = None) -> int:
    from .. import registry

    parser = argparse.ArgumentParser(
        prog="astrotune-delta-diagnose",
        description="Calibration of the quoted errors, from every observation of every star.",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None, help="default: <run-dir>/delta")
    parser.add_argument("--min-entries", type=int, default=1000)
    parser.add_argument("--rebuild", action="store_true", help="rebuild the star table")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    out_dir = args.out_dir or args.run_dir / "delta"
    stars, keys = get_star_table(args.run_dir, out_dir, args.rebuild)
    labels = [registry.get_spec(k).label for k in keys]
    res = diagnose(stars, len(keys), None, args.min_entries)
    write_diagnostics(res, labels, out_dir / "diagnostics")
    print(format_summary(res, labels))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
