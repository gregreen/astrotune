#!/usr/bin/env python
"""Isolate the SDSS-V (CLAM) stars the map moves the wrong way, and describe them.

    python scripts/clam_bad_stars.py --run-dir models/<run>

The selection is made in the *harmonized* [Fe/H]-[alpha/Fe] plane: the stars that
end up below the line through ``(0, -0.1)`` and ``(-1.0, +0.1)``, i.e. the
bottom-left corner relative to the bulk of the catalogs. The script reports

1. what fraction of those stars carry an [alpha/Fe] measurement, against the same
   fraction for all CLAM stars;
2. the number of CLAM stars shared with every other catalog, with the catalogs
   that lack [alpha/Fe] or log g marked;
3. for each shared catalog, a 2x2 figure -- raw and harmonized, [Fe/H] and
   [alpha/Fe] -- with equal aspect and one set of limits per row, so that the
   left and right panels can be compared directly.

"Raw" is the catalog's own value and "harmonized" is where the trained map sends
it, ``f_c(x)`` at the end of the flow; a star observed twice by one catalog has
the mean of its rows. Both columns use the same data, so any difference between
them is the map's doing.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from astrotune import config, registry
from astrotune.training_data import load_observations

log = logging.getLogger(__name__)

PARAM_LABELS = (
    r"$5\ln(T_\mathrm{eff}/5000\,\mathrm{K})$",
    r"$\left[\mathrm{Fe/H}\right]$",
    r"$\log g$",
    r"$\left[\alpha/\mathrm{Fe}\right]$",
)
FEH, ALPHA = 1, 3
#: Line through these two points in the harmonized plane; stars below it are "bad".
LINE_POINTS = ((0.0, -0.1), (-1.0, 0.1))
MISSING = 1e6


def line_alpha(feh, points=LINE_POINTS):
    """[alpha/Fe] of the selection line at ``feh``."""
    (x0, y0), (x1, y1) = points
    return y0 + (y1 - y0) * (np.asarray(feh) - x0) / (x1 - x0)


def map_rows(bij, x: np.ndarray, c: np.ndarray, batch_size: int) -> np.ndarray:
    """``f_c(x)`` at the end of the flow, for every row."""
    import tensorflow as tf

    out = np.empty_like(x)
    for lo in range(0, len(x), batch_size):
        hi = min(lo + batch_size, len(x))
        path = bij.return_path(
            tf.constant(x[lo:hi], dtype=tf.float32),
            tf.constant(c[lo:hi], dtype=tf.int32),
            2,
        ).numpy()
        out[lo:hi] = path[-1]
    return out


def per_key_means(key: np.ndarray, arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean of ``arr`` (and its count) per unique value of ``key``."""
    uk, inv = np.unique(key, return_inverse=True)
    counts = np.bincount(inv).astype("f8")
    out = np.empty((len(uk), arr.shape[1]), dtype="f4")
    for j in range(arr.shape[1]):
        out[:, j] = np.bincount(inv, weights=arr[:, j], minlength=len(uk)) / counts
    return uk, out, counts


def coverage(xs_cat: np.ndarray) -> Dict[str, str]:
    """``"all"``/``"some"``/``"none"`` for the parameters that can be missing."""
    out = {}
    n = len(xs_cat)
    for name, j in (("logg", 2), ("alpha", ALPHA)):
        frac = float((xs_cat[:, j] < MISSING).mean()) if n else 0.0
        out[name] = "all" if frac > 0.999 else ("none" if frac < 1e-6 else f"some {100*frac:.0f}%")
    return out


def scatter_text(a: np.ndarray, b: np.ndarray) -> str:
    """Correlation, median offset and robust scatter of ``a`` against ``b``."""
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 10:
        return "n<10"
    a, b = a[ok], b[ok]
    r = np.corrcoef(a, b)[0, 1] if a.std() > 0 and b.std() > 0 else np.nan
    d = a - b
    return (f"r={r:.2f}  med $\\Delta$={np.median(d):+.2f}\n"
            f"scatter(1.48 MAD)={1.4826*np.median(np.abs(d - np.median(d))):.3f}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="clam_bad_stars", description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--observations-path", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="default: <run-dir>/clam_bad")
    parser.add_argument("--clam-key", default="sdssv_clam")
    parser.add_argument("--min-pairs", type=int, default=200,
                        help="draw the pair figure for catalogs sharing this many CLAM stars")
    parser.add_argument("--max-stars", type=int, default=None,
                        help="subsample CLAM stars (default: all)")
    parser.add_argument("--batch-size", type=int, default=32768)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from astrotune.model.io import enable_memory_growth, load_bijector

    enable_memory_growth()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "clam_bad"
    out_dir.mkdir(parents=True, exist_ok=True)

    path = args.observations_path
    if path is None:
        import json
        with open(run_dir / "config.json") as f:
            path = config.PROJECT_ROOT / json.load(f)["observations_path"]
    obs, _ = load_observations(path, fix_unknown_errors=False)

    keys = registry.keys()
    labels = [registry.get_spec(k).label for k in keys]
    clam_cat = keys.index(args.clam_key)
    c, x, xs, star = obs["c"], obs["x"], obs["xs"], obs["star"]
    n_cat = len(keys)

    clam_rows = np.flatnonzero(c == clam_cat)
    clam_stars = np.unique(star[clam_rows])
    if args.max_stars and len(clam_stars) > args.max_stars:
        clam_stars = np.sort(np.random.default_rng(0).choice(
            clam_stars, args.max_stars, replace=False))
    log.info("CLAM: %d rows, %d stars", len(clam_rows), len(clam_stars))

    # every observation of those stars, then one mean per (star, catalog)
    rows = np.flatnonzero(np.isin(star, clam_stars))
    log.info("%d observations of those stars; integrating", len(rows))
    bij = load_bijector(run_dir)
    mapped = map_rows(bij, x[rows], c[rows], args.batch_size)

    key = star[rows].astype("i8") * n_cat + c[rows]
    uk, x_med, _ = per_key_means(key, x[rows])
    _, m_med, _ = per_key_means(key, mapped)
    _, s_med, _ = per_key_means(key, xs[rows])  # a mean of sigmas is only used as a mask

    def lookup(cat: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Per CLAM star: the (raw, mapped, sigma) of catalog ``cat``, or NaN."""
        want = clam_stars * n_cat + cat
        j = np.searchsorted(uk, want)
        j = np.clip(j, 0, len(uk) - 1)
        found = uk[j] == want
        xj = np.where(found[:, None], x_med[j], np.nan)
        mj = np.where(found[:, None], m_med[j], np.nan)
        sj = np.where(found[:, None], s_med[j], np.nan)
        return xj, mj, sj, found

    x_clam, m_clam, s_clam, _ = lookup(clam_cat)
    has_alpha = s_clam[:, ALPHA] < MISSING

    bad = m_clam[:, ALPHA] < line_alpha(m_clam[:, FEH])
    log.info("bad (below the selection line): %d of %d CLAM stars (%.1f%%)",
             int(bad.sum()), len(bad), 100 * bad.mean())

    # --- 1. coverage of the bad subset ------------------------------------
    lines = []
    lines.append("1. [alpha/Fe] coverage")
    lines.append(f"   bad stars:  {int(bad.sum()):7d}/{len(bad):7d}  "
                 f"{100*bad.mean():5.1f}% of CLAM")
    lines.append(f"     with [alpha/Fe]: {100*has_alpha[bad].mean():5.1f}%"
                 f"  ({int(has_alpha[bad].sum())} of {int(bad.sum())})")
    lines.append(f"   all CLAM stars: with [alpha/Fe]: {100*has_alpha.mean():5.1f}%"
                 f"  ({int(has_alpha.sum())} of {len(has_alpha)})")
    if has_alpha.any():
        lines.append(f"   -> {has_alpha[bad].mean()/max(has_alpha.mean(), 1e-9):.2f}x "
                     "the rate among all CLAM stars")
        lines.append(f"   [Fe/H] of the bad stars (harmonized): median "
                     f"{np.median(m_clam[bad, FEH]):+.2f}; all CLAM "
                     f"{np.median(m_clam[:, FEH]):+.2f}")
        lines.append(f"   teff/logg of the bad stars are measured: "
                     f"{100*(s_clam[bad,0] < MISSING).mean():.0f}% / "
                     f"{100*(s_clam[bad,2] < MISSING).mean():.0f}%")
    print("\n".join(lines))

    # --- 2. pairs ---------------------------------------------------------
    table = []
    for cat in range(n_cat):
        if cat == clam_cat:
            continue
        xj, mj, sj, found = lookup(cat)
        cov = coverage(xs[c == cat])
        table.append(dict(cat=cat, n=int(found.sum()), cov=cov, x=xj, m=mj, s=sj))
    table.sort(key=lambda r: -r["n"])

    hdr = (f"{'paired survey':26s} {'n_CLAM':>7s} {'logg':>9s} {'[alpha/Fe]':>11s} "
           f"{'r raw':>6s} {'r map':>6s} {'med d raw':>10s} {'med d map':>10s} "
           f"{'scat raw':>9s} {'scat map':>9s}")
    lines = ["", "2. CLAM stars shared with each survey ([Fe/H], raw vs harmonized)",
             hdr, "-" * len(hdr)]
    for row in table:
        xj, mj = row["x"], row["m"]
        ok_f = np.isfinite(xj[:, FEH]) & np.isfinite(x_clam[:, FEH])
        r_raw = np.corrcoef(xj[ok_f, FEH], x_clam[ok_f, FEH])[0, 1] if ok_f.sum() > 10 else np.nan
        ok_m = np.isfinite(mj[:, FEH]) & np.isfinite(m_clam[:, FEH])
        r_map = np.corrcoef(mj[ok_m, FEH], m_clam[ok_m, FEH])[0, 1] if ok_m.sum() > 10 else np.nan
        d_raw = np.median((x_clam[ok_f, FEH] - xj[ok_f, FEH]))
        d_map = np.median((m_clam[ok_m, FEH] - mj[ok_m, FEH]))
        s_raw = 1.4826 * np.median(np.abs(x_clam[ok_f, FEH] - xj[ok_f, FEH]
                                          - d_raw)) if ok_f.sum() > 10 else np.nan
        s_map = 1.4826 * np.median(np.abs(m_clam[ok_m, FEH] - mj[ok_m, FEH]
                                          - d_map)) if ok_m.sum() > 10 else np.nan
        flag = []
        if row["cov"]["alpha"] == "none":
            flag.append("(no alpha)")
        elif row["cov"]["alpha"].startswith("some"):
            flag.append(f"({row['cov']['alpha']} alpha)")
        if row["cov"]["logg"] == "none":
            flag.append("(no logg)")
        name = labels[row["cat"]] + (" " + " ".join(flag) if flag else "")
        lines.append(f"{name:26s} {row['n']:7d} {row['cov']['logg']:>9s} "
                     f"{row['cov']['alpha']:>11s} {r_raw:6.2f} {r_map:6.2f} "
                     f"{d_raw:+10.2f} {d_map:+10.2f} {s_raw:9.3f} {s_map:9.3f}")
    print("\n".join(lines))
    (out_dir / "pairs_summary.txt").write_text("\n".join(lines) + "\n")

    # --- 3. figures -------------------------------------------------------
    n_fig = 0
    for row in table:
        if row["n"] < args.min_pairs:
            continue
        xj, mj, sj = row["x"], row["m"], row["s"]
        name = labels[row["cat"]]
        fig, axes = plt.subplots(2, 2, figsize=(9.5, 9.5), dpi=110)
        for r, (j, pname) in enumerate(((FEH, "[Fe/H]"), (ALPHA, "[alpha/Fe]"))):
            both = (sj[:, j] < MISSING) & (s_clam[:, j] < MISSING) \
                & np.isfinite(xj[:, j]) & np.isfinite(x_clam[:, j])
            raw = (xj[both, j], x_clam[both, j])
            mapped_ok = (sj[:, j] < MISSING) & (s_clam[:, j] < MISSING) \
                & np.isfinite(mj[:, j]) & np.isfinite(m_clam[:, j])
            mapped = (mj[mapped_ok, j], m_clam[mapped_ok, j])
            # one set of limits per row, from both columns
            vals = np.concatenate([raw[0], raw[1], mapped[0], mapped[1]]) \
                if len(raw[0]) or len(mapped[0]) else np.array([0.0, 1.0])
            lo, hi = np.percentile(vals, [0.3, 99.7])
            pad = 0.05 * max(hi - lo, 0.2)
            lim = (lo - pad, hi + pad)
            for col, (xs_, ys_, title) in enumerate(
                    ((*raw, "raw"), (*mapped, "harmonized"))):
                ax = axes[r, col]
                if len(xs_) == 0:
                    ax.text(0.5, 0.5, f"no pair measures {pname}", ha="center",
                            va="center", transform=ax.transAxes, fontsize=9)
                    ax.set_xlim(lim); ax.set_ylim(lim)
                    continue
                sel = bad[both if col == 0 else mapped_ok]
                ax.scatter(xs_[~sel], ys_[~sel], s=3, c="0.6", alpha=0.35,
                           edgecolors="none", rasterized=True)
                ax.scatter(xs_[sel], ys_[sel], s=4, c="crimson", alpha=0.6,
                           edgecolors="none", rasterized=True,
                           label="bad stars" if col == 0 and r == 0 else None)
                ax.plot(lim, lim, color="k", lw=0.8, ls="--")
                ax.set_xlim(lim); ax.set_ylim(lim)
                ax.set_aspect("equal", adjustable="box")
                ax.grid(alpha=0.2)
                ax.text(0.03, 0.97, scatter_text(xs_, ys_), transform=ax.transAxes,
                        va="top", fontsize=7.5,
                        bbox=dict(fc="white", ec="0.7", alpha=0.85, lw=0.5))
                ax.set_title(f"{title}: {pname}  (n={len(xs_)})", fontsize=9)
                ax.set_xlabel(f"{name} {pname}", fontsize=8)
                ax.set_ylabel(f"SDSS-V (CLAM) {pname}", fontsize=8)
                ax.tick_params(labelsize=7)
        axes[0, 0].legend(loc="lower right", fontsize=7, markerscale=2)
        in_pair = np.isfinite(xj[:, FEH]) & np.isfinite(x_clam[:, FEH])
        fig.suptitle(
            f"{name}: {int(row['n'])} CLAM stars in common, "
            f"{int((bad & in_pair).sum())} of them bad", fontsize=11)
        fig.tight_layout()
        out = out_dir / f"pair_{registry.keys()[row['cat']]}.png"
        fig.savefig(out)
        plt.close(fig)
        n_fig += 1
    log.info("wrote %d pair figures to %s", n_fig, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
