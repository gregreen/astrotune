#!/usr/bin/env python
"""Look at the stars one catalog's map sends to the bottom-left of [Fe/H]-[a/Fe].

    python scripts/clam_low_alpha_selection.py            # data/sdssv_clam_subset.h5

Reads the subset file written by ``scripts/clam_subset.py`` (which carries ``x``,
``f_c(x)`` and the inferred ``y`` for every observation of every star the focal
catalog observes) and

1. selects the stars whose focal-catalog ``f(x)`` lies to the bottom-left of the
   line through ``(0, -0.1)`` and ``(-1.0, 0)``, i.e. ``a < -0.1 - 0.1 [Fe/H]``;
2. for those stars, plots the focal catalog's ``f(x) - y`` and ``f(x) - x`` in the
   ``([Fe/H], [a/Fe])`` plane;
3. does the same for the observations of those stars from *all the other*
   catalogs.

``f(x) - y`` is the residual the data term penalises: a healthy fit scatters
around the origin with the size of the catalog's own uncertainty. ``f(x) - x``
is how far the map moved the catalog's own numbers; a map that is *moving* a
population somewhere is visible there even when the residuals are small.

Output: ``diagnostic_plots/<catalog>_selection/`` (the two figures) and the same
numbers on stdout.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

#: The line's two end points in ([Fe/H], [a/Fe])
DEFAULT_LINE = ((0.0, -0.1), (-1.0, 0.0))


def _halves(fig, title, rows, xlabel, ylabel, out):
    """A 2-column grid: the differences on the left, ``f(x) - x`` on the right.

    ``rows`` is a sequence of ``(left, right)`` pairs, each of them a ``(dx, dy)``
    tuple; one row per transformation of the residual (raw, then divided by the
    catalog's own uncertainty). Every row has its own symmetric limits, shared by
    its two panels.
    """
    axes = np.atleast_2d(fig.subplots(len(rows), 2))
    for r, row in enumerate(rows):
        span = 0.1
        for dx, dy in row:
            span = max(span, np.percentile(np.abs(dx), 99), np.percentile(np.abs(dy), 99))
        span *= 1.15
        for c, ((dx, dy), label) in enumerate(zip(row, ("$f(x) - y$", "$f(x) - x$"))):
            ax = axes[r, c]
            ax.axhline(0.0, color="k", lw=0.6, alpha=0.5)
            ax.axvline(0.0, color="k", lw=0.6, alpha=0.5)
            ax.scatter(dx, dy, s=1, alpha=0.06, rasterized=True, c="C0")
            ax.plot([np.median(dx)], [np.median(dy)], marker="+", color="C3", ms=14, mew=2)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_xlim(-span, span)
            ax.set_ylim(-span, span)
            ax.set_aspect("equal")
            ax.grid(alpha=0.2)
            ax.set_title(label, fontsize=9)
            ax.text(
                0.03, 0.97,
                f"n = {len(dx):,}\nmedian ({np.median(dx):+.3f}, {np.median(dy):+.3f})\n"
                f"rms ({dx.std():.3f}, {dy.std():.3f})",
                transform=ax.transAxes, va="top", fontsize=8,
                bbox=dict(fc="white", ec="0.7", alpha=0.9),
            )
    fig.suptitle(title)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  wrote {out}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--file", type=Path, default=Path("data/sdssv_clam_subset.h5"))
    parser.add_argument("--catalog", default=None,
                        help="focal catalog (default: attrs['catalog'] of the file)")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="default: diagnostic_plots/<catalog>_selection")
    parser.add_argument("--min-rows", type=int, default=100,
                        help="skip the ensemble panel of catalogs with fewer rows")
    args = parser.parse_args(argv)

    with h5py.File(args.file, "r") as f:
        keys = str(f.attrs["catalog_keys"]).split(",")
        catalog = args.catalog or str(f.attrs["catalog"])
        c = np.asarray(f["obs/catalog"][:])
        star = np.asarray(f["obs/star"][:])
        x = np.asarray(f["obs/x"][:], dtype="f8")
        fx = np.asarray(f["obs/fx"][:], dtype="f8")
        sigma = np.asarray(f["obs/sigma_x"][:], dtype="f8")
        y = np.asarray(f["stars/y"][:], dtype="f8")
        step = int(f.attrs["checkpoint_step"])
    if catalog not in keys:
        print(f"{catalog} is not one of {keys}", file=sys.stderr)
        return 1
    k = keys.index(catalog)
    out_dir = args.out_dir or Path("diagnostic_plots") / f"{catalog}_selection"

    (feh0, a0), (feh1, a1) = DEFAULT_LINE
    slope = (a1 - a0) / (feh1 - feh0)
    print(f"{catalog}: {np.count_nonzero(c == k):,} rows, map from step {step}")
    print(f"selection: [a/Fe] < {a0:+.2f} {slope:+.2f} * [Fe/H]")

    # --- 1. the stars whose focal f(x) is below the line -----------------------
    mine = c == k
    n_stars = len(np.unique(star))
    sum_feh = np.zeros(n_stars)
    sum_a = np.zeros(n_stars)
    counts = np.zeros(n_stars, dtype=int)
    np.add.at(sum_feh, star[mine], fx[mine, 1])
    np.add.at(sum_a, star[mine], fx[mine, 3])
    np.add.at(counts, star[mine], 1)
    has = counts > 0
    mean_feh = np.where(has, sum_feh / np.maximum(counts, 1), np.nan)
    mean_a = np.where(has, sum_a / np.maximum(counts, 1), np.nan)
    selected = has & (mean_a < a0 + slope * (mean_feh - feh0))
    print(f"{int(has.sum()):,} stars have a {catalog} observation "
          f"({counts[has].mean():.2f} rows each); {int(selected.sum()):,} are below the line "
          f"({100 * selected.mean():.1f}% of the others)")
    print(f"  their mean f(x): [Fe/H] median {np.nanmedian(mean_feh[selected]):+.2f}, "
          f"[a/Fe] median {np.nanmedian(mean_a[selected]):+.2f}")

    sel_rows = selected[star]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    def two_rows(mask, sigma_rows):
        """(f - y, f - x) and the same divided by the row's own uncertainty.

        The division is in *x*-space: ``sigma_x`` is the uncertainty of the
        catalog's own value, which is what the data term charges ``f(x) - y``
        with, and it is the only one available per observation. Rows whose
        [alpha/Fe] is a placeholder (``sigma_x > 1e6``) cannot be standardised
        and are dropped from that row.
        """
        dy = (fx[mask, 1] - y[star[mask], 1], fx[mask, 3] - y[star[mask], 3])
        dx = (fx[mask, 1] - x[mask, 1], fx[mask, 3] - x[mask, 3])
        s = sigma_rows[mask]
        ok = (s[:, 1] < 1e6) & (s[:, 3] < 1e6) & (s[:, 1] > 0) & (s[:, 3] > 0)
        if not np.all(ok):
            print(f"  ({int((~ok).sum()):,} rows without both uncertainties are "
                  f"left out of the standardised row)")
        zy = (dy[0][ok] / s[ok, 1], dy[1][ok] / s[ok, 3])
        zx = (dx[0][ok] / s[ok, 1], dx[1][ok] / s[ok, 3])
        return [(dy, dx), (zy, zx)]

    # --- 2. the focal catalog ---------------------------------------------------
    use = mine & sel_rows
    rows = two_rows(use, sigma)
    fig = plt.figure(figsize=(9.5, 8.6), layout="constrained")
    _halves(
        fig,
        f"{catalog}: {int(selected.sum()):,} stars whose $f(x)$ is below the line\n"
        "(bottom row: divided by the catalog's own uncertainty)",
        rows,
        r"$\Delta$[Fe/H]", r"$\Delta$[$\alpha$/Fe]",
        out_dir / "selection_fx_minus_y_and_x.png",
    )
    # --- 3. every other catalog --------------------------------------------------
    other = ~mine & sel_rows
    rows = two_rows(other, sigma)
    fig = plt.figure(figsize=(9.5, 8.6), layout="constrained")
    _halves(
        fig,
        f"all other catalogs, same {int(selected.sum()):,} stars\n"
        "(bottom row: divided by each row's own uncertainty)",
        rows,
        r"$\Delta$[Fe/H]", r"$\Delta$[$\alpha$/Fe]",
        out_dir / "selection_others_fx_minus_y_and_x.png",
    )

    # --- the same numbers, catalog by catalog -----------------------------------
    print(f"\n{'catalog':22s} {'rows':>7s} {'med f-y [Fe/H]':>15s} {'[a/Fe]':>8s} "
          f"{'rms f-y [Fe/H]':>15s} {'[a/Fe]':>8s} {'med f-x [a/Fe]':>15s}")
    for ci, key in enumerate(keys):
        m = (c == ci) & sel_rows
        if m.sum() < args.min_rows:
            continue
        d_fy = np.c_[fx[m, 1] - y[star[m], 1], fx[m, 3] - y[star[m], 3]]
        d_fx = np.c_[fx[m, 1] - x[m, 1], fx[m, 3] - x[m, 3]]
        print(f"{key:22s} {m.sum():7d} {np.median(d_fy[:, 0]):+15.3f} {np.median(d_fy[:, 1]):+8.3f} "
              f"{d_fy[:, 0].std():15.3f} {d_fy[:, 1].std():8.3f} "
              f"{np.median(d_fx[:, 1]):+15.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
