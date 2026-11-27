#!/usr/bin/env python
"""The off-centre mode of the CLAM residual cloud: what are those stars?

    python scripts/clam_offmode.py

``scripts/clam_low_alpha_selection.py`` shows that CLAM's ``f(x) - y`` in the
``([Fe/H], [a/Fe])`` plane has two modes, one on the origin and one near
``(-0.3, -0.2)``. This splits the CLAM rows between the two modes (two-means,
started from those two centres) and reports

(a) the distribution of the [Fe/H] and [a/Fe] *uncertainties* of each mode, as
    the loss sees them (``sigma_x``, i.e. scaled and floored) and as the catalog
    quoted them (undoing ``UNC_SCALES`` and ``ERROR_FLOORS``);
(b) how many of the stars in each mode have more than one CLAM measurement --
    both in the prepared table (rows per star) and in the raw catalog
    (``n_boss_visits``, ``n_good_visits``), which is the number of spectra CLAM
    fitted before any cut.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

CENTRES = np.array([[0.0, 0.0], [-0.3, -0.2]])
PARAM = {1: "[Fe/H]", 3: "[a/Fe]"}


def two_means(d, centres, n_iter=20):
    """Assign to the nearest centre and move the centres; fixed initialisation."""
    centres = centres.copy()
    for _ in range(n_iter):
        dist = ((d[:, None, :] - centres[None, :, :]) ** 2).sum(-1)
        label = np.argmin(dist, axis=1)
        new = np.array([d[label == k].mean(0) if np.any(label == k) else centres[k]
                        for k in range(len(centres))])
        if np.allclose(new, centres):
            break
        centres = new
    return label, centres


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--file", type=Path, default=Path("data/sdssv_clam_subset.h5"))
    parser.add_argument("--raw", type=Path,
                        default=Path("data/raw/sdssv_clam/boss_clam_lite-1.0.0.h5"))
    parser.add_argument("--out-dir", type=Path,
                        default=Path("diagnostic_plots/sdssv_clam_selection"))
    args = parser.parse_args(argv)

    with h5py.File(args.file, "r") as f:
        keys = str(f.attrs["catalog_keys"]).split(",")
        k = keys.index(str(f.attrs["catalog"]))
        c = np.asarray(f["obs/catalog"][:])
        star = np.asarray(f["obs/star"][:])
        ids = np.asarray(f["obs/id"][:])
        x = np.asarray(f["obs/x"][:], dtype="f8")
        fx = np.asarray(f["obs/fx"][:], dtype="f8")
        sigma = np.asarray(f["obs/sigma_x"][:], dtype="f8")
        y = np.asarray(f["stars/y"][:], dtype="f8")

    mine = c == k
    resid = np.c_[fx[mine, 1] - y[star[mine], 1], fx[mine, 3] - y[star[mine], 3]]
    label, centres = two_means(resid, CENTRES)
    order = np.argsort(-np.linalg.norm(centres, axis=1))  # off-centre mode first
    print(f"{int(mine.sum()):,} CLAM rows in {args.file}")
    for i in order:
        print(f"  mode {i}: centre ({centres[i, 0]:+.3f}, {centres[i, 1]:+.3f}), "
              f"{int((label == i).sum()):,} rows ({100 * np.mean(label == i):.1f}%)")
    off = order[0]
    is_off = label == off

    # --- uncertainties -----------------------------------------------------------
    from astrotune.catalogs import sdssv_clam as clam

    scales = np.asarray(clam.UNC_SCALES, dtype="f8")
    floors = np.asarray(clam.ERROR_FLOORS, dtype="f8")
    sig_mine = sigma[mine]
    quoted = np.sqrt(np.clip(sig_mine ** 2 - floors ** 2, 0, None)) / scales

    print("\nuncertainties, off-centre mode vs the rest (median [p16, p84]):")
    for j in (1, 3):
        for name, values in (("sigma_x (loss)", sig_mine[:, j]), ("quoted", quoted[:, j])):
            a, b = values[is_off], values[~is_off]
            print(f"  {name:15s} {PARAM[j]:7s} off {np.median(a):.4f} "
                  f"[{np.percentile(a, 16):.4f}, {np.percentile(a, 84):.4f}]   "
                  f"on {np.median(b):.4f} [{np.percentile(b, 16):.4f}, {np.percentile(b, 84):.4f}]")
    # Where the two modes sit in the catalog's own coordinates: the uncertainties
    # came out identical, so the difference has to be elsewhere.
    x_mine = x[mine]
    y_mine = y[star[mine]]
    print("\nthe catalog's own values (median), off-centre vs the rest:")
    names = ("Teff (K)", "[Fe/H]", "logg", "[a/Fe]")
    scale = np.array([1.0, 1.0, 1.0, 1.0])
    for j, name in enumerate(names):
        a, b = x_mine[is_off, j], x_mine[~is_off, j]
        if j == 0:
            a, b = 5000 * np.exp(a / 5), 5000 * np.exp(b / 5)
        print(f"  x  {name:9s} off {np.median(a):9.3f}   on {np.median(b):9.3f}")
    for j, name in enumerate(names):
        a, b = y_mine[is_off, j], y_mine[~is_off, j]
        if j == 0:
            a, b = 5000 * np.exp(a / 5), 5000 * np.exp(b / 5)
        print(f"  y  {name:9s} off {np.median(a):9.3f}   on {np.median(b):9.3f}")
    lo, hi = clam.ALPHA_RANGE
    for name, m in (("off-centre", is_off), ("on origin", ~is_off)):
        edge = np.mean((x_mine[m, 3] < lo + 0.02) | (x_mine[m, 3] > hi - 0.02)) * 100
        print(f"  {name:11s}: {edge:.2f}% of its rows have x[a/Fe] within 0.02 of the cut "
              f"({lo:+.2f}, {hi:+.2f})")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), layout="constrained")
    for row, (title, values) in enumerate(
        (("as the loss sees it ($\\sigma_x$)", sig_mine), ("as the catalog quoted it", quoted))
    ):
        for col, j in enumerate((1, 3)):
            ax = axes[row, col]
            lo = np.percentile(values[:, j], 0.5)
            hi = np.percentile(values[:, j], 99.5)
            bins = np.linspace(lo, hi, 120)
            ax.hist(values[is_off, j], bins=bins, histtype="step", density=True,
                    label=f"off-centre ({is_off.sum():,})", color="C1")
            ax.hist(values[~is_off, j], bins=bins, histtype="step", density=True,
                    label=f"on origin ({(~is_off).sum():,})", color="C0")
            ax.set_xlabel(f"$\\sigma$({PARAM[j]})")
            ax.set_ylabel("probability density")
            ax.set_title(f"{PARAM[j]} — {title}", fontsize=9)
            ax.legend(fontsize=7)
            ax.grid(alpha=0.2)
            if row == 1:
                ax.set_xscale("log")
    fig.suptitle("CLAM: uncertainties of the two residual modes")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "offmode_uncertainties.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nwrote {out}")

    # --- where the two modes are, before and after the map -------------------------
    # Columns 1 and 3 of x are [Fe/H] and [alpha/Fe] (0 is Teff, 2 is log g): the
    # whole 4-vector must not go into a 2-D scatter, which plots Teff against
    # [Fe/H] and looks nothing like an abundance plane.
    from astrotune.prep import plots

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.0), layout="constrained")
    # Each row is one plane, drawn three times -- x, f(x), and the star's inferred y --
    # with one set of limits per row; the Kiel row is in the transformed temperature
    # that ``plots`` uses for every other Kiel diagram in the project.
    y_mine = y[star[mine]]  # one row per CLAM observation, so y[star] for the same star
    rows = (
        ("abundance",
         [np.c_[x_mine[:, 1], x_mine[:, 3]],
          np.c_[fx[mine, 1], fx[mine, 3]],
          np.c_[y_mine[:, 1], y_mine[:, 3]]],
         "[Fe/H]", r"[$\alpha$/Fe]"),
        ("kiel",
         [np.c_[x_mine[:, 0], x_mine[:, 2]],
          np.c_[fx[mine, 0], fx[mine, 2]],
          np.c_[y_mine[:, 0], y_mine[:, 2]]],
         None, None),
    )
    titles = ("the catalog's own values, $x$", "where the map sends them, $f(x)$",
              "the star's inferred $y$")
    for row, (kind, planes, xlabel, ylabel) in enumerate(rows):
        both = np.vstack(planes)
        xlim = np.percentile(both[:, 0], [0.05, 99.95])
        ylim = np.percentile(both[:, 1], [0.05, 99.95])
        xlim = xlim + np.array([-1, 1]) * 0.05 * np.ptp(xlim)
        ylim = ylim + np.array([-1, 1]) * 0.05 * np.ptp(ylim)
        for col, (values, title) in enumerate(zip(planes, titles)):
            ax = axes[row, col]
            ax.scatter(values[~is_off, 0], values[~is_off, 1], s=0.6, alpha=0.03,
                       c="C0", rasterized=True,
                       label=f"mode on the origin ({(~is_off).sum():,})")
            ax.scatter(values[is_off, 0], values[is_off, 1], s=0.6, alpha=0.06,
                       c="C1", rasterized=True,
                       label=f"off-centre mode ({is_off.sum():,})")
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            if kind == "kiel":
                plots._style_kiel_axis(ax)  # inverts both axes, Teff ticks in kK
            else:
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                ax.grid(alpha=0.2)
            ax.set_title(title, fontsize=9)
            if col == 0:
                ax.legend(fontsize=7, markerscale=12, loc="upper left")
    fig.suptitle("CLAM: the two residual modes in $x$, $f(x)$ and $y$")
    out = args.out_dir / "offmode_x_and_fx.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")

    # --- repeat measurements -------------------------------------------------------
    star_ids = np.asarray(f["stars/id"][:]) if False else None  # noqa: F841 (kept simple)
    off_rows = np.flatnonzero(mine)[is_off]
    on_rows = np.flatnonzero(mine)[~is_off]
    off_stars = np.unique(star[off_rows])
    on_stars = np.unique(star[on_rows])

    n_rows_per_star = np.bincount(star[mine], minlength=len(y))
    print("\nprepared-table CLAM rows per star (a '2nd measurement' in the table):")
    for name, ss in (("off-centre", off_stars), ("on origin", on_stars)):
        n = n_rows_per_star[ss]
        print(f"  {name:11s} {len(ss):7,} stars; rows/star: median {np.median(n):.0f}, "
              f"max {n.max()}, fraction with >=2 rows: {np.mean(n >= 2) * 100:.3f}%")

    # --- the raw catalog: how many spectra did CLAM fit for these stars? -------------
    if args.raw.exists():
        with h5py.File(args.raw, "r") as f:
            raw_ids = np.asarray(f["meta/gaia_dr3_source_id"][:], dtype="i8")
            visits = np.asarray(f["meta/n_boss_visits"][:], dtype="f8")
            good = np.asarray(f["meta/n_good_visits"][:], dtype="f8")
        keep = raw_ids > 0
        raw_ids, visits, good = raw_ids[keep], visits[keep], good[keep]
        # one row per star in the raw catalog is enough for n_boss_visits
        uniq, first = np.unique(raw_ids, return_index=True)
        v_per_star = visits[first]
        g_per_star = good[first]
        n_raw = np.bincount(np.searchsorted(uniq, raw_ids), minlength=len(uniq))

        mine_ids = ids[mine]
        print(f"\nraw catalog: {len(uniq):,} stars, {len(raw_ids):,} spectra "
              f"({np.median(v_per_star):.0f} BOSS visits per star, median)")
        print(f"{'mode':11s} {'stars':>7s} {'>=2 spectra':>12s} {'>=2 visits':>11s} "
              f"{'>=2 good':>9s} {'median visits':>14s}")
        for name, ss in (("off-centre", off_stars), ("on origin", on_stars)):
            # the star's Gaia id: take any CLAM row of the star
            sel = np.flatnonzero(mine)
            star_of_row = star[sel]
            first_row = np.unique(star_of_row, return_index=True)[1]
            id_of_star = np.full(len(y), 0, dtype="i8")
            id_of_star[star_of_row[first_row]] = ids[sel[first_row]]
            sid = id_of_star[ss]
            pos = np.clip(np.searchsorted(uniq, sid), 0, len(uniq) - 1)
            hit = uniq[pos] == sid
            print(f"  {name:9s} {len(ss):7,} {np.mean(n_raw[pos[hit]] >= 2) * 100:11.2f}% "
                  f"{np.mean(v_per_star[pos[hit]] >= 2) * 100:10.2f}% "
                  f"{np.mean(g_per_star[pos[hit]] >= 2) * 100:8.2f}% "
                  f"{np.median(v_per_star[pos[hit]]):14.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
