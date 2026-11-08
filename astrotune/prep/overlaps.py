"""How many stars does a catalog share with the others, and at what temperature?

For a prepared catalog (typically a candidate that is not yet in the model),
count the stars it has in common with each registered catalog, matching on the
Gaia DR3 source id, and split the shared stars by the candidate's own Teff.

    python -m astrotune.prep.overlaps desi_nagaraj2026

writes ``diagnostic_plots/<key>/overlaps.png`` and ``overlaps.json``.

Counts reported per partner catalog:

``n_stars``
    unique Gaia ids present in both catalogs;
``n_pairs``
    rows of the many-to-many join on ``id``, i.e. what training would see: a
    star with 3 rows in one catalog and 2 in the other contributes 6 pairs
    (duplicate ids are kept, see ``porting_project.md`` §5.4);
``n_stars_alpha``
    shared stars for which **both** catalogs measure [alpha/Fe]. These are the
    stars that constrain [alpha/Fe]; a catalog without [alpha/Fe], or a star
    whose [alpha/Fe] is marked missing (``sigma_x >= 1e6``), does not count.

The partners are the registered catalogs and the other prepared candidates.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from .. import config, registry
from .io import load_catalog
from .transforms import TEFF_REF_K, TEFF_SCALE

log = logging.getLogger(__name__)

#: Teff bin edges (K): fine at the cool end, where the M dwarfs are, and
#: covering every catalog (stars outside the bins are not in the heatmaps)
TEFF_EDGES = (
    0.0, 3500.0, 3750.0, 4000.0, 4300.0, 4600.0, 5000.0, 5500.0, 6000.0, 6500.0, 8000.0, 1e6,
)
#: Stars below this Teff (K) count as "cool"
COOL_TEFF = 4600.0
#: A statistical uncertainty at or above this marks a parameter that is not measured
MISSING_SIGMA = 1e6
ALPHA_INDEX = 3


def x_to_teff(x0) -> np.ndarray:
    """Invert the Teff transformation ``x0 = 5 ln(Teff / 5000 K)``."""
    return TEFF_REF_K * np.exp(np.asarray(x0, dtype="f8") / TEFF_SCALE)


def _unique_ids(table):
    """Unique ids of a prepared table, with per-id row counts and [alpha/Fe] status.

    Returns ``(ids, first_row, n_rows, alpha)``; ``alpha[i]`` is True if any row
    of star ``i`` has [alpha/Fe] measured.
    """
    ids = np.asarray(table["id"])
    u_ids, first, inverse, counts = np.unique(
        ids, return_index=True, return_inverse=True, return_counts=True
    )
    sigma = np.asarray(table["sigma_x"])
    if sigma.shape[1] > ALPHA_INDEX:
        row_alpha = sigma[:, ALPHA_INDEX] < MISSING_SIGMA
        alpha = np.bincount(inverse, weights=row_alpha.astype("f8"), minlength=len(u_ids)) > 0
    else:
        alpha = np.zeros(len(u_ids), dtype=bool)
    return u_ids, first, counts, alpha


def default_partners(key: str) -> list:
    """Registered catalogs and other candidates that have been prepared."""
    keys = list(registry.keys()) + list(registry.candidate_keys())
    return [k for k in keys if k != key and config.catalog_path(k).exists()]


def overlap_counts(
    key: str,
    partner_keys: Optional[Sequence[str]] = None,
    teff_edges: Sequence[float] = TEFF_EDGES,
) -> dict:
    """Overlap of catalog ``key`` with each partner catalog.

    Returns a dict with ``teff_edges``, ``n_catalog`` (rows of ``key``) and a
    list ``partners`` of dicts (``key``, ``label``, ``candidate``, ``n_rows``,
    ``n_stars``, ``n_pairs``, ``n_stars_by_teff``, ``n_stars_cool`` and, for the
    stars with [alpha/Fe] in both catalogs, ``n_stars_alpha``,
    ``n_stars_alpha_by_teff``, ``n_stars_alpha_cool``).
    """
    edges = np.asarray(teff_edges, dtype="f8")
    cand = load_catalog(key)
    teff = x_to_teff(np.asarray(cand["x"])[:, 0])
    u_ids, first, c_counts, c_alpha = _unique_ids(cand)
    u_teff = teff[first]

    partners = []
    for pkey in partner_keys if partner_keys is not None else default_partners(key):
        if pkey == key or not config.catalog_path(pkey).exists():
            continue
        other = load_catalog(pkey)
        o_ids, _, o_counts, o_alpha = _unique_ids(other)
        shared, i_c, i_o = np.intersect1d(u_ids, o_ids, return_indices=True)
        both_alpha = c_alpha[i_c] & o_alpha[i_o]
        hist, _ = np.histogram(u_teff[i_c], bins=edges)
        hist_alpha, _ = np.histogram(u_teff[i_c][both_alpha], bins=edges)
        partners.append(
            dict(
                key=pkey,
                label=registry.get_spec(pkey).label,
                candidate=pkey in registry.candidate_keys(),
                n_rows=int(len(other)),
                n_stars=int(len(shared)),
                n_pairs=int(np.sum(c_counts[i_c].astype("i8") * o_counts[i_o])),
                n_stars_by_teff=hist.tolist(),
                n_stars_cool=int(np.sum(u_teff[i_c] < COOL_TEFF)),
                n_stars_alpha=int(np.count_nonzero(both_alpha)),
                n_stars_alpha_by_teff=hist_alpha.tolist(),
                n_stars_alpha_cool=int(np.sum(both_alpha & (u_teff[i_c] < COOL_TEFF))),
            )
        )
    return dict(
        key=key,
        n_catalog=int(len(cand)),
        n_unique_ids=int(len(u_ids)),
        teff_edges=edges.tolist(),
        teff_range=[float(teff.min()), float(teff.max())],
        partners=partners,
    )


def _heatmap(ax, counts, edges, y, n_cool_bins, title):
    from matplotlib.colors import LogNorm

    show = np.where(counts > 0, counts, np.nan)
    vmax = max(np.nanmax(show) if np.any(counts > 0) else 10, 10)
    im = ax.imshow(show, aspect="auto", norm=LogNorm(vmin=1, vmax=vmax), cmap="viridis")
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            c = int(counts[i, j])
            ax.text(j, i, f"{c:,}" if c else "0", ha="center", va="center", fontsize=6.5,
                    color="w" if 0 < c < 0.3 * vmax else "k")
    names = [f"{lo:.0f}-{hi:.0f}" for lo, hi in zip(edges[:-1], edges[1:])]
    names[0], names[-1] = f"<{edges[1]:.0f}", f">{edges[-2]:.0f}"
    ax.set_xticks(np.arange(len(edges) - 1), names, rotation=35, fontsize=7)
    ax.set_yticks(y, [])
    ax.axvline(n_cool_bins - 0.5, c="r", lw=1.5)
    ax.set_xlabel("Teff of the shared stars (K); cool to the left of the red line", fontsize=8)
    ax.set_title(title, fontsize=9)
    return im


def plot_overlaps(result: dict, outdir: Path, name: str = "overlaps") -> Path:
    """Shared stars per catalog (bars), and catalog x Teff-bin heatmaps.

    Left: stars in common (blue), of which cool (red), and of which have
    [alpha/Fe] measured in both catalogs (black ticks). Middle: all shared stars
    by Teff. Right: only those with [alpha/Fe] in both catalogs.
    """
    if not os.environ.get("DISPLAY"):
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from . import plots

    partners = sorted(result["partners"], key=lambda p: p["n_stars"], reverse=True)
    labels = [p["label"] + (" [candidate]" if p.get("candidate") else "") for p in partners]
    edges = np.asarray(result["teff_edges"])
    n_cool_bins = int(np.sum(edges[1:] <= COOL_TEFF))
    counts = np.array([p["n_stars_by_teff"] for p in partners], dtype=float)
    counts_alpha = np.array([p["n_stars_alpha_by_teff"] for p in partners], dtype=float)

    fig, (ax0, ax1, ax2) = plt.subplots(
        1, 3, figsize=(17, 0.4 * len(partners) + 2.6),
        gridspec_kw=dict(width_ratios=[1, 1.2, 1.2]), layout="constrained",
    )
    y = np.arange(len(partners))
    n_max = max(max(p["n_stars"] for p in partners), 1)
    ax0.barh(y, [p["n_stars"] for p in partners], color="C0", label="all Teff")
    ax0.barh(y, [p["n_stars_cool"] for p in partners], color="C3", label=f"Teff < {COOL_TEFF:.0f} K")
    ax0.plot([max(p["n_stars_alpha"], 0.8) for p in partners], y, "|", c="k", ms=10, mew=2,
             label="[alpha/Fe] in both")
    ax0.set_xscale("log")
    ax0.set_yticks(y, labels, fontsize=8)
    ax0.invert_yaxis()
    for yi, p in zip(y, partners):
        ax0.text(max(p["n_stars"], 1) * 1.15, yi,
                 f"{p['n_stars']:,}  ({p['n_stars_alpha']:,} with [a/Fe])", va="center", fontsize=6.5)
    ax0.set_xlim(0.8, n_max * 60)
    ax0.set_xlabel("stars in common (unique Gaia ids)")
    ax0.legend(fontsize=7, loc="lower right")

    im = _heatmap(ax1, counts, edges, y, n_cool_bins, "all shared stars")
    _heatmap(ax2, counts_alpha, edges, y, n_cool_bins, "shared stars with [alpha/Fe] in both catalogs")
    fig.colorbar(im, ax=ax2, label="stars", shrink=0.6)
    fig.suptitle(
        f"{registry.get_spec(result['key']).label}: {result['n_catalog']:,} stars, "
        f"Teff {result['teff_range'][0]:.0f}-{result['teff_range'][1]:.0f} K",
        fontsize=10,
    )
    plots.setup_style()
    return plots.save(fig, outdir, name)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="astrotune-overlaps",
        description="Count the stars a prepared catalog shares with the registered catalogs.",
    )
    parser.add_argument("key", help="catalog key (a registered catalog or a candidate)")
    parser.add_argument("--partners", nargs="*", default=None, help="catalog keys (default: all)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    result = overlap_counts(args.key, args.partners)
    outdir = config.diagnostic_dir(args.key)
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "overlaps.json", "w") as f:
        json.dump(result, f, indent=1)
    path = plot_overlaps(result, outdir)

    cool = f"Teff<{COOL_TEFF:.0f}K"
    print(f"{'catalog':28s} {'rows':>9s} {'stars':>8s} {'pairs':>9s} {cool:>9s} | {'both a/Fe':>9s} {cool:>9s}")
    for p in sorted(result["partners"], key=lambda p: -p["n_stars"]):
        label = p["label"] + (" *" if p.get("candidate") else "")
        print(f"{label:28s} {p['n_rows']:9,d} {p['n_stars']:8,d} {p['n_pairs']:9,d} {p['n_stars_cool']:9,d} | "
              f"{p['n_stars_alpha']:9,d} {p['n_stars_alpha_cool']:9,d}")
    print("(* = candidate catalog, not in the model)")
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
