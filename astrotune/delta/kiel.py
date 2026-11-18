"""Where the learned correction sits in the Kiel diagram, per catalog.

One figure per catalog, a 4x4 grid: a column per parameter (Teff, [Fe/H], logg,
[alpha/Fe]), a row per [Fe/H] bin -- metal-rich at the top and metal-poor at the
bottom, so [Fe/H] descends down the figure like the luminosity does in a Kiel
diagram. Inside each panel a Kiel projection (5 ln Teff/5000 K against logg) is
coloured by the model's correction, twice over:

``<key>.png``
    ``ln(sigma_out / sigma_in)`` = ``0.5 ln(scale + add/C)``, the dimensionless
    inflation of the quoted error: 0 is "left alone" (grey), negative is "shrunk"
    (red) and positive is "widened" (blue), the astronomical sense of the colour.
    It is directly comparable between catalogs of different precision.
``<key>_add.png``
    the additive sigma on its own, in harmonized units -- the quantity that behaves
    like the old ``Delta C``, and the one that says where a fixed floor is not
    enough.

The model is a function of a single entry (``(theta, catalog) -> ln_scale,
ln_add``), so it can be evaluated on any prepared catalog row; the question is only
which rows to show:

``sample="model"`` (default)
    the entries of the star table: stars that at least two catalogs observe, which
    is what the model was fitted to. This is its region of validity.
``sample="all"``
    every prepared row, single-catalog stars included. The network has no notion of
    being outside its training set, so this shows a smooth continuation rather than
    a warning; compare the two to see where it is extrapolating. Between them the
    sample covers 5.39M of 14.79M prepared entries.

The frame (``Teff``/logg limits and bin edges) is taken from the training sample in
both cases, so the two versions are directly comparable and a panel of ``all`` that
is blank outside the frame is telling you something.

Bins with no entries are masked and drawn white; a parameter that a catalog does not
measure gets an empty, annotated column.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import binned_statistic_dd
from tqdm.auto import tqdm

if not os.environ.get("DISPLAY"):  # headless by default
    import matplotlib

    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
from matplotlib.ticker import FuncFormatter

from .. import registry
from .diagnostics import PARAM_NAMES
from .model import LN_MIN, load_delta_model, model_shared, numpy_delta_fn
from .stars import N_PARAMS, get_star_table

log = logging.getLogger(__name__)

#: Four [Fe/H] bins over the range the catalogs cover, fixed for every catalog
FEH_EDGES = np.array([-2.5, -1.75, -1.0, -0.25, 0.5])

#: Which of the four parameters each column shows, in the order of ``PARAM_NAMES``
COLUMNS = tuple(range(N_PARAMS))

#: The additive sigma that stands for "nothing is added" (the model's smooth floor)
ADD_MIN = float(np.exp(0.5 * LN_MIN))


@dataclass
class KielCube:
    """Per catalog, parameter, [Fe/H] row and Kiel pixel: the two corrections.

    ``log_ratio`` is the median of ``ln(sigma_out / sigma_in)`` = the dimensionless
    inflation of the quoted error, which is what the merge sees (0 = unchanged,
    negative = shrunk); ``sigma_add`` is the median additive term on its own, in
    harmonized units, which is the quantity that behaves like the old ``Delta C``.
    """

    teff_edges: np.ndarray
    logg_edges: np.ndarray
    feh_edges: np.ndarray
    #: ``(n_cat, 4, n_feh, n_teff, n_logg)``; NaN where a bin has no entries
    log_ratio: np.ndarray
    sigma_add: np.ndarray
    #: entries per bin, same shape
    count: np.ndarray
    #: ``(n_cat, 4)``: does this catalog measure that parameter at all
    measures: np.ndarray
    keys: Tuple[str, ...]
    sample: str

    def limits(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        return ((self.teff_edges[0], self.teff_edges[-1]),
                (self.logg_edges[0], self.logg_edges[-1]))

    def values(self, kind: str) -> np.ndarray:
        if kind == "ratio":
            return self.log_ratio
        if kind == "add":
            return self.sigma_add
        raise ValueError(f"unknown kind {kind!r}: use 'ratio' or 'add'")


def _entries(run_dir: Path, sample: str, keys: Sequence[str], out_dir: Optional[Path]):
    """Yield ``(catalog index, x, ln sigma_x, measured)`` for one catalog at a time."""
    from ..apply import padded_inputs
    from ..training_data import load_catalog

    if sample not in ("model", "all"):
        raise ValueError(f"unknown sample {sample!r}: use 'model' or 'all'")
    if sample == "model":
        stars, _ = get_star_table(run_dir, out_dir)
        c = stars["c"]
        for k in range(len(keys)):
            sel = np.flatnonzero(c == k)
            yield k, stars["x"][sel], stars["ls"][sel], stars["m"][sel]
        return
    for k, key in enumerate(keys):
        t = load_catalog(key)
        x, sigma, measured = padded_inputs(t)
        # The encoding the model was trained with: 0 (ln 1) on unmeasured axes, not
        # ln(1e9), which is a value it has never seen.
        yield k, x, np.log(np.where(measured, sigma, 1.0)).astype("f4"), measured


def _frame(stars_teff: np.ndarray, stars_logg: np.ndarray, n_bins: Tuple[int, int]):
    """Bin edges shared by every catalog and both samples, from the training stars."""
    def edges(v, n):
        lo, hi = np.percentile(v[np.isfinite(v)], [1.0, 99.0])
        w = hi - lo
        return np.linspace(lo - 0.05 * w, hi + 0.05 * w, n + 1)

    return edges(stars_teff, n_bins[0]), edges(stars_logg, n_bins[1])


def build_kiel_cube(run_dir, model=None, sample: str = "model", keys=None,
                    n_bins: Tuple[int, int] = (80, 70), batch_size: int = 262144,
                    out_dir: Optional[Path] = None) -> KielCube:
    """Median ``Delta C`` in a (Teff, logg, [Fe/H]) cube for every catalog.

    Entries are streamed through the model in large batches and reduced into the
    cube as they come, so the memory is the cube itself (a few hundred kB per
    catalog), not the per-entry variances.
    """
    run_dir = Path(run_dir)
    out_dir = Path(out_dir) if out_dir else run_dir / "delta"
    keys = tuple(keys) if keys is not None else registry.run_keys(run_dir)
    n_cat = len(keys)
    model = model if model is not None else load_delta_model(out_dir, keys=keys)
    delta_fn = numpy_delta_fn(model, batch_size=batch_size)

    stars, _ = get_star_table(run_dir, out_dir)
    teff_edges, logg_edges = _frame(stars["x"][:, 0], stars["x"][:, 2], n_bins)
    del stars

    # The model's category is the group; the shared systematic of a group is part of
    # the error its catalogs have when compared with the rest of the sky.
    group_of = registry.group_index(keys)
    shared = model_shared(model)
    n_feh = len(FEH_EDGES) - 1
    shape = (N_PARAMS, n_feh, len(teff_edges) - 1, len(logg_edges) - 1)
    log_ratio = np.full((n_cat,) + shape, np.nan, dtype="f4")
    sigma_add = np.full((n_cat,) + shape, np.nan, dtype="f4")
    count = np.zeros((n_cat,) + shape, dtype="i4")
    measures = np.zeros((n_cat, N_PARAMS), dtype=bool)

    for k, x, ls, measured in tqdm(_entries(run_dir, sample, keys, out_dir),
                                   total=n_cat, desc=f"delta cube ({sample})"):
        # A panel needs Teff, logg and [Fe/H]: they are the axes and the rows.
        base = measured[:, 0] & measured[:, 2] & measured[:, 1]
        if not base.any():
            log.warning("%s: no entry measures Teff, logg and [Fe/H]; skipped",
                        registry.get_spec(keys[k]).label)
            continue
        ln_scale, ln_add = delta_fn(x, ls, np.full(len(x), group_of[k], dtype="i4"))
        shared_k = 0.0 if shared is None else shared[group_of[k]]
        # sigma_out / sigma_in, per parameter: 1 means the quoted error is left alone.
        # The group's shared variance is in absolute units, so it is divided by the
        # entry's own variance to become part of the ratio; the additive page below
        # keeps the model's own additive term, since the shared part is a constant of
        # the group and would only shift every panel's colour scale.
        c_in = np.exp(2.0 * ls.astype("f8"))
        var = ((np.exp(ln_scale.astype("f8")) * c_in + np.exp(ln_add.astype("f8"))
                + shared_k) / c_in)
        lr = 0.5 * np.log(np.maximum(var, 1e-300))
        sa = np.exp(0.5 * ln_add.astype("f8"))
        for p in range(N_PARAMS):
            sel = base & measured[:, p]
            if sel.sum() < 10:
                continue
            measures[k, p] = True
            sample3 = np.stack([x[sel, 0], x[sel, 2], x[sel, 1]], axis=1)  # Teff, logg, [Fe/H]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # empty bins -> NaN
                for dest, arr in ((log_ratio, lr), (sigma_add, sa)):
                    stat, _, _ = binned_statistic_dd(
                        sample3, arr[sel, p], statistic="median",
                        bins=[teff_edges, logg_edges, FEH_EDGES],
                    )
                    dest[k, p] = np.moveaxis(stat, -1, 0).astype("f4")  # ([Fe/H], Teff, logg)
                n, _, _ = binned_statistic_dd(
                    sample3, lr[sel, p], statistic="count",
                    bins=[teff_edges, logg_edges, FEH_EDGES],
                )
            count[k, p] = np.moveaxis(n, -1, 0).astype("i4")

    return KielCube(teff_edges, logg_edges, FEH_EDGES.copy(), log_ratio, sigma_add,
                    count, measures, keys, sample)


def _ratio_limit(cube: KielCube, k: Optional[int] = None, percentile: float = 99.5) -> float:
    """Symmetric colour limit for ``ln(sigma_out/sigma_in)``.

    Centred on 0 (no change) by construction, so the diverging palette is honest;
    the limit is a high percentile of ``|value|`` so a few bins cannot flatten it.
    """
    v = cube.log_ratio if k is None else cube.log_ratio[k]
    v = v[np.isfinite(v)]
    lim = float(np.percentile(np.abs(v), percentile)) if v.size else 0.1
    return max(lim, 0.05)


def _add_scale(cube: KielCube, k: Optional[int] = None,
               percentile: float = 99.5) -> Tuple[float, float]:
    """Colour limits for the additive sigma: from the floor to a high percentile."""
    v = cube.sigma_add if k is None else cube.sigma_add[k]
    v = v[np.isfinite(v)]
    vmax = float(np.percentile(v, percentile)) if v.size else 2 * ADD_MIN
    return ADD_MIN, max(vmax, 2 * ADD_MIN)


def _scale(cube: KielCube, k: Optional[int] = None,
           percentile: float = 99.5) -> Tuple[float, float]:
    """Backwards-compatible alias for the additive colour limits."""
    return _add_scale(cube, k, percentile)


def _tick(v: float, _pos=None) -> str:
    """One decimal, and no "-0.0" on a reversed axis."""
    return "0.0" if abs(v) < 0.05 else f"{v:.1f}"


def plot_catalog_cube(cube: KielCube, k: int, label: str, path: Path,
                      vmin: float, vmax: float, min_count: int = 1,
                      dpi: int = 120, kind: str = "ratio") -> Path:
    """One catalog: 4x4 panels, a single colourbar, a fixed layout.

    ``kind="ratio"`` draws ``ln(sigma_out/sigma_in)`` on a diverging palette centred
    on 0; ``kind="add"`` draws the additive sigma on a log palette starting at the
    floor. The axes, rows, titles and layout are identical, so the two pages of a
    catalog line up panel for panel.
    """
    (teff_lo, teff_hi), (logg_lo, logg_hi) = cube.limits()
    extent = (teff_lo, teff_hi, logg_lo, logg_hi)
    values = cube.values(kind)
    if kind == "ratio":
        norm = TwoSlopeNorm(vcenter=0.0, vmin=-abs(vmin), vmax=abs(vmax))
        # coolwarm_r: blue for a larger error (the astronomical direction, as in
        # blueshift), red for a smaller one, and a grey neutral at zero so that
        # "unchanged" cannot be confused with the white "no data" mask.
        cmap, cbar_label = "coolwarm_r", r"$\ln(\sigma_{\rm out}/\sigma_{\rm in})$"
        title = f"{label}: learned error inflation ({cube.sample} sample)"
    else:
        norm = LogNorm(vmin=vmin, vmax=vmax)
        cmap, cbar_label = "turbo", r"$\sigma_{\rm add}$ (harmonized units)"
        title = f"{label}: learned additive error ({cube.sample} sample)"

    fig = plt.figure(figsize=(13.5, 12.0))
    margins = dict(left=0.085, right=0.885, bottom=0.055, top=0.925,
                   wspace=0.06, hspace=0.08)
    gs = fig.add_gridspec(4, 4, **margins)
    fig.subplots_adjust(**margins)

    im = None
    n_feh = len(cube.feh_edges) - 1
    for row in range(n_feh):
        feh = n_feh - 1 - row  # [Fe/H] descends down the figure: metal-rich on top
        for col in range(len(COLUMNS)):
            ax = fig.add_subplot(gs[row, col])
            img = values[k, col, feh].T  # rows = logg, columns = Teff
            img = np.ma.masked_invalid(img)
            img = np.ma.masked_where(cube.count[k, col, feh].T < min_count, img)
            if img.mask.all():
                note = "not measured" if not cube.measures[k, col] else "no stars"
                if not cube.measures[k, col] and row > 0:
                    note = ""  # already said in the top panel of this column
                ax.text(0.5, 0.5, note, ha="center", va="center",
                        transform=ax.transAxes, color="0.6", fontsize=8)
            im = ax.imshow(img, origin="lower", extent=extent, aspect="auto",
                           cmap=cmap, norm=norm, interpolation="nearest")
            ax.set_xlim(teff_hi, teff_lo)  # hot to the left, like the video planes
            ax.set_ylim(logg_hi, logg_lo)
            ax.xaxis.set_major_formatter(FuncFormatter(_tick))
            ax.tick_params(labelsize=6)
            if row == 0:
                ax.set_title(PARAM_NAMES[col], fontsize=10)
            if col == 0:
                lo, hi = cube.feh_edges[feh], cube.feh_edges[feh + 1]
                ax.set_ylabel(f"[Fe/H]\n{lo:+.2f} … {hi:+.2f}", fontsize=8)
            else:
                ax.set_yticklabels([])
            if row < n_feh - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(r"$5\ln(T_{\rm eff}/5000\,{\rm K})$", fontsize=8)

    cax = fig.add_axes([0.905, 0.10, 0.018, 0.80])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(cbar_label, fontsize=9)
    cb.ax.tick_params(labelsize=7)
    fig.suptitle(title, fontsize=11)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def make_kiel_figures(run_dir, out_dir=None, model=None, sample: str = "model",
                      keys=None, n_bins: Tuple[int, int] = (80, 70),
                      batch_size: int = 262144, min_count: int = 1,
                      scale: str = "catalog", dpi: int = 120) -> List[Path]:
    """Build the cube and draw two pages per catalog; returns the paths."""
    run_dir = Path(run_dir)
    delta_dir = run_dir / "delta"
    out_dir = Path(out_dir) if out_dir else delta_dir / "kiel"
    out_dir.mkdir(parents=True, exist_ok=True)
    cube = build_kiel_cube(run_dir, model, sample, keys, n_bins, batch_size, delta_dir)
    ratio_global = _ratio_limit(cube)
    add_global = _add_scale(cube)[1]
    limits = {}
    for k, key in enumerate(cube.keys):
        if scale == "global":
            limits[key] = (_ratio_limit(cube), add_global)
        else:
            limits[key] = (_ratio_limit(cube, k), _add_scale(cube, k)[1])
    json.dump(dict(sample=sample, keys=list(cube.keys), scale=scale,
                   teff_edges=cube.teff_edges.tolist(), logg_edges=cube.logg_edges.tolist(),
                   feh_edges=cube.feh_edges.tolist(),
                   ratio_limit_global=ratio_global, add_vmax_global=add_global,
                   limits={key: list(v) for key, v in limits.items()}),
              open(out_dir / "kiel_cube.json", "w"), indent=1)
    np.savez_compressed(out_dir / "kiel_cube.npz", log_ratio=cube.log_ratio,
                        sigma_add=cube.sigma_add, count=cube.count,
                        measures=cube.measures, teff_edges=cube.teff_edges,
                        logg_edges=cube.logg_edges, feh_edges=cube.feh_edges)
    paths = []
    for k, key in enumerate(cube.keys):
        if not cube.measures[k].any():
            log.warning("%s: nothing to plot", registry.get_spec(key).label)
            continue
        ratio_lim, add_max = limits[key]
        label = registry.get_spec(key).label
        for kind, vmin, vmax, suffix in (("ratio", -ratio_lim, ratio_lim, ""),
                                         ("add", ADD_MIN, add_max, "_add")):
            paths.append(plot_catalog_cube(
                cube, k, label, out_dir / f"{key}{suffix}.png", vmin, vmax,
                min_count, dpi, kind=kind,
            ))
        log.info("Wrote %s", out_dir / f"{key}.png")
    return paths


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="astrotune-delta-kiel",
        description="Per-catalog maps of the learned extra variance in the Kiel diagram.",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None, help="default: <run-dir>/delta/kiel")
    parser.add_argument("--all", action="store_true",
                        help="use every prepared row, not just the stars the model was fitted on")
    parser.add_argument("--catalogs", nargs="+", metavar="KEY", default=None)
    parser.add_argument("--bins", nargs=2, type=int, default=(80, 70), metavar=("TEFF", "LOGG"))
    parser.add_argument("--batch-size", type=int, default=262144)
    parser.add_argument("--min-count", type=int, default=1,
                        help="entries a bin needs to be drawn")
    parser.add_argument("--scale", choices=("catalog", "global"), default="catalog",
                        help="colour limits per catalog, or one range for all figures")
    parser.add_argument("--dpi", type=int, default=120)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    paths = make_kiel_figures(
        args.run_dir, args.out_dir, sample="all" if args.all else "model",
        keys=args.catalogs, n_bins=tuple(args.bins), batch_size=args.batch_size,
        min_count=args.min_count, scale=args.scale, dpi=args.dpi,
    )
    print(f"{len(paths)} figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
