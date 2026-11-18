"""What the correction does to the *distribution* of the quoted uncertainties.

The calibration statistics say whether the errors are right on average; they do not
say whether the corrected errors are plausible as errors. A per-parameter scale of
0.13 shrinks every [alpha/Fe] error by a factor of eight, which is a strong claim
about a quantity whose quoted spread is ~0.05 -- so this module draws, for each
catalog and parameter, the quoted uncertainties against the calibrated ones on the
same bins.

Per entry the diagonal of the covariance becomes ``S_ii = scale_i C_ii + add_i``
(the congruence leaves the diagonal scaled by ``s_i = exp(ln_scale / 2)``), so the
two distributions are ``sqrt(C_ii)`` and ``sqrt(scale_i C_ii + add_i)``.

The default sample is the training stars, i.e. the entries the model was fitted on.
``--all`` uses every prepared row instead (one row per star and catalog, as in the
star table), so single-catalog stars are included; the transformation is the one
``astrotune-apply`` cached, so no model run is needed for that.

Run with ``python -m astrotune.delta.uncertainties --run-dir models/X``.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Iterator, Optional, Sequence, Tuple

import numpy as np
from tqdm.auto import tqdm

if not os.environ.get("DISPLAY"):  # headless by default
    import matplotlib

    matplotlib.use("Agg")

import matplotlib.pyplot as plt

from .. import registry
from ..config import ERROR_FLOORS
from .diagnostics import PARAM_NAMES
from .model import load_delta_model, numpy_delta_fn
from .stars import N_PARAMS, get_star_table, split

log = logging.getLogger(__name__)


def _entries(run_dir: Path, sample: str, keys: Sequence[str],
             out_dir: Optional[Path] = None) -> Iterator[Tuple[int, np.ndarray, np.ndarray,
                                                                np.ndarray, np.ndarray]]:
    """Yield ``(catalog, x, ln sigma_x, diag(C), measured)`` one catalog at a time."""
    if sample == "train":
        stars, star_keys = get_star_table(run_dir, out_dir)
        keys = tuple(keys) if keys is not None else star_keys
        train, _ = split(stars)
        for k in range(len(keys)):
            sel = train["c"] == k
            yield (k, train["x"][sel], train["ls"][sel],
                   np.diagonal(train["C"][sel], axis1=1, axis2=2).astype("f8"), train["m"][sel])
        return
    if sample != "all":
        raise ValueError(f"unknown sample {sample!r}: use 'train' or 'all'")

    from ..apply import load_cached, padded_inputs
    from ..training_data import load_tables

    run_dir = Path(run_dir)
    cache_dir = run_dir / "harmonized" / "transform_cache_none"
    if not cache_dir.exists():
        raise FileNotFoundError(
            f"{cache_dir} does not exist; run `astrotune-apply --run-dir {run_dir}` first.")
    tables = load_tables(keys, dedupe="none")
    for k, (key, t) in enumerate(tqdm(list(zip(keys, tables)), desc="uncertainties (all)")):
        est = load_cached(cache_dir, registry.category_index(key), np.asarray(t["id"]), t)
        if est is None:
            raise FileNotFoundError(f"No usable cached transformation for {key} in {cache_dir}.")
        # The same row selection as the star table (first valid row per star), so the
        # entries here are the ones the model would see.
        ids = np.asarray(t["id"])
        x, sigma, measured = padded_inputs(t)
        ok = np.flatnonzero(np.isfinite(est.y).all(1) & np.isfinite(est.cov).all((1, 2)))
        _, first = np.unique(ids[ok], return_index=True)
        rows = ok[first]
        ln_s = np.log(np.where(measured, sigma, 1.0)).astype("f4")
        yield (k, x[rows], ln_s[rows],
               np.diagonal(est.cov[rows], axis1=1, axis2=2).astype("f8"), est.measured[rows])


def uncertainty_distributions(run_dir, model=None, sample: str = "train", keys=None,
                              out_dir: Optional[Path] = None, stars=None,
                              min_entries: int = 100) -> dict:
    """Quoted and calibrated uncertainties per catalog and parameter.

    Returns ``{catalog index: (raw, calibrated, measured)}`` with ``(n, 4)`` arrays.
    """
    run_dir = Path(run_dir)
    delta_dir = run_dir / "delta"
    keys = tuple(keys) if keys is not None else registry.run_keys(run_dir)
    model = model if model is not None else load_delta_model(delta_dir, keys=keys)
    delta_fn = numpy_delta_fn(model)

    out = {}
    for k, x, ls, c_diag, measured in _entries(run_dir, sample, keys, delta_dir):
        ln_scale, ln_add = delta_fn(x, ls, np.full(len(x), k, dtype="i4"))
        raw = np.sqrt(np.maximum(c_diag, 0.0))
        cal = np.sqrt(np.exp(ln_scale.astype("f8")) * c_diag + np.exp(ln_add.astype("f8")))
        out[k] = (raw, cal, measured)
    return out


def plot_catalog_uncertainty(k: int, label: str, raw: np.ndarray, cal: np.ndarray,
                             measured: np.ndarray, path: Path, sample: str = "train",
                             min_entries: int = 100, dpi: int = 110) -> Optional[Path]:
    """One catalog, one row: quoted against calibrated uncertainties per parameter."""
    params = [p for p in range(N_PARAMS) if int(measured[:, p].sum()) >= min_entries]
    if not params:
        log.warning("%s: no parameter has %d entries", label, min_entries)
        return None
    fig, axes = plt.subplots(1, len(params), figsize=(3.7 * len(params), 4.0),
                            squeeze=False, layout="constrained")
    for ax, p in zip(axes[0], params):
        sel = measured[:, p]
        r, c = raw[sel, p], cal[sel, p]
        lo = np.percentile(np.concatenate([r, c]), 0.2)
        hi = np.percentile(np.concatenate([r, c]), 99.8)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            ax.set_axis_off()
            continue
        bins = np.geomspace(max(lo, 1e-6), hi, 70)
        for values, color, name in ((r, "C0", "quoted"), (c, "C3", "calibrated")):
            ax.hist(values, bins=bins, histtype="step", lw=1.3, density=True,
                    color=color, label=f"{name}: median {np.median(values):.3g}")
        ax.axvline(ERROR_FLOORS[p], color="0.6", ls=":", lw=1.0, label="merge floor")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(bottom=1e-3)
        ax.set_title(f"{PARAM_NAMES[p]}  x{np.median(c) / np.median(r):.2f}", fontsize=10)
        ax.set_xlabel("uncertainty (harmonized units)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=6, loc="upper left")
    axes[0][0].set_ylabel("density per bin", fontsize=8)
    fig.suptitle(f"{label}: uncertainties before and after the correction "
                 f"({sample} sample, {len(raw):,} entries)", fontsize=10)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def make_uncertainty_plots(run_dir, out_dir=None, model=None, sample: str = "train",
                           keys=None, min_entries: int = 100, dpi: int = 110,
                           dists: Optional[dict] = None) -> list:
    """Draw one figure per catalog; returns the paths."""
    run_dir = Path(run_dir)
    out_dir = Path(out_dir) if out_dir else run_dir / "delta" / "uncertainty_distribution"
    keys = tuple(keys) if keys is not None else registry.run_keys(run_dir)
    dists = dists if dists is not None else uncertainty_distributions(
        run_dir, model, sample, keys, min_entries=min_entries)
    paths = []
    for k, key in enumerate(keys):
        if k not in dists:
            continue
        raw, cal, measured = dists[k]
        path = plot_catalog_uncertainty(k, registry.get_spec(key).label, raw, cal, measured,
                                        out_dir / f"{key}.png", sample, min_entries, dpi)
        if path is not None:
            paths.append(path)
            log.info("Wrote %s", path)
    return paths


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="astrotune-delta-uncertainty",
        description="Distribution of the quoted and calibrated uncertainties, per catalog.",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="default: <run-dir>/delta/uncertainty_distribution")
    parser.add_argument("--all", action="store_true",
                        help="every prepared row, not just the training stars")
    parser.add_argument("--catalogs", nargs="+", metavar="KEY", default=None)
    parser.add_argument("--min-entries", type=int, default=100)
    parser.add_argument("--dpi", type=int, default=110)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    paths = make_uncertainty_plots(
        args.run_dir, args.out_dir, sample="all" if args.all else "train",
        keys=args.catalogs, min_entries=args.min_entries, dpi=args.dpi,
    )
    print(f"{len(paths)} figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
