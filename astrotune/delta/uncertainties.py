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
from .covariance import entry_cov_np, sigma_out_np
from .diagnostics import PARAM_NAMES
from .model import load_delta_model, model_shared, numpy_delta_fn
from .stars import N_PARAMS, get_star_table, split

log = logging.getLogger(__name__)

#: The entry arrays these figures need
_FIELDS = ("x", "ls", "sig", "sig0", "J", "m")


def _entries(run_dir: Path, sample: str, keys: Sequence[str],
             out_dir: Optional[Path] = None) -> Iterator[Tuple[int, dict]]:
    """Yield ``(catalog, arrays)`` one catalog at a time.

    ``arrays`` holds the entry's own parameters, its quoted sigma (``sig``), the
    prepared sigma the current pipeline uses (``sig0``), the bijector's Jacobian and
    the measured mask.
    """
    if sample == "train":
        stars, star_keys = get_star_table(run_dir, out_dir)
        keys = tuple(keys) if keys is not None else star_keys
        train, _ = split(stars)
        for k in range(len(keys)):
            sel = train["c"] == k
            yield k, {name: train[name][sel] for name in _FIELDS}
        return
    if sample != "all":
        raise ValueError(f"unknown sample {sample!r}: use 'train' or 'all'")

    from ..apply import make_forward_and_jacobian
    from ..model import load_bijector
    from ..training_data import load_tables
    from .stars import _catalog_entries

    run_dir = Path(run_dir)
    tables = load_tables(keys, dedupe="none")
    fwd_jac = make_forward_and_jacobian(load_bijector(run_dir))
    for k, (key, t) in enumerate(tqdm(list(zip(keys, tables)), desc="uncertainties (all)")):
        c = _catalog_entries(t, key, fwd_jac, registry.category_index(key))
        yield k, {name: c[name] for name in _FIELDS}


def uncertainty_distributions(run_dir, model=None, sample: str = "train", keys=None,
                              out_dir: Optional[Path] = None, stars=None,
                              min_entries: int = 100) -> dict:
    """Quoted and calibrated uncertainties per catalog and parameter.

    Returns ``{catalog index: (raw, calibrated, measured)}`` with ``(n, 4)`` arrays.
    The calibrated sigma includes the group's shared systematic, i.e. it is the error
    the entry has when compared with the rest of the sky -- the one the merge uses.
    """
    run_dir = Path(run_dir)
    delta_dir = run_dir / "delta"
    keys = tuple(keys) if keys is not None else registry.run_keys(run_dir)
    model = model if model is not None else load_delta_model(delta_dir, keys=keys)
    delta_fn = numpy_delta_fn(model)
    group_of = registry.group_index(keys)
    shared = model_shared(model)

    out = {}
    for k, a in _entries(run_dir, sample, keys, delta_dir):
        ln_scale, ln_add = delta_fn(a["x"], a["ls"],
                                    np.full(len(a["x"]), group_of[k], dtype="i4"))
        cal = entry_cov_np(a["sig"], a["J"], ln_scale, ln_add, measured=a["m"])
        if shared is not None:
            ii = np.arange(N_PARAMS)
            cal[:, ii, ii] += shared[group_of[k]]
        raw = entry_cov_np(a["sig0"], a["J"], measured=a["m"])
        out[k] = (sigma_out_np(raw), sigma_out_np(cal), a["m"])
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
