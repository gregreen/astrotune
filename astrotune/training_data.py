"""Training data: stars observed in more than one catalog, plus control points.

The loss has one free harmonized parameter vector ``y`` per star (the
"latent-y" model), so the training data are *observations*: one row per catalog
measurement, carrying the Gaia source id that ties a star's rows to its ``y``.
Only stars that at least two *different* catalogs observe are kept, because a
single measurement leaves ``y`` free (its best fit is exact) and therefore says
nothing about the maps. Duplicate Gaia ids inside one catalog are kept, as the
project does not dedupe by default; each is a separate observation of that star.

*Control points* are sampled from each catalog (and from a wide Gaussian
covering the whole parameter space) and regularize the transformations: the
same mixture as before, with the number of points tied to the number of
observations instead of the number of pairs.

Everything here is NumPy/astropy only (no TensorFlow). The pair-based builder
that reproduced cells 347-357 of the reference notebook was removed together
with the pair loss; the control points are drawn from the same distributions,
but the file is not row-for-row comparable with the legacy
``pairs_and_ctrl_points.h5`` (the pair stage used to consume part of the RNG).

    python -m astrotune.training_data            # build and save
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import astropy.table
import h5py
import numpy as np
from tqdm.auto import tqdm

from . import config, registry
from .prep.io import load_catalog

log = logging.getLogger(__name__)

N_PARAMS = config.N_PARAMS

#: One row per catalog measurement of a star: Gaia id, parameters,
#: regularization error (``xe``), statistical uncertainty (``xs``), catalog index
OBS_KEYS = ("id", "x", "xe", "xs", "c")
CTRL_KEYS = ("x_control", "xe_control", "c_control", "from_catalog")

#: Regularization error substituted for "unknown" (>1e6) values when loading
UNKNOWN_XE = 0.05

#: ``x_err`` stored for a parameter that a catalog does not measure
MISSING_XE = 1e9

#: Regularization error of the dispersed control points in the legacy analysis
LEGACY_DISPERSED_XE = 0.2

#: How the ``x_err`` of the dispersed control points is chosen
DISPERSED_XERR_MODES = ("per-catalog", "uniform")

#: Seed used for the published run
DEFAULT_SEED = 3

_COLUMNS = ("id", "x", "x_err", "sigma_x")


def dedupe_by_id(table: astropy.table.Table) -> astropy.table.Table:
    """Keep only the first row for every Gaia id."""
    _, first = np.unique(np.asarray(table["id"]), return_index=True)
    return table[np.sort(first)]


def load_tables(
    keys: Optional[Sequence[str]] = None, *, dedupe: str = "none"
) -> List[astropy.table.Table]:
    """Load the prepared catalog tables, in registry order.

    ``dedupe="none"`` keeps duplicate Gaia ids exactly as the legacy notebook
    did, so a star can have several observations from the same catalog.
    ``dedupe="first"`` keeps one row per id in every catalog.
    """
    if dedupe not in ("none", "first"):
        raise ValueError(f"dedupe must be 'none' or 'first', not {dedupe!r}")
    keys = registry.keys() if keys is None else tuple(keys)
    tables = []
    for key in keys:
        t = load_catalog(key)[list(_COLUMNS)]
        if dedupe == "first":
            n = len(t)
            t = dedupe_by_id(t)
            log.info("%s: dropped %d duplicate-id rows", key, n - len(t))
        tables.append(t)
    return tables


def ids_in_several_catalogs(tables) -> np.ndarray:
    """Sorted Gaia ids that at least two *different* catalogs observe.

    Duplicates of an id inside one catalog do not count as two catalogs: a
    single catalog's repeat measurements add information about that catalog's
    error, not about the maps.
    """
    per_catalog = [np.unique(np.asarray(t["id"], dtype="i8")) for t in tables]
    ids, n_catalogs = np.unique(np.concatenate(per_catalog), return_counts=True)
    return ids[n_catalogs >= 2]


def extract_observations(table, c: int, ids: np.ndarray) -> Dict[str, np.ndarray]:
    """The rows of ``table`` (catalog ``c``) whose star is in ``ids``.

    Each row is one catalog measurement: the Gaia id that ties the observations
    of a star to its free harmonized parameters, the parameters, the
    regularization error (``xe``), the statistical uncertainty (``xs``) and the
    catalog index. Missing parameters are padded to 4 columns with ``x = 0`` and
    ``xe = xs = 1e9``, as the pair builder did.
    """
    idx = np.flatnonzero(np.isin(np.asarray(table["id"]), ids))
    x_in = np.asarray(table["x"])
    d = x_in.shape[1]
    x = np.zeros((len(idx), N_PARAMS), dtype="f4")
    xe = np.full((len(idx), N_PARAMS), MISSING_XE, dtype="f4")
    xs = np.full((len(idx), N_PARAMS), MISSING_XE, dtype="f4")
    x[:, :d] = x_in[idx]
    xe[:, :d] = np.asarray(table["x_err"])[idx]
    xs[:, :d] = np.asarray(table["sigma_x"])[idx]
    return dict(
        id=np.asarray(table["id"], dtype="i8")[idx],
        x=x,
        xe=xe,
        xs=xs,
        c=np.full(len(idx), c, dtype="i4"),
    )


def build_observations(tables) -> Dict[str, np.ndarray]:
    """Every observation of a star that at least two catalogs observe.

    The rows are sorted by Gaia id and then by catalog, so the observations of
    one star are contiguous and can be sliced or gathered as a block.
    """
    if len(tables) < 2:
        raise ValueError("at least two catalogs are needed to match stars")
    ids = ids_in_several_catalogs(tables)
    log.info("%d stars are observed by at least two catalogs", len(ids))

    obs: Dict[str, list] = {key: [] for key in OBS_KEYS}
    for c, t in enumerate(tables):
        res = extract_observations(t, c, ids)
        log.info("Catalog %d: %d observations", c, len(res["id"]))
        for key in OBS_KEYS:
            obs[key].append(res[key])

    out = {key: np.concatenate(obs[key], axis=0) for key in OBS_KEYS}
    order = np.lexsort((out["c"], out["id"]))
    out = {key: value[order] for key, value in out.items()}
    log.info("%d observations of %d stars", len(out["id"]), len(ids))
    if len(out["id"]) == 0:
        log.warning("no star is observed by two catalogs; the training data are empty")
    return out


def count_observations_per_catalog(obs_dict, n_cat: int) -> np.ndarray:
    """Number of observations that come from each catalog."""
    return np.bincount(obs_dict["c"], minlength=n_cat)


def dispersed_x_err(tables, keys: Optional[Sequence[str]] = None) -> List[np.ndarray]:
    """Regularization error of the dispersed points of each catalog, ``(4,)`` each.

    The dispersed points are the prior in the part of parameter space where a
    catalog has no data, so they use one value per catalog and parameter, not a
    function of position: the functional forms of the catalogs' ``x_err`` are
    only meaningful where they have data, and the real control points already
    carry them there. The value is the median ``x_err`` of the catalog's stars
    that measure the parameter, so it follows the catalog; a catalog module can
    override it with ``X_ERR_DISPERSED``. A parameter that no star of the
    catalog measures stays :data:`MISSING_XE`, which :func:`load_observations`
    turns into the strong pin that the catalog's real control points have.

    ``keys`` are the catalog keys (to find the overrides); omit them to use the
    medians only.
    """
    out = []
    for i, t in enumerate(tables):
        xe = np.asarray(t["x_err"], dtype="f8")
        value = np.full(N_PARAMS, MISSING_XE)
        for j in range(xe.shape[1]):
            measured = xe[:, j][xe[:, j] < 1e6]
            if len(measured):
                value[j] = np.median(measured)
        if keys is not None:
            override = getattr(registry.get_module(keys[i]), "X_ERR_DISPERSED", None)
            if override is not None:
                override = np.asarray(override, dtype="f8")
                value[: len(override)] = override
        out.append(value)
    return out


def build_control_points(
    tables, obs_dict, rng, *, dispersed_xerr: str = "per-catalog", keys=None
) -> Dict[str, np.ndarray]:
    """Control points from each catalog, plus dispersed points covering the space.

    The mixture is the same as in the pair-based version. For each catalog the
    number of control points is the least of

      * (# of observations) / (# of catalogs - 1), which makes the number of
        control points roughly equal to the number of observations (the data
        term of the latent-y loss has one residual per observation, so this
        keeps the balance between the data term and the penalties that the
        pair loss had);
      * twice the number of observations from the catalog, so that a catalog is
        not represented much more than its observations are;
      * the number of rows in the catalog.

    (with a floor of 1024 on the second term). Additionally, ``n_max // 16``
    dispersed points per catalog are drawn from a wide Gaussian built from the
    union of the 1-99 percentile boxes of all catalogs. Their ``x_err`` is
    chosen per catalog (:func:`dispersed_x_err`), or is the legacy uniform 0.2
    with ``dispersed_xerr="uniform"``; the positions do not depend on the mode.
    """
    if dispersed_xerr not in DISPERSED_XERR_MODES:
        raise ValueError(f"dispersed_xerr must be one of {DISPERSED_XERR_MODES}")
    n_tab = len(tables)
    n_cat = count_observations_per_catalog(obs_dict, n_tab)
    ctrl: Dict[str, list] = {k: [] for k in CTRL_KEYS}

    n_max = obs_dict["x"].shape[0] // (n_tab - 1)

    for i, t in enumerate(tables):
        n_sel = min(n_max, max(2 * n_cat[i], 1024), len(t))
        log.info("Extracting %d control points from catalog %d", n_sel, i)
        idx = np.arange(len(t))
        rng.shuffle(idx)
        idx = idx[:n_sel]
        x_i = np.asarray(t["x"][idx]).astype("f4")
        xe_i = np.asarray(t["x_err"][idx]).astype("f4")
        n, d = x_i.shape
        if d < N_PARAMS:  # pad missing [alpha/Fe]
            x_i = np.concatenate([x_i, np.zeros((n, N_PARAMS - d), dtype="f4")], axis=1)
            xe_i = np.concatenate([xe_i, np.full((n, N_PARAMS - d), 1e9, dtype="f4")], axis=1)
        if np.any(np.isnan(x_i)):
            log.warning("NaNs detected in control points of catalog %d", i)
        ctrl["x_control"].append(x_i)
        ctrl["xe_control"].append(xe_i)
        ctrl["c_control"].append(np.full(len(idx), i, dtype="i4"))
        ctrl["from_catalog"].append(np.ones(len(idx), dtype="bool"))

    # Dispersed points covering the whole parameter space
    n_draw = n_max // 16
    if dispersed_xerr == "per-catalog":
        disp_xe = dispersed_x_err(tables, keys)
    else:
        disp_xe = [np.full(N_PARAMS, LEGACY_DISPERSED_XE)] * n_tab
    x_pct = []
    for t in tables:
        x_pct_i = np.percentile(np.asarray(t["x"]), [1.0, 99.0], axis=0)
        if x_pct_i.shape[1] < N_PARAMS:  # missing columns are ignored by nanmin/nanmax
            pad = np.full((2, N_PARAMS - x_pct_i.shape[1]), np.nan, dtype="f4")
            x_pct_i = np.concatenate([x_pct_i, pad], axis=1)
        x_pct.append(x_pct_i)
    x_pct = np.array(x_pct)
    x_low = np.nanmin(x_pct[:, 0], axis=0)
    x_high = np.nanmax(x_pct[:, 1], axis=0)
    x_mean = 0.5 * (x_high + x_low)
    x_width = 0.5 * (x_high - x_low)
    for i in range(n_tab):
        log.info("Generating %d dispersed points for catalog %d", n_draw, i)
        x_draw = x_mean[None, :] + x_width[None, :] * rng.normal(size=(n_draw, len(x_mean)))
        x_draw = x_draw.astype("f4")
        ctrl["x_control"].append(x_draw)
        ctrl["xe_control"].append(np.tile(disp_xe[i].astype("f4"), (n_draw, 1)))
        ctrl["c_control"].append(np.full(len(x_draw), i, dtype="i4"))
        ctrl["from_catalog"].append(np.zeros(len(x_draw), dtype="bool"))

    n = sum(len(x) for x in ctrl["x_control"])
    idx = np.arange(n)
    rng.shuffle(idx)
    return {key: np.concatenate(ctrl[key], axis=0)[idx] for key in CTRL_KEYS}


def build_training_data(
    tables=None,
    *,
    seed: int = DEFAULT_SEED,
    dedupe: str = "none",
    dispersed_xerr: str = "per-catalog",
) -> Tuple[dict, dict]:
    """Build ``(obs_dict, ctrl_dict)`` from the catalogs.

    The observations are deterministic; the RNG drives the control-point stage
    only, so ``seed`` sets which control points are drawn (and, through
    ``X_ERR_DISPERSED``, not their errors). ``dispersed_xerr="uniform"`` gives
    the legacy uniform 0.2 instead of the per-catalog values.
    """
    keys = None
    if tables is None:
        keys = registry.keys()
        tables = load_tables(keys, dedupe=dedupe)
    rng = np.random.default_rng(seed)
    obs_dict = build_observations(tables)
    ctrl_dict = build_control_points(
        tables, obs_dict, rng, dispersed_xerr=dispersed_xerr, keys=keys
    )
    return obs_dict, ctrl_dict


def save_observations(
    obs_dict, ctrl_dict, path: Optional[Path] = None, attrs: Optional[dict] = None
) -> Path:
    """Write the file; ``attrs`` (seed, dedupe, dispersed_xerr, ...) are stored as attributes."""
    path = Path(path or config.OBSERVATIONS_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    kw = dict(compression="gzip", compression_opts=3, chunks=True)
    with h5py.File(path, "w") as f:
        for key, value in (attrs or {}).items():
            f.attrs[key] = value
        for key in tqdm(obs_dict, desc="Saving observations"):
            f.create_dataset(f"obs_dict/{key}", data=obs_dict[key], **kw)
        for key in tqdm(ctrl_dict, desc="Saving control points"):
            f.create_dataset(f"ctrl_dict/{key}", data=ctrl_dict[key], **kw)
    log.info("Wrote %s", path)
    return path


def load_observations(
    path: Optional[Path] = None, *, fix_unknown_errors: bool = True
) -> Tuple[dict, dict]:
    """Load ``(obs_dict, ctrl_dict)``.

    ``obs_dict["star"]`` is the index of the star's free harmonized vector: the
    observations are grouped by Gaia id, and ``star`` runs from 0 to
    ``obs_dict["n_stars"] - 1`` in the order the ids first appear. It is
    derived here rather than stored, and is what a training step gathers the
    per-star parameters with. The observations are otherwise unchanged:
    ``xs = 1e9`` still means "not measured" and is left to the loss.

    With ``fix_unknown_errors`` (the legacy training default), regularization
    errors ``> 1e6`` of the *control points* (placeholders for missing
    parameters) are replaced by 0.05, which pins the map's missing axis to the
    identity. ``ctrl_dict["missing_control"]`` (boolean, ``(n, n_dim)``)
    always records which control-point parameters were placeholders.
    """
    path = Path(path or config.OBSERVATIONS_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist; build it with `python -m astrotune.training_data`."
        )
    with h5py.File(path, "r") as f:
        obs_dict = {k: f["obs_dict"][k][:] for k in f["obs_dict"].keys()}
        ctrl_dict = {k: f["ctrl_dict"][k][:] for k in f["ctrl_dict"].keys()}

    # The dense index of the free harmonized parameters of every observation.
    stars, star = np.unique(obs_dict["id"], return_inverse=True)
    obs_dict["star"] = star
    obs_dict["n_stars"] = int(len(stars))

    # Remember which parameters are not measured; the fix-up below hides it.
    ctrl_dict["missing_control"] = ctrl_dict["xe_control"] > 1e6

    if fix_unknown_errors:
        ctrl_dict["xe_control"][ctrl_dict["xe_control"] > 1e6] = UNKNOWN_XE
    return obs_dict, ctrl_dict


def read_attrs(path: Optional[Path] = None) -> dict:
    """Attributes stored with the file (empty for files made before they existed)."""
    with h5py.File(Path(path or config.OBSERVATIONS_PATH), "r") as f:
        return {k: (v.item() if hasattr(v, "item") else v) for k, v in f.attrs.items()}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="astrotune-observations",
        description="Build the training observations and control points.",
    )
    parser.add_argument("-o", "--output", type=Path, default=config.OBSERVATIONS_PATH)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--dedupe",
        choices=("none", "first"),
        default="none",
        help="'none' (default) keeps duplicate Gaia ids as separate observations of "
        "the star; 'first' keeps one row per id in every catalog",
    )
    parser.add_argument(
        "--dispersed-xerr",
        choices=DISPERSED_XERR_MODES,
        default="per-catalog",
        help="x_err of the dispersed control points: 'per-catalog' (default) uses each "
        "catalog's median x_err per parameter (or X_ERR_DISPERSED), with parameters a "
        "catalog does not measure pinned; 'uniform' is the legacy 0.2 for all",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    obs_dict, ctrl_dict = build_training_data(
        seed=args.seed, dedupe=args.dedupe, dispersed_xerr=args.dispersed_xerr
    )
    n_stars = len(np.unique(obs_dict["id"]))
    log.info(
        "%d observations of %d stars, %d control points",
        len(obs_dict["x"]),
        n_stars,
        len(ctrl_dict["x_control"]),
    )
    save_observations(
        obs_dict, ctrl_dict, args.output,
        attrs=dict(
            seed=args.seed, dedupe=args.dedupe, dispersed_xerr=args.dispersed_xerr,
            catalogs=",".join(registry.keys()),
            error_floors=",".join(f"{v:g}" for v in config.ERROR_FLOORS),
            n_stars=n_stars,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
