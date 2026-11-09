"""Training data: pairs of stars observed in two catalogs, plus control points.

*Pairs* anchor the catalogs to one another: a star in catalogs ``j`` and ``k``
should be mapped to the same place in the harmonized space. *Control points*
are sampled from each catalog (and from a wide Gaussian covering the whole
parameter space) and are used to regularize the transformations.

Everything here is NumPy/astropy only (no TensorFlow). The builder reproduces
cells 347-357 of the reference notebook, including the stateful RNG, so that
with ``dedupe="none"``, ``dispersed_xerr="uniform"`` and the same catalog tables
the output is identical to the legacy ``pairs_and_ctrl_points.h5``. By default
the ``x_err`` of the dispersed control points is chosen per catalog instead of
the legacy uniform 0.2 (see :func:`dispersed_x_err`); positions are unchanged.

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

PAIR_KEYS = ("x0", "x1", "xe0", "xe1", "xs0", "xs1", "c0", "c1")
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
    did, so the join in :func:`extract_pairs` is many-to-many for them.
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


def extract_pairs(t0, t1, c=(0, 1), rng=None) -> Dict[str, np.ndarray]:
    """Pairs of stars in both ``t0`` and ``t1`` (matched on ``id``).

    Each pair is randomly oriented, so either catalog can be on the left
    (``x0``) or right (``x1``). Missing parameters are padded to 4 columns with
    ``x = 0`` and ``xe = xs = 1e9``.
    """
    if rng is None:
        rng = np.random.default_rng()

    t_joined = astropy.table.join(
        t0, t1, keys="id", join_type="inner", table_names=("0", "1")
    )
    log.info("%d pairs for categories %s", len(t_joined), tuple(c))

    n, d0 = t_joined["x_0"].shape
    _, d1 = t_joined["x_1"].shape

    # Which catalog goes on the left side of the pair
    k = rng.integers(0, 2, size=n)

    x0 = np.full((n, N_PARAMS), 0.0, dtype="f4")
    x1 = np.full((n, N_PARAMS), 0.0, dtype="f4")
    xe0 = np.full((n, N_PARAMS), 1e9, dtype="f4")
    xe1 = np.full((n, N_PARAMS), 1e9, dtype="f4")
    xs0 = np.full((n, N_PARAMS), 1e9, dtype="f4")
    xs1 = np.full((n, N_PARAMS), 1e9, dtype="f4")
    c0 = np.empty((n,), dtype="i4")
    c1 = np.empty((n,), dtype="i4")

    # (x0, x1) = (t0, t1)
    idx = k == 0
    x0[idx, :d0] = t_joined["x_0"][idx]
    xe0[idx, :d0] = t_joined["x_err_0"][idx]
    xs0[idx, :d0] = t_joined["sigma_x_0"][idx]
    c0[idx] = c[0]
    x1[idx, :d1] = t_joined["x_1"][idx]
    xe1[idx, :d1] = t_joined["x_err_1"][idx]
    xs1[idx, :d1] = t_joined["sigma_x_1"][idx]
    c1[idx] = c[1]

    # (x0, x1) = (t1, t0)
    idx = k == 1
    x0[idx, :d1] = t_joined["x_1"][idx]
    xe0[idx, :d1] = t_joined["x_err_1"][idx]
    xs0[idx, :d1] = t_joined["sigma_x_1"][idx]
    c0[idx] = c[1]
    x1[idx, :d0] = t_joined["x_0"][idx]
    xe1[idx, :d0] = t_joined["x_err_0"][idx]
    xs1[idx, :d0] = t_joined["sigma_x_0"][idx]
    c1[idx] = c[0]

    return dict(x0=x0, x1=x1, xe0=xe0, xe1=xe1, xs0=xs0, xs1=xs1, c0=c0, c1=c1)


def build_pairs(tables, rng) -> Dict[str, np.ndarray]:
    """Pairs for every combination of catalogs, concatenated and shuffled."""
    pair_dict: Dict[str, list] = {k: [] for k in PAIR_KEYS}
    n_tab = len(tables)
    for j in range(n_tab):
        for k in range(j + 1, n_tab):
            res = extract_pairs(tables[j], tables[k], c=(j, k), rng=rng)
            for key in PAIR_KEYS:
                pair_dict[key].append(res[key])

    n = sum(len(x) for x in pair_dict["x0"])
    idx = np.arange(n)
    rng.shuffle(idx)
    return {key: np.concatenate(pair_dict[key], axis=0)[idx] for key in PAIR_KEYS}


def count_pairs_per_catalog(pair_dict, n_cat) -> np.ndarray:
    """Number of paired stars that come from each catalog."""
    return np.bincount(pair_dict["c0"], minlength=n_cat) + np.bincount(
        pair_dict["c1"], minlength=n_cat
    )


def dispersed_x_err(tables, keys: Optional[Sequence[str]] = None) -> List[np.ndarray]:
    """Regularization error of the dispersed points of each catalog, ``(4,)`` each.

    The dispersed points are the prior in the part of parameter space where a
    catalog has no data, so they use one value per catalog and parameter, not a
    function of position: the functional forms of the catalogs' ``x_err`` are
    only meaningful where they have data, and the real control points already
    carry them there. The value is the median ``x_err`` of the catalog's stars
    that measure the parameter, so it follows the catalog; a catalog module can
    override it with ``X_ERR_DISPERSED``. A parameter that no star of the
    catalog measures stays :data:`MISSING_XE`, which :func:`load_training_data`
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
    tables, pair_dict, rng, *, dispersed_xerr: str = "per-catalog", keys=None
) -> Dict[str, np.ndarray]:
    """Control points from each catalog, plus dispersed points covering the space.

    For each catalog the number of control points is the least of

      * (# of pairs) / (# of catalogs - 1), which makes the number of control
        points roughly equal to the number of pairs;
      * twice the number of paired stars from the catalog, so that a catalog
        is not represented much more than its paired stars are;
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
    n_cat = count_pairs_per_catalog(pair_dict, n_tab)
    ctrl: Dict[str, list] = {k: [] for k in CTRL_KEYS}

    n_max = pair_dict["x0"].shape[0] // (n_tab - 1)

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
    """Build ``(pair_dict, ctrl_dict)`` from the catalogs.

    The RNG is shared and stateful across the pair and control-point stages,
    so the order of operations matters for exact reproducibility.
    ``dispersed_xerr="uniform"`` reproduces the legacy dispersed-point errors.
    """
    keys = None
    if tables is None:
        keys = registry.keys()
        tables = load_tables(keys, dedupe=dedupe)
    rng = np.random.default_rng(seed)
    pair_dict = build_pairs(tables, rng)
    ctrl_dict = build_control_points(
        tables, pair_dict, rng, dispersed_xerr=dispersed_xerr, keys=keys
    )
    return pair_dict, ctrl_dict


def save_training_data(
    pair_dict, ctrl_dict, path: Optional[Path] = None, attrs: Optional[dict] = None
) -> Path:
    """Write the file; ``attrs`` (seed, dedupe, dispersed_xerr, ...) are stored as attributes."""
    path = Path(path or config.PAIRS_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    kw = dict(compression="gzip", compression_opts=3, chunks=True)
    with h5py.File(path, "w") as f:
        for key, value in (attrs or {}).items():
            f.attrs[key] = value
        for key in tqdm(pair_dict, desc="Saving pairs"):
            f.create_dataset(f"pair_dict/{key}", data=pair_dict[key], **kw)
        for key in tqdm(ctrl_dict, desc="Saving control points"):
            f.create_dataset(f"ctrl_dict/{key}", data=ctrl_dict[key], **kw)
    log.info("Wrote %s", path)
    return path


def load_training_data(
    path: Optional[Path] = None, *, fix_unknown_errors: bool = True
) -> Tuple[dict, dict]:
    """Load ``(pair_dict, ctrl_dict)``.

    With ``fix_unknown_errors`` (the training default), regularization errors
    ``> 1e6`` (placeholders for missing parameters) are replaced by 0.05.
    ``ctrl_dict["missing_control"]`` (boolean, ``(n, n_dim)``) always records
    which control-point parameters were placeholders.
    """
    path = Path(path or config.PAIRS_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist; build it with `python -m astrotune.training_data`."
        )
    with h5py.File(path, "r") as f:
        pair_dict = {k: f["pair_dict"][k][:] for k in f["pair_dict"].keys()}
        ctrl_dict = {k: f["ctrl_dict"][k][:] for k in f["ctrl_dict"].keys()}

    # Remember which parameters are not measured; the fix-up below hides it.
    ctrl_dict["missing_control"] = ctrl_dict["xe_control"] > 1e6

    if fix_unknown_errors:
        for key in ("xe0", "xe1"):
            pair_dict[key][pair_dict[key] > 1e6] = UNKNOWN_XE
        ctrl_dict["xe_control"][ctrl_dict["xe_control"] > 1e6] = UNKNOWN_XE
    return pair_dict, ctrl_dict


def read_attrs(path: Optional[Path] = None) -> dict:
    """Attributes stored with the file (empty for files made before they existed)."""
    with h5py.File(Path(path or config.PAIRS_PATH), "r") as f:
        return {k: (v.item() if hasattr(v, "item") else v) for k, v in f.attrs.items()}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="astrotune-pairs",
        description="Build the training pairs and control points.",
    )
    parser.add_argument("-o", "--output", type=Path, default=config.PAIRS_PATH)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--dedupe",
        choices=("none", "first"),
        default="none",
        help="'none' (default) reproduces the legacy many-to-many join on duplicate "
        "Gaia ids; 'first' keeps one row per id in every catalog",
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
    pair_dict, ctrl_dict = build_training_data(
        seed=args.seed, dedupe=args.dedupe, dispersed_xerr=args.dispersed_xerr
    )
    log.info(
        "%d pairs, %d control points",
        len(pair_dict["x0"]),
        len(ctrl_dict["x_control"]),
    )
    save_training_data(
        pair_dict, ctrl_dict, args.output,
        attrs=dict(
            seed=args.seed, dedupe=args.dedupe, dispersed_xerr=args.dispersed_xerr,
            catalogs=",".join(registry.keys()),
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
