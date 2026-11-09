"""Pairs of harmonized estimates of the same star, from two different catalogs.

The table is built from the cache that ``astrotune-apply`` writes (the
harmonized values ``y`` and covariances ``C`` of every catalog entry) and the
prepared catalog tables (the catalog's own parameters ``x`` and reported
uncertainties ``sigma_x``, which are the inputs of the delta model).

Choices, all deliberate:

* **One row per star per catalog** (the first one in the catalog's table).
  Stars with several spectra in one catalog would otherwise create many-to-many
  pairs that over-weight them and are strongly correlated. (The bijector
  training keeps them; see ``porting_project.md``.)
* A pair is formed for every two catalogs that contain the star; side ``a`` is
  the catalog with the lower index.
* Parameters that either catalog does not measure are flagged in ``m``
  (``m[:, p]`` True = measured by both) and must be ignored.
* The train/validation split is by star (a hash of the Gaia id), so that all
  pairs of a star fall in the same part.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .. import registry

log = logging.getLogger(__name__)

N_PARAMS = 4

#: Fraction of stars (by id hash) used for validation
VAL_FRAC = 1 / 8

PairTable = Dict[str, np.ndarray]

#: Array names of a pair table and their meaning
FIELDS = {
    "id": "Gaia DR3 source id",
    "ca": "catalog index of side a",
    "cb": "catalog index of side b (> ca)",
    "xa": "side a: the catalog's own parameters (0 where not measured)",
    "xb": "side b: same",
    "lsa": "side a: ln of the reported sigma_x (0 where not measured)",
    "lsb": "side b: same",
    "ya": "side a: harmonized parameters",
    "yb": "side b: same",
    "Ca": "side a: harmonized covariance (4,4)",
    "Cb": "side b: same",
    "m": "parameter measured by both catalogs",
}


def val_mask(ids: np.ndarray, frac: float = VAL_FRAC) -> np.ndarray:
    """Validation stars: a hash of the id, so a star is always on the same side."""
    h = (np.asarray(ids).astype("u8") * np.uint64(0x9E3779B97F4A7C15)) >> np.uint64(40)
    return (h % np.uint64(1 << 16)) < np.uint64(frac * (1 << 16))


def _catalog_arrays(est, table):
    """First valid row per star, sorted by id: the arrays a pair needs."""
    from ..apply import padded_inputs

    ids = np.asarray(table["id"])
    x, sigma, measured = padded_inputs(table)
    ok = np.flatnonzero(np.isfinite(est.y).all(1) & np.isfinite(est.cov).all((1, 2)))
    _, first = np.unique(ids[ok], return_index=True)
    rows = ok[first]
    ln_s = np.log(np.where(measured, sigma, 1.0)).astype("f4")
    return dict(
        id=ids[rows],
        x=x[rows],
        ls=ln_s[rows],
        y=est.y[rows].astype("f4"),
        C=est.cov[rows].astype("f4"),
        m=est.measured[rows],
    )


def build_pair_table(run_dir, cache_dir: Optional[Path] = None) -> Tuple[PairTable, Sequence[str]]:
    """Build the pair table for a trained run; returns ``(pairs, catalog_keys)``."""
    from ..apply import load_cached
    from ..training_data import load_tables

    run_dir = Path(run_dir)
    keys = registry.run_keys(run_dir)
    cache_dir = Path(cache_dir) if cache_dir else run_dir / "harmonized" / "transform_cache_none"
    if not cache_dir.exists():
        raise FileNotFoundError(
            f"{cache_dir} does not exist. Run `astrotune-apply --run-dir {run_dir}` first "
            "(it caches the transformed estimates)."
        )
    tables = load_tables(keys, dedupe="none")

    cats = []
    for i, (key, t) in enumerate(zip(keys, tables)):
        est = load_cached(cache_dir, i, np.asarray(t["id"]), t)
        if est is None:
            raise FileNotFoundError(
                f"No usable cached transformation for {key} in {cache_dir} (missing, or made from a "
                f"different x or sigma_x): run `astrotune-apply --run-dir {run_dir}` (again)."
            )
        cats.append(_catalog_arrays(est, t))
        log.info("%s: %d stars (first row per id)", key, len(cats[-1]["id"]))
    del tables

    parts = {k: [] for k in FIELDS}
    for j in range(len(cats)):
        for k in range(j + 1, len(cats)):
            ia, ib = cats[j]["id"], cats[k]["id"]  # sorted and unique
            pos = np.searchsorted(ib, ia)
            pos[pos == len(ib)] = 0
            a = np.flatnonzero(ib[pos] == ia)
            b = pos[a]
            if len(a) == 0:
                continue
            m = cats[j]["m"][a] & cats[k]["m"][b]
            keep = m.any(1)
            a, b, m = a[keep], b[keep], m[keep]
            parts["id"].append(ia[a])
            parts["ca"].append(np.full(len(a), j, dtype="i4"))
            parts["cb"].append(np.full(len(a), k, dtype="i4"))
            for side, c, idx in (("a", cats[j], a), ("b", cats[k], b)):
                parts["x" + side].append(c["x"][idx])
                parts["ls" + side].append(c["ls"][idx])
                parts["y" + side].append(c["y"][idx])
                parts["C" + side].append(c["C"][idx])
            parts["m"].append(m)
    pairs = {k: np.concatenate(v) for k, v in parts.items()}
    log.info("%d pairs from %d catalogs", len(pairs["id"]), len(cats))
    return pairs, tuple(keys)


def counts_per_pair(pairs: PairTable, n_cat: int) -> np.ndarray:
    """``(n_cat, n_cat)`` number of pairs for each catalog pair (upper triangle)."""
    key = pairs["ca"].astype("i8") * n_cat + pairs["cb"]
    return np.bincount(key, minlength=n_cat * n_cat).reshape(n_cat, n_cat)


def subset(pairs: PairTable, sel: np.ndarray) -> PairTable:
    return {k: v[sel] for k, v in pairs.items()}


def split(pairs: PairTable, frac: float = VAL_FRAC) -> Tuple[PairTable, PairTable]:
    """``(train, validation)`` split by star."""
    v = val_mask(pairs["id"], frac)
    return subset(pairs, ~v), subset(pairs, v)


def save_pair_table(path: Path, pairs: PairTable, keys: Sequence[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, _keys=json.dumps(list(keys)), **pairs)
    tmp.replace(path)


def load_pair_table(path: Path) -> Tuple[PairTable, Sequence[str]]:
    with np.load(path) as f:
        keys = tuple(json.loads(str(f["_keys"])))
        return {k: f[k] for k in FIELDS}, keys


def get_pair_table(run_dir, out_dir: Optional[Path] = None, rebuild: bool = False):
    """Load ``<out_dir>/pair_table.npz`` if it exists, else build and save it."""
    run_dir = Path(run_dir)
    out_dir = Path(out_dir) if out_dir else run_dir / "delta"
    path = out_dir / "pair_table.npz"
    if path.exists() and not rebuild:
        pairs, keys = load_pair_table(path)
        if tuple(keys) == tuple(registry.run_keys(run_dir)):
            log.info("Loaded %s", path)
            return pairs, keys
        log.warning("%s was made for other catalogs; rebuilding", path)
    pairs, keys = build_pair_table(run_dir)
    save_pair_table(path, pairs, keys)
    return pairs, keys
