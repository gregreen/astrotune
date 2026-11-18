"""Every observation of every star, grouped by star.

This is the data the post-hoc error calibration is fitted to. It replaces the
older pair table, which listed unordered *pairs* of catalog entries and so put
one likelihood term on each pair -- for a star in ``K`` catalogs that is
``K (K - 1) / 2`` terms built from ``K`` numbers, neither independent nor
correctly weighted. Here a star is one row of the likelihood and its entries are
summed over, which is the same frame of reference the bijection training uses
(the latent ``y`` of a star, with all of its observations).

Layout. Entries are stored in star order, so the entries of star ``s`` are the
slice ``off[s]:off[s + 1]``. That keeps every per-star operation a contiguous
reduction (:func:`numpy.add.reduceat`, ``unsorted_segment_sum``) and needs
neither a ragged tensor nor a loop over stars.

Choices, all deliberate:

* **One entry per star per catalog** (the first one in the catalog's table).
  Repeat spectra within a catalog would be correlated, and the calibration
  question is about agreement *between* catalogs. (The bijection training keeps
  them; see ``porting_project.md``.)
* **Only stars with two or more entries.** A star with one entry says nothing
  about whether the two errors are right: with the star's value free, its
  likelihood is flat in the added variance.
* Parameters that an entry does not measure are flagged in ``m`` and carry a
  placeholder variance of ``MISSING_VAR``, so they drop out of every inverse and
  determinant exactly (see :func:`astrotune.delta.model.star_nll`).
* The train/validation split is by star (a hash of the Gaia id), so all entries
  of a star fall on the same side.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Iterator, Optional, Sequence, Tuple

import numpy as np

from .. import registry

log = logging.getLogger(__name__)

N_PARAMS = 4

#: Variance placeholder for a parameter the catalog does not measure: 1/sigma^2
#: is then ~1e-9, i.e. no information, without putting a non-positive-definite
#: matrix in front of the Cholesky.
MISSING_VAR = 1e9

#: Fraction of stars (by id hash) used for validation
VAL_FRAC = 1 / 8

#: ``(entries, 4)`` arrays plus the per-star bookkeeping
StarTable = Dict[str, np.ndarray]

#: Array names of a star table and their meaning. The first group is indexed by
#: entry (``N`` rows, in star order), the second by star.
FIELDS = {
    "x": "the catalog's own parameters (0 where not measured)",
    "ls": "ln of the reported sigma_x (0 where not measured)",
    "c": "catalog index",
    "g": "group index (catalogs that share a pipeline share a group)",
    "y": "harmonized parameters",
    "C": "harmonized covariance (4,4)",
    "m": "parameter measured by this entry",
    "seg": "star index of the entry",
    "id": "Gaia DR3 source id, per star",
    "off": "entry offset of each star (``n_star + 1`` values)",
    "n_entry": "entries of each star",
}

ENTRY_FIELDS = ("x", "ls", "c", "y", "C", "m", "seg", "g")
STAR_FIELDS = ("id", "off", "n_entry")


def val_mask(ids: np.ndarray, frac: float = VAL_FRAC) -> np.ndarray:
    """Validation stars: a hash of the id, so a star is always on the same side."""
    h = (np.asarray(ids).astype("u8") * np.uint64(0x9E3779B97F4A7C15)) >> np.uint64(40)
    return (h % np.uint64(1 << 16)) < np.uint64(frac * (1 << 16))


def _catalog_entries(est, table):
    """First valid row per star, sorted by id: the arrays one catalog contributes."""
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


def build_star_table(run_dir, cache_dir: Optional[Path] = None) -> Tuple[StarTable, Sequence[str]]:
    """Build the star table for a trained run; returns ``(stars, catalog_keys)``."""
    from ..apply import load_cached
    from ..training_data import load_tables

    run_dir = Path(run_dir)
    keys = registry.run_keys(run_dir)
    # ``apply`` caches catalog ``i`` under its *registry* index, and only runs on runs
    # whose catalogs are a prefix of the registry (it checks). Reading the cache by
    # position instead would silently take another catalog's harmonized values for a
    # run whose catalog list is not that prefix, so insist on it here too.
    if tuple(registry.keys()[: len(keys)]) != tuple(keys):
        raise RuntimeError(
            "The registry order no longer matches the run's catalog list "
            f"({list(keys)}); the transformation cache is indexed by registry position."
        )
    cache_dir = Path(cache_dir) if cache_dir else run_dir / "harmonized" / "transform_cache_none"
    if not cache_dir.exists():
        raise FileNotFoundError(
            f"{cache_dir} does not exist. Run `astrotune-apply --run-dir {run_dir}` first "
            "(it caches the transformed estimates)."
        )
    tables = load_tables(keys, dedupe="none")

    cats = []
    for key, t in zip(keys, tables):
        est = load_cached(cache_dir, registry.category_index(key), np.asarray(t["id"]), t)
        if est is None:
            raise FileNotFoundError(
                f"No usable cached transformation for {key} in {cache_dir} (missing, or made from a "
                f"different x or sigma_x): run `astrotune-apply --run-dir {run_dir}` (again)."
            )
        cats.append(_catalog_entries(est, t))
        log.info("%s: %d stars (first row per id)", key, len(cats[-1]["id"]))
    del tables

    # Concatenate the catalogs, then sort by star so that a star's entries are contiguous.
    ids_raw = np.concatenate([c["id"] for c in cats])
    cat_raw = np.concatenate([np.full(len(c["id"]), j, dtype="i4") for j, c in enumerate(cats)])
    raw = {name: np.concatenate([c[name] for c in cats]) for name in ("x", "ls", "y", "C", "m")}
    del cats
    order = np.lexsort((cat_raw, ids_raw))  # by star, then by catalog index
    ids = ids_raw[order]

    # Stars observed by at least two catalogs; ``n_entry`` is the multiplicity.
    # Entries are in id order, so the entries of a star are contiguous and the kept
    # ones are ``np.repeat(keep, count)`` -- no loop over stars. ``final`` gathers
    # the kept entries straight from the concatenated arrays, once.
    star_all, _, count_all = np.unique(ids, return_index=True, return_counts=True)
    keep = count_all >= 2
    final = order[np.flatnonzero(np.repeat(keep, count_all))]
    star_id, count = star_all[keep], count_all[keep]
    off = np.r_[0, np.cumsum(count)].astype("i8")
    n_ent = int(off[-1])

    stars: StarTable = {
        "id": star_id.astype("i8"),
        "off": off,
        "n_entry": count.astype("i4"),
        "seg": np.repeat(np.arange(len(star_id), dtype="i4"), count),
        "c": cat_raw[final].astype("i4"),
    }
    for name in ("x", "ls", "y", "C", "m"):
        stars[name] = raw[name][final]
    log.info(
        "%d entries of %d stars (%d catalogs, at most %d entries per star)",
        n_ent, len(star_id), len(keys), int(count.max()) if len(count) else 0,
    )
    return stars, tuple(keys)


def subset(stars: StarTable, keep_stars: np.ndarray) -> StarTable:
    """The given stars, with their entries."""
    off = stars["off"]
    if keep_stars.dtype != bool:
        mask = np.zeros(len(stars["id"]), dtype=bool)
        mask[keep_stars] = True
    else:
        mask = keep_stars
    # ``np.repeat`` of the per-star flag is the vectorised form of the loop over
    # stars, and keeps the entries of a star together.
    entry = np.repeat(mask, np.diff(off))
    out: StarTable = {
        "id": stars["id"][mask],
        "n_entry": stars["n_entry"][mask],
    }
    out["off"] = np.r_[0, np.cumsum(out["n_entry"])].astype("i8")
    out["seg"] = np.repeat(np.arange(int(mask.sum()), dtype="i4"), out["n_entry"])
    for name in ENTRY_FIELDS:
        if name == "seg" or name not in stars:
            continue
        out[name] = stars[name][entry]
    return out


def split(stars: StarTable, frac: float = VAL_FRAC) -> Tuple[StarTable, StarTable]:
    """``(train, validation)`` split by star."""
    v = val_mask(stars["id"], frac)
    return subset(stars, ~v), subset(stars, v)


def block_indices(stars: StarTable, perm: Optional[np.ndarray] = None,
                  target_entries: int = 131072) -> Iterator[Tuple[np.ndarray, np.ndarray, int]]:
    """Yield ``(entry idx, star segment, n_stars)`` for blocks of whole stars.

    Every block holds all the entries of the stars it contains, which is what the
    likelihood needs: the star's common value is a reduction over exactly those
    rows. ``perm`` reorders the stars (one epoch), and the entry index is built
    with ``np.repeat`` rather than a loop over stars.
    """
    n_star = len(stars["id"])
    perm = np.arange(n_star) if perm is None else np.asarray(perm)
    off = stars["off"][:-1][perm]
    count = stars["n_entry"][perm]
    cum = np.cumsum(count)
    # Star-block starts: the position of every ``target_entries``-th entry.
    edges = np.searchsorted(cum, np.arange(0, int(cum[-1]), target_entries), side="right")
    edges = np.unique(np.r_[edges, len(perm)])
    for b0, b1 in zip(edges[:-1], edges[1:]):
        cnt = count[b0:b1]
        new_start = np.r_[0, np.cumsum(cnt)[:-1]]
        src = off[b0:b1]
        idx = np.repeat(src - new_start, cnt) + np.arange(int(cnt.sum()))
        seg = np.repeat(np.arange(b1 - b0, dtype="i4"), cnt)
        yield idx, seg, b1 - b0


def save_star_table(path: Path, stars: StarTable, keys: Sequence[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, _keys=json.dumps(list(keys)), **stars)
    tmp.replace(path)


def load_star_table(path: Path) -> Tuple[StarTable, Sequence[str]]:
    with np.load(path) as f:
        keys = tuple(json.loads(str(f["_keys"])))
        # ``g`` is derived on load (``with_groups``), so a cache written before it
        # existed is still valid.
        return {k: f[k] for k in FIELDS if k in f}, keys


def get_star_table(run_dir, out_dir: Optional[Path] = None, rebuild: bool = False):
    """Load ``<out_dir>/star_table.npz`` if it exists, else build and save it."""
    run_dir = Path(run_dir)
    out_dir = Path(out_dir) if out_dir else run_dir / "delta"
    path = out_dir / "star_table.npz"
    if path.exists() and not rebuild:
        stars, keys = load_star_table(path)
        if tuple(keys) == tuple(registry.run_keys(run_dir)):
            log.info("Loaded %s", path)
            return with_groups(stars, keys), keys
        log.warning("%s was made for other catalogs; rebuilding", path)
    stars, keys = build_star_table(run_dir)
    save_star_table(path, stars, keys)
    return with_groups(stars, keys), keys


def with_groups(stars: StarTable, keys: Sequence[str]) -> StarTable:
    """Add the group index ``g`` of every entry, derived from its catalog ``c``.

    Catalogs that run the same pipeline on the same spectra share one model
    category (see :data:`astrotune.registry.GROUPS`), so the delta model's
    parameters -- and its shared systematic -- are indexed by ``g``, not by ``c``.
    Deriving it on load keeps existing ``star_table.npz`` caches valid.
    """
    g_of_catalog = np.asarray(registry.group_index(keys), dtype="i4")
    out = dict(stars)
    out["g"] = g_of_catalog[np.asarray(stars["c"], dtype="i8")]
    return out
