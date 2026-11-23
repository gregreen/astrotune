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
    "ls": "ln of the quoted (raw) sigma (0 where not measured)",
    "sig": "the catalog's quoted sigma, in its own units (0 where not measured)",
    "sig0": "the prepared sigma_x (scaled and floored), for comparison",
    "J": "Jacobian dy/dx of the bijector at this entry (4,4)",
    "c": "catalog index",
    "g": "group index (catalogs that share a pipeline share a group)",
    "y": "harmonized parameters",
    "m": "parameter measured by this entry",
    "seg": "star index of the entry",
    "id": "Gaia DR3 source id, per star",
    "off": "entry offset of each star (``n_star + 1`` values)",
    "n_entry": "entries of each star",
}

ENTRY_FIELDS = ("x", "ls", "sig", "sig0", "J", "c", "y", "m", "seg", "g")
STAR_FIELDS = ("id", "off", "n_entry")


def val_mask(ids: np.ndarray, frac: float = VAL_FRAC) -> np.ndarray:
    """Validation stars: a hash of the id, so a star is always on the same side."""
    h = (np.asarray(ids).astype("u8") * np.uint64(0x9E3779B97F4A7C15)) >> np.uint64(40)
    return (h % np.uint64(1 << 16)) < np.uint64(frac * (1 << 16))


def raw_sigma(table, key: str) -> np.ndarray:
    """The catalog's *quoted* sigma, in our parameter units, ``(n, d)``.

    From the prepared table's own ``sigma_x_raw`` when it has one, else recovered from
    ``sigma_x`` with :func:`astrotune.prep.error_scale.quoted_sigma`, which inverts the
    catalog's rescale and the error floor exactly -- so tables written before that
    column existed still work. This is what the delta model is fitted on: it learns the
    rescale and the floor itself.
    """
    if "sigma_x_raw" in table.colnames:
        return np.asarray(table["sigma_x_raw"], dtype="f4")
    from ..prep.error_scale import quoted_sigma

    return quoted_sigma(key, np.asarray(table["sigma_x"])).astype("f4")


def _catalog_entries(table, key: str, fwd_jac, index: int):
    """First valid row per star, sorted by id: the arrays one catalog contributes.

    ``sig`` is the quoted sigma (:func:`raw_sigma`), ``sig0`` the prepared one the
    current pipeline propagates, and ``y``/``J`` the bijector's value and Jacobian at
    those rows -- both from one ``fwd_jac`` call, which is what lets the delta model
    work in the catalogs' units.

    Nothing here comes from ``astrotune-apply``: the star table is a function of the
    prepared table and the bijector alone. That is deliberate -- the delta stage is
    half of the latent-y/delta iteration, and routing it through a transform cache
    made it depend on a directory that a retrained bijector silently invalidates.
    """
    from ..apply import padded_inputs

    ids = np.asarray(table["id"])
    x, sigma, measured = padded_inputs(table)
    ok = np.flatnonzero(np.isfinite(x).all(1))
    _, first = np.unique(ids[ok], return_index=True)
    rows = ok[first]

    quoted = raw_sigma(table, key)
    sig = np.zeros_like(x)
    sig[:, : quoted.shape[1]] = quoted
    sig = np.where(measured, sig, 0.0)

    # Chunked: a catalog can contribute millions of rows and the Jacobian is dense.
    n_dim = x.shape[1]
    chunk = 65_536
    jac = np.empty((len(rows), n_dim, n_dim), dtype="f4")
    y = np.empty((len(rows), n_dim), dtype="f4")
    for lo in range(0, len(rows), chunk):
        sl = slice(lo, min(lo + chunk, len(rows)))
        y_hat, j = fwd_jac(x[rows[sl]], np.full(len(rows[sl]), index, dtype="i4"))
        jac[sl] = np.asarray(j, dtype="f4")
        y[sl] = np.asarray(y_hat, dtype="f4")
    m = measured[rows]
    # An unmeasured axis carries no value: the map said something there, but nothing
    # measured it, so it is the placeholder 0 -- the same encoding apply writes.
    return dict(
        id=ids[rows],
        x=x[rows],
        # a catalog may quote sigma = 0 on a parameter it does measure; ln(sigma) is
        # then -inf, which the model tolerates (it does not feed it to the heads) but
        # which has no meaning, so clip it.
        ls=np.log(np.where(measured, np.maximum(sig, 1e-30), 1.0)).astype("f4"),
        sig=sig[rows].astype("f4"),
        sig0=np.where(measured, sigma, 0.0)[rows].astype("f4"),
        J=jac,
        y=np.where(m, y, 0.0),
        m=m,
    )


def build_star_table(run_dir, cache_dir: Optional[Path] = None) -> Tuple[StarTable, Sequence[str]]:
    """Build the star table for a trained run; returns ``(stars, catalog_keys)``.

    ``cache_dir`` is accepted and ignored: the table no longer reads ``apply``'s
    transform cache (see :func:`_catalog_entries`), so the only state it needs is
    the prepared tables and the run's bijector.
    """
    from ..apply import make_forward_and_jacobian
    from ..model import load_bijector
    from ..model.io import enable_memory_growth
    from ..training_data import load_tables

    run_dir = Path(run_dir)
    keys = registry.run_keys(run_dir)
    tables = load_tables(keys, dedupe="none")

    enable_memory_growth()
    fwd_jac = make_forward_and_jacobian(load_bijector(run_dir))

    cats = []
    for key, t in zip(keys, tables):
        cats.append(_catalog_entries(t, key, fwd_jac, registry.category_index(key)))
        log.info("%s: %d stars (first row per id)", key, len(cats[-1]["id"]))
    del tables

    # Concatenate the catalogs, then sort by star so that a star's entries are contiguous.
    ids_raw = np.concatenate([c["id"] for c in cats])
    cat_raw = np.concatenate([np.full(len(c["id"]), j, dtype="i4") for j, c in enumerate(cats)])
    names = ("x", "ls", "sig", "sig0", "J", "y", "m")
    raw = {name: np.concatenate([c[name] for c in cats]) for name in names}
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
    for name in names:
        stars[name] = raw[name][final]
    log.info(
        "%d entries of %d stars (%d catalogs, at most %d entries per star)",
        n_ent, len(star_id), len(keys), int(count.max()) if len(count) else 0,
    )
    return stars, tuple(keys)


def as_prepared(stars: StarTable) -> StarTable:
    """A view of ``stars`` whose ``sig`` is the prepared ``sig0``.

    ``sig`` is the *quoted* sigma the delta model corrects; ``sig0`` is what the
    pipeline propagates today (quoted, scaled, floored). Swapping them gives the
    reference arm of every comparison: the covariance the merge uses without any
    delta model.
    """
    out = dict(stars)
    out["sig"] = stars["sig0"]
    return out


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
        if tuple(keys) != tuple(registry.run_keys(run_dir)):
            log.warning("%s was made for other catalogs; rebuilding", path)
        elif "J" not in stars or "sig" not in stars:
            # Written before the delta model moved to the catalogs' own units.
            log.warning("%s has no Jacobian; rebuilding", path)
        else:
            log.info("Loaded %s", path)
            return with_groups(stars, keys), keys
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
