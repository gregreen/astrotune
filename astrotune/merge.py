"""Combine the harmonized estimates of each star into one estimate per star.

Every catalog entry is a Gaussian in the harmonized space, ``N(y_i, C_i)``
(see :mod:`astrotune.apply` for how ``y_i`` and ``C_i`` are obtained). Entries
that refer to the same Gaia DR3 source are combined as a product of Gaussians,

    C = (sum_i (C_i + e^2 I)^-1)^-1,        y = C sum_i (C_i + e^2 I)^-1 y_i,

where ``e = eps_err`` is a small floor that keeps the inverses well behaved and
stops any one catalog from dominating. Entries that disagree with the combined
estimate (``chi2 > chi2_max``) are dropped and the combination is repeated; this
is iterated ``n_iter`` times. ``chi2`` is always measured for *all* entries
against the current combined estimate, so an entry that was dropped can return.
The entry (or entries) of a star with the lowest ``chi2`` is never dropped.

Parameters that a catalog does not measure are encoded as ``y = 0``,
``C_pp = 1e9`` and zero covariance with the other parameters (the notebook's
convention for [alpha/Fe]). They carry (almost) no weight in the sums, and a
star for which *no* contributing entry measures a parameter ends up with exactly
that encoding; ``measured`` records which parameters are real.

Differences from the reference notebook (all intended; see ``porting_project.md``):

* ``params_err`` is ``sqrt(C_ii + e^2)``. The notebook computed
  ``sqrt(C_ii^2 + e^2)``, squaring a variance.
* A star left with one entry after outlier rejection takes that entry. The
  notebook returned a zero estimate with zero covariance for it.
* ``lndet_cov`` is evaluated on the measured parameters only. With an unmeasured
  parameter the notebook's value contained ``ln(1e9)``.
* The best-matching entry of every star is never rejected. In the notebook a
  single wild entry could drag the first combined estimate so far that *every*
  entry of the star failed ``chi2``; the star then alternated between "no
  entries" (NaN) and "all entries" from one iteration to the next.
* ``chi2_max_used`` (largest ``chi2`` among the entries used) is returned, so
  stars whose entries disagree can be cut.
* ``params_source`` is ``uint32`` (the notebook's ``uint16`` overflows at 16
  catalogs).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
from tqdm.auto import tqdm

log = logging.getLogger(__name__)

#: Floor added in quadrature to every entry's uncertainty (notebook: 0.03)
EPS_ERR = 0.03

#: Entries with chi2 (4 parameters) above this are dropped (notebook: 25; raised
#: to 30 by the owner because typical chi2 values are ~15, so 25 cuts too sharply)
CHI2_MAX = 30.0

#: Number of reject-and-recombine iterations (notebook: 3)
N_ITER = 3

#: Variance that stands for "this parameter is not measured"
MISSING_VAR = 1e9

#: ``params_source`` is a bit mask with one bit per catalog
MAX_CATALOGS = 32

_CHUNK = 1_000_000


@dataclass
class Estimates:
    """The harmonized estimates of one catalog (one row per catalog entry)."""

    index: int  # catalog index = bit position in ``params_source``
    ids: np.ndarray  # (n,) Gaia DR3 source ids
    y: np.ndarray  # (n, d) harmonized parameters
    cov: np.ndarray  # (n, d, d) covariances in the harmonized space
    measured: np.ndarray  # (n, d) bool

    def __len__(self) -> int:
        return len(self.ids)


@dataclass
class Merged:
    """Result of :func:`combine`. One row per unique source, sorted by id."""

    ids: np.ndarray
    params: np.ndarray  # (n, d); NaN if no entry was used
    cov: np.ndarray  # (n, d, d)
    err: np.ndarray  # (n, d) sqrt(C_ii + eps^2)
    lndet_cov: np.ndarray  # (n,) over the measured parameters
    params_source: np.ndarray  # (n,) uint32 bit mask of the contributing catalogs
    n_used: np.ndarray  # (n,) number of entries combined
    chi2_max_used: np.ndarray  # (n,) largest chi2 of the entries used (vs the estimate)
    measured: np.ndarray  # (n, d) bool: measured by at least one entry used
    # Per catalog, per entry of that catalog's table (same order as the input)
    chi2: List[np.ndarray] = field(default_factory=list)
    used: List[np.ndarray] = field(default_factory=list)
    # Rejected fraction per catalog after each iteration: (n_iter, n_cat)
    rejected_history: List[Dict[int, Dict[str, float]]] = field(default_factory=list)


class _Entries:
    """All entries of all catalogs, sorted by source, with their inverse covariances."""

    def __init__(self, estimates: Sequence[Estimates], eps_err: float):
        if len(estimates) > MAX_CATALOGS:
            raise ValueError(f"At most {MAX_CATALOGS} catalogs fit in the uint32 bit mask.")
        d = estimates[0].y.shape[1]
        self.d = d
        self.eps_err = eps_err
        self.cat_index = [e.index for e in estimates]
        self.lengths = [len(e) for e in estimates]

        ids = np.concatenate([e.ids for e in estimates])
        self.ids_unique, inv = np.unique(ids, return_inverse=True)
        self.n_stars = len(self.ids_unique)

        # Entries sorted by star (stable, so each catalog keeps its own order)
        self.order = np.argsort(inv, kind="stable")
        star = inv[self.order]
        self.star = star
        self.starts = np.flatnonzero(np.r_[True, star[1:] != star[:-1]])
        assert len(self.starts) == self.n_stars

        cat_all = np.concatenate([np.full(len(e), e.index, dtype="i4") for e in estimates])
        self.cat = cat_all[self.order]
        self.bit = (np.uint32(1) << self.cat.astype("u4")).astype("u4")

        self.y = np.concatenate([e.y for e in estimates]).astype("f8")[self.order]
        self.measured = np.concatenate([e.measured for e in estimates])[self.order]
        cov = np.concatenate([e.cov for e in estimates])[self.order]
        self.cov = cov.astype("f4")

        # (C_i + e^2 I)^-1 and (C_i + e^2 I)^-1 y_i; float64 because the
        # unmeasured axes make the condition number ~ 1e12.
        eps_cov = eps_err**2 * np.eye(d)
        n = len(self.y)
        self.cinv = np.empty((n, d, d), dtype="f8")
        for i in tqdm(range(0, n, _CHUNK), desc="inverting covariances", leave=False):
            self.cinv[i : i + _CHUNK] = np.linalg.inv(cov[i : i + _CHUNK].astype("f8") + eps_cov)
        self.cinv_y = np.einsum("nij,nj->ni", self.cinv, self.y)

        self.counts = np.diff(np.r_[self.starts, n])

    # -- combination ---------------------------------------------------------
    def combine(self, used: np.ndarray):
        """Combine the entries flagged ``used``; returns arrays per star."""
        d, n_stars = self.d, self.n_stars
        w = used.astype("f8")
        n_used = np.add.reduceat(used.astype("i4"), self.starts)

        sum_cinv = np.add.reduceat(self.cinv.reshape(-1, d * d) * w[:, None], self.starts)
        sum_cinv_y = np.add.reduceat(self.cinv_y * w[:, None], self.starts)

        params = np.full((n_stars, d), np.nan)
        cov = np.full((n_stars, d, d), np.nan)

        multi = np.flatnonzero(n_used >= 2)
        for i in range(0, len(multi), _CHUNK):
            s = multi[i : i + _CHUNK]
            c = np.linalg.inv(sum_cinv[s].reshape(-1, d, d))
            cov[s] = c
            params[s] = np.einsum("nij,nj->ni", c, sum_cinv_y[s])

        # One entry: take it as it is (no floor in the covariance)
        pos = np.flatnonzero(used)
        entry_of_star = np.full(n_stars, -1, dtype="i8")
        entry_of_star[self.star[pos]] = pos
        one = np.flatnonzero(n_used == 1)
        e = entry_of_star[one]
        params[one] = self.y[e]
        cov[one] = self.cov[e]

        measured = np.logical_or.reduceat(used[:, None] & self.measured, self.starts, axis=0)
        source = np.bitwise_or.reduceat(np.where(used, self.bit, np.uint32(0)).astype("u4"), self.starts)
        return params, cov, n_used, measured, source

    def chi2(self, params: np.ndarray) -> np.ndarray:
        """chi2 of every entry against the combined estimate of its star."""
        dy = self.y - params[self.star]
        return np.einsum("ni,nij,nj->n", dy, self.cinv, dy, optimize=True)


def _enforce_missing(params, cov, measured):
    """Exact encoding for parameters that no used entry measures."""
    d = params.shape[1]
    for p in range(d):
        m = ~measured[:, p]
        params[m, p] = 0.0
        cov[m, p, :] = 0.0
        cov[m, :, p] = 0.0
        cov[m, p, p] = MISSING_VAR


def lndet_measured(cov: np.ndarray, measured: np.ndarray) -> np.ndarray:
    """``ln det C`` over the measured parameters only (NaN if ``C`` is NaN)."""
    c = cov.copy()
    d = c.shape[1]
    for p in range(d):
        m = ~measured[:, p]
        c[m, p, :] = 0.0
        c[m, :, p] = 0.0
        c[m, p, p] = 1.0
    _, lndet = np.linalg.slogdet(c)
    return lndet


def combine(
    estimates: Sequence[Estimates],
    eps_err: float = EPS_ERR,
    chi2_max: float = CHI2_MAX,
    n_iter: int = N_ITER,
) -> Merged:
    """Merge the per-catalog estimates into one estimate per Gaia source."""
    ent = _Entries(estimates, eps_err)
    log.info("%d entries of %d catalogs, %d unique sources", len(ent.y), len(estimates), ent.n_stars)

    used = np.ones(len(ent.y), dtype=bool)
    params, cov, n_used, measured, source = ent.combine(used)

    history = []
    multi_entry = ent.counts[ent.star] >= 2
    for it in range(n_iter):
        chi2 = ent.chi2(params)
        used = ~(chi2 > chi2_max)
        # ... but never drop the best-matching entry of a star
        used |= chi2 <= np.minimum.reduceat(chi2, ent.starts)[ent.star]
        params, cov, n_used, measured, source = ent.combine(used)

        stats = {}
        for k in sorted(set(ent.cat_index)):
            sel = ent.cat == k
            rej = ~used[sel]
            stats[k] = dict(
                rejected_frac=float(rej.mean()) if sel.any() else 0.0,
                rejected_frac_multi=float(rej[multi_entry[sel]].mean()) if (multi_entry & sel).any() else 0.0,
            )
        history.append(stats)
        log.info(
            "iteration %d: %.3f%% of entries rejected (%.3f%% of entries of multiply observed stars)",
            it + 1, 100 * (~used).mean(), 100 * (~used[multi_entry]).mean(),
        )

    _enforce_missing(params, cov, measured)
    err = np.sqrt(np.diagonal(cov, axis1=1, axis2=2) + eps_err**2)
    lndet = lndet_measured(cov, measured)

    # Return the per-entry results in the original, per-catalog order
    chi2 = ent.chi2(params)
    inv_order = np.empty_like(ent.order)
    inv_order[ent.order] = np.arange(len(ent.order))
    chi2_max_used = np.maximum.reduceat(np.where(used, chi2, 0.0), ent.starts)
    chi2_all, used_all = chi2[inv_order], used[inv_order]
    bounds = np.cumsum([0] + ent.lengths)
    return Merged(
        ids=ent.ids_unique,
        params=params,
        cov=cov,
        err=err,
        lndet_cov=lndet,
        params_source=source,
        n_used=n_used.astype("u1"),
        chi2_max_used=chi2_max_used,
        measured=measured,
        chi2=[chi2_all[a:b] for a, b in zip(bounds[:-1], bounds[1:])],
        used=[used_all[a:b] for a, b in zip(bounds[:-1], bounds[1:])],
        rejected_history=history,
    )


def catalogs_in(params_source: np.ndarray, index: int) -> np.ndarray:
    """Boolean mask: did catalog ``index`` contribute to the star's estimate?"""
    return (np.asarray(params_source) & np.uint32(1 << index)) != 0
