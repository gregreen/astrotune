"""How well calibrated are the quoted errors, in the harmonized space?

Every catalog entry of a star is a measurement of the same unknown value ``z``, so
the residual to look at is the entry's offset from that star's combined value,
measured in units of the variance that offset should have:

    d_k = y_k - z,     Cov(d_k) = S_k - A^-1,     A = sum_k S_k^-1,

where ``S_k = C_k`` (the propagated covariance), or
``S_k = diag(s_k) C_k diag(s_k) + diag(add_k)`` once a delta model exists. The
``-A^-1`` term is what the star framing adds: ``z``
is itself estimated from the entries, so an entry's residual is smaller than its
own covariance by the covariance of the combined value. This module measures how
far the entries of a :mod:`star table <astrotune.delta.stars>` are from that:

``scale_ratio``
    robust standard deviation of ``z_p = d_p / sqrt(Cov(d)_pp)``; 1 if calibrated,
    > 1 if the quoted errors are too small.
``PIT``
    the star-level ``chi2 = sum_k d_k^T S_k^-1 d_k`` on ``sum_k n_k - n_shared``
    degrees of freedom (``n_shared`` is the number of parameters that at least one
    entry measures); its chi-squared CDF value is uniform on [0, 1] if calibrated.
    One number per star, so the stars are independent, which the pair-based version
    could not offer: it had ``K (K-1) / 2`` correlated statistics per ``K - 1`` dof.
tails
    the fraction of stars with a chi-squared survival probability below the
    thresholds, and of ``|z| > 3`` (0.27 % if Gaussian).
``excess``
    per catalog and parameter, the variance the catalog needs on top of its quoted
    error. It is *negative* where the quoted errors are too large. Estimated from the
    stars in exactly two catalogs, where ``Var(y_1 - y_2) = S_1 + S_2`` holds
    exactly, so ``median(d^2) - median(S_1 + S_2)`` is an unbiased estimate of the
    sum of the two catalogs' excesses; solving those equations by weighted least
    squares is the classical three-cornered hat. Stars with more entries are left to
    the likelihood (:func:`three_cornered_hat` says why). The maximum-likelihood
    version of the same quantity is the ``constant per catalog`` model that
    ``astrotune-delta-train`` fits.
correlations
    of ``z`` among the four parameters (core of the distribution), against what the
    residual covariance itself predicts.
Teff dependence
    the scale ratio in bins of harmonized Teff.

Run with ``astrotune-delta-diagnose --run-dir models/X``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence

import numpy as np
from scipy import stats

if not os.environ.get("DISPLAY"):  # headless by default
    import matplotlib

    matplotlib.use("Agg")

import matplotlib.pyplot as plt

from .covariance import entry_cov_np
from .stars import N_PARAMS, StarTable, block_indices, get_star_table

log = logging.getLogger(__name__)

PARAM_NAMES = ("Teff", "[Fe/H]", "logg", "[a/Fe]")
SF_THRESHOLDS = (0.05, 1e-3, 1e-6)

#: ``delta_fn(x, ln_sigma, catalog) -> (ln_scale, ln_add)``, each ``(n, 4)``
DeltaFn = Callable[[np.ndarray, np.ndarray, np.ndarray], tuple]


def robust_scale(v: np.ndarray) -> float:
    """1.4826 x the median absolute deviation, ignoring NaNs (nan if too few values)."""
    v = v[np.isfinite(v)]
    if len(v) < 5:
        return np.nan
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


def _segments(seg: np.ndarray) -> np.ndarray:
    """Start offset of every run of equal ``seg`` (a block holds whole stars)."""
    return np.r_[0, np.flatnonzero(np.diff(seg)) + 1]


def _entry_cov(stars: StarTable, idx: np.ndarray,
               delta_fn: Optional[DeltaFn] = None,
               group_cov: Optional[np.ndarray] = None) -> np.ndarray:
    """``S = J diag(exp(ln_scale) sig^2 + exp(ln_add)) J^T`` for the entries ``idx``.

    With ``group_cov`` (one shared variance per group and parameter) the shared part
    is added as well, which makes this the covariance an entry has when compared with
    anything *outside* its group -- the one the merge uses. The shared part lives in
    the harmonized space, the scale and the additive term in the catalog's own.
    """
    ln_scale = ln_add = None
    if delta_fn is not None:
        ln_scale, ln_add = delta_fn(stars["x"][idx], stars["ls"][idx], stars["g"][idx])
    S = entry_cov_np(stars["sig"][idx], stars["J"][idx], ln_scale, ln_add,
                     measured=stars["m"][idx])
    if group_cov is not None:
        ii = np.arange(N_PARAMS)
        S[:, ii, ii] += np.asarray(group_cov, dtype="f8")[stars["g"][idx]]
    return S


def pair_chi2(stars: StarTable, delta_fn: Optional[DeltaFn] = None, chunk: int = 100_000,
              max_pairs: Optional[int] = None, seed: int = 0,
              nu: Optional[np.ndarray] = None,
              group_cov: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """``chi2`` of every pair of entries of the same star, one number per pair.

    For two entries of one star the difference ``y_a - y_b`` has covariance
    ``S_a + S_b`` *exactly* -- the two measurement errors are independent and the
    star's own value cancels, with no estimated quantity in between -- so on the
    parameters both entries measure, ``d^T (S_a + S_b)^-1 d`` is a chi-squared
    variable with ``nu`` degrees of freedom when the covariances are right. That
    makes this a direct test of the covariance model, independent of how the star's
    value was combined.

    ``nu`` (one tail weight per group) is recorded per pair as ``tail``: under a
    Student's t the same quadratic form is ``d * F(d, tail)``, so the reference curve
    changes shape even though the statistic does not. ``group_cov`` adds the shared
    part to the difference for pairs from *different* groups, where it does not
    cancel; pairs inside one group are unaffected by it, which is a useful check.

    Unlike the star-level chi2 this is *not* a set of independent statistics: the
    ``K (K - 1) / 2`` pairs of a star share entries. It is a distributional check
    (does the histogram match the ideal curve), not a test with a p-value.
    """
    ii = np.arange(N_PARAMS)
    tail = _star_nu(stars, nu)
    chi2, nu_dim, nu_val = [], [], []
    for idx, seg, n in block_indices(stars, None, chunk):
        S = _entry_cov(stars, idx, delta_fn)  # independent part only
        y, m = stars["y"][idx].astype("f8"), stars["m"][idx]
        g_idx = stars["g"][idx]
        shared = np.zeros((len(idx), N_PARAMS)) if group_cov is None else \
            np.asarray(group_cov, dtype="f8")[g_idx]
        counts = np.bincount(seg, minlength=n)
        starts = np.r_[0, np.cumsum(counts)[:-1]]
        for k in np.unique(counts):
            if k < 2:
                continue
            st = starts[counts == k]
            iu, ju = np.triu_indices(int(k), 1)
            a = (st[:, None] + iu[None, :]).ravel()
            b = (st[:, None] + ju[None, :]).ravel()
            both = m[a] & m[b]
            key = both.dot(1 << ii)
            star_of_pair = tail[stars["seg"][idx][a]] if nu is not None else None
            for kk in np.unique(key):
                sel = bool_key(kk)
                dof = int(sel.sum())
                if dof == 0:
                    continue
                rows = key == kk
                M = (S[a[rows]] + S[b[rows]])[:, sel][:, :, sel]
                # the shared part cancels between two entries of the same group
                same = g_idx[a[rows]] == g_idx[b[rows]]
                if not same.all():
                    extra = np.where(same[:, None], 0.0, shared[a[rows]] + shared[b[rows]])
                    dg = np.arange(M.shape[1])
                    M[:, dg, dg] += extra[:, sel]
                d = (y[a[rows]] - y[b[rows]])[:, sel]
                sol = np.linalg.solve(M, d[:, :, None])[:, :, 0]
                chi2.append(np.einsum("ni,ni->n", d, sol))
                nu_dim.append(np.full(int(rows.sum()), dof, dtype="i4"))
                nu_val.append(np.full(int(rows.sum()), np.inf) if star_of_pair is None
                              else star_of_pair[rows])
        if max_pairs is not None and sum(len(v) for v in chi2) > 4 * max_pairs:
            break  # a subsample is enough for a distribution; see the docstring
    out = dict(chi2=np.concatenate(chi2) if chi2 else np.zeros(0),
               nu=np.concatenate(nu_dim) if nu_dim else np.zeros(0, dtype="i4"),
               tail=np.concatenate(nu_val) if nu_val else np.zeros(0))
    if nu is None:
        del out["tail"]
    if max_pairs is not None and len(out["chi2"]) > max_pairs:
        keep = np.random.default_rng(seed).choice(len(out["chi2"]), max_pairs, replace=False)
        keep.sort()
        out = {k: v[keep] for k, v in out.items()}
    return out


def bool_key(key: int) -> np.ndarray:
    """The 4-bool mask encoded in an integer key (inverse of ``mask.dot(1 << i)``)."""
    return np.array([bool(key >> i & 1) for i in range(N_PARAMS)])


def chi2_mixture_pdf(x: np.ndarray, nu: np.ndarray,
                     tail: Optional[np.ndarray] = None) -> np.ndarray:
    """Density of the ideal reference, mixed over the observed masks (and tail).

    A pair that shares 4 parameters and a pair that shares 3 are both chi-squared
    variables, so the ideal curve of the pooled histogram is the mixture weighted by
    how many pairs have each ``nu`` -- which is what the catalogs' masks dictate, not
    a free parameter. With a Student's t of ``tail`` degrees of freedom the quadratic
    form is ``d * F(d, tail)`` instead, so the reference becomes the same mixture of
    F distributions (scaled back to the quadratic-form units the histogram uses).
    """
    nu = np.asarray(nu)
    if tail is None:
        nus, counts = np.unique(nu, return_counts=True)
        w = counts / max(counts.sum(), 1)
        return sum(float(wi) * stats.chi2.pdf(x, int(n)) for n, wi in zip(nus, w))
    tail = np.asarray(tail, dtype="f8")
    # bucket the continuous dof so that the mixture has a manageable number of terms
    keys, counts = np.unique(np.stack([nu, np.round(tail * 2.0) / 2.0], axis=1),
                             axis=0, return_counts=True)
    w = counts / counts.sum()
    out = np.zeros(np.shape(x), dtype="f8")
    for (d, t), wi in zip(keys, w):
        out += float(wi) * stats.f.pdf(np.asarray(x) / d, int(d), max(t, 2.01)) / d
    return out


def pair_chi2_text(before: Dict[str, np.ndarray], after: Dict[str, np.ndarray]) -> str:
    """The pair chi2 distribution: median, core and tail, quoted errors vs the model."""
    def line(name, d):
        r = d["chi2"] / np.maximum(d["nu"], 1)
        sf = _null_sf(d["chi2"], d["nu"], d.get("tail", np.full(len(r), np.inf)))
        return (f"{name:22s} median chi2/nu {np.median(r):5.2f}   "
                f"P(chi2 < 1) {np.mean(d['chi2'] < 1):6.2%}   "
                f"P(null sf < 1e-3) {np.mean(sf < 1e-3):7.3%}   "
                f"({len(d['chi2']):,} pairs)")

    return "\n".join([line("quoted errors", before), line("with the delta model", after)])


def plot_pair_chi2(before: Dict[str, np.ndarray], after: Dict[str, np.ndarray],
                   path: Path, title: str = "") -> Path:
    """Histogram of the pair chi2 before and after the model, against the ideal curve."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    hi = float(np.percentile(np.concatenate([before["chi2"], after["chi2"]]), 99.5))
    hi = max(hi, 10.0)
    bins = np.linspace(0.0, hi, 120)
    fig, ax = plt.subplots(figsize=(9, 6), layout="constrained")
    for d, color, label in ((before, "C0", "quoted errors"),
                            (after, "C3", "with the learned correction")):
        ax.hist(d["chi2"], bins=bins, density=True, histtype="step", lw=1.4,
                color=color, label=f"{label}: median $\\chi^2/\\nu$ = "
                                   f"{np.median(d['chi2'] / np.maximum(d['nu'], 1)):.2f}")
    x = np.linspace(0.0, hi, 800)
    tail = after.get("tail")
    ax.plot(x, chi2_mixture_pdf(x, after["nu"], tail), "k--", lw=1.2,
            label="ideal (mixture over $\\nu$ and tail weight)" if tail is not None
            else "ideal (mixture over $\\nu$)")
    ax.set_yscale("log")
    ax.set_ylim(1e-6, None)
    ax.set(xlabel=r"$\chi^2$ of the difference of two observations of one star",
           ylabel="density per bin (log scale)")
    ax.set_title(title or "pair chi2, before and after the correction", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def entry_stats(stars: StarTable, delta_fn: Optional[DeltaFn] = None,
                chunk: int = 200_000, nu: Optional[np.ndarray] = None,
                group_cov: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """Residuals of every entry against its star's combined value.

    Returns, per entry, ``d``, ``sd`` (the residual standard deviation), ``z`` (NaN
    where a parameter is not measured), ``chi2``, ``dof``, ``c`` and ``x``; per
    star, ``star_chi2``, ``star_dof``, ``star_nu`` and ``star_n``; the survival
    function and PIT of the star chi2 under the model's null; and the summed
    predicted correlation matrix (``corr_pred``) of the residual covariance.

    With ``nu`` (one degrees of freedom per catalog) the null of the star chi2 is
    ``dof * F(dof, nu)`` rather than ``chi2_dof``: under a Student's t the quadratic
    form carries the random scale, so the exact reference is an F distribution.
    """
    n_ent = int(stars["off"][-1])
    n_star = len(stars["id"])
    ii = np.arange(N_PARAMS)
    d = np.full((n_ent, N_PARAMS), np.nan)
    sd = np.full((n_ent, N_PARAMS), np.nan)
    chi2 = np.zeros(n_ent)
    dof = np.zeros(n_ent, dtype="i4")
    star_chi2 = np.zeros(n_star)
    star_dof = np.zeros(n_star, dtype="i4")
    corr_sum, corr_n = np.zeros((N_PARAMS, N_PARAMS)), 0

    for idx, seg, n in block_indices(stars, None, chunk):
        m = stars["m"][idx]
        mf = m.astype("f8")
        S = _entry_cov(stars, idx, delta_fn, group_cov)
        Sinv = np.linalg.inv(S)
        starts = _segments(seg)

        # Per-star precision sum A, precision-weighted mean y and its covariance.
        A = np.add.reduceat(Sinv.reshape(len(idx), N_PARAMS * N_PARAMS), starts,
                            axis=0).reshape(n, N_PARAMS, N_PARAMS)
        b = np.add.reduceat(np.einsum("nij,nj->ni", Sinv, stars["y"][idx].astype("f8")), starts, axis=0)
        Ainv = np.linalg.inv(A)
        z = np.einsum("nij,nj->ni", Ainv, b)

        dd = stars["y"][idx].astype("f8") - z[seg]
        resid_var = np.diagonal(S, axis1=1, axis2=2) - np.diagonal(Ainv, axis1=1, axis2=2)[seg]
        quad = np.einsum("ni,ni->n", dd, np.einsum("nij,nj->ni", Sinv, dd))

        d[idx] = np.where(m, dd, np.nan)
        sd[idx] = np.sqrt(np.where(m, resid_var, np.nan))
        chi2[idx] = quad
        dof[idx] = m.sum(1)
        # Per-star results go to the table's own star index, not the block's: every
        # block numbers its stars from 0.
        gseg = stars["seg"][idx]
        star_idx = gseg[starts]
        star_chi2[star_idx] = np.add.reduceat(quad, starts)
        shared = np.add.reduceat(mf, starts, axis=0) > 0  # parameter some entry measures
        star_dof[star_idx] = np.add.reduceat(m.sum(1), starts) - shared.sum(1)

        full = m.all(1) & np.isfinite(resid_var).all(1)
        if full.any():
            resid_cov = S[full] - Ainv[gseg[full] - gseg[0]]
            s = np.sqrt(np.maximum(np.diagonal(resid_cov, axis1=1, axis2=2), 1e-300))
            corr_sum += (resid_cov / s[:, :, None] / s[:, None, :]).sum(0)
            corr_n += int(full.sum())

    star_nu = _star_nu(stars, nu)
    return dict(
        d=d, sd=sd, z=d / sd, chi2=chi2, dof=dof,
        sf=_null_sf(star_chi2, star_dof, star_nu),
        pit=_null_sf(star_chi2, star_dof, star_nu, cdf=True),
        c=stars["c"], x=stars["x"], m=stars["m"],
        star_chi2=star_chi2, star_dof=star_dof, star_nu=star_nu,
        star_n=stars["n_entry"].astype("i4"),
        corr_pred=corr_sum / max(corr_n, 1),
    )


def _star_nu(stars: StarTable, nu: Optional[np.ndarray]) -> np.ndarray:
    """The star's tail weight: the mean over its entries of its groups' nu.

    A star's entries share one scale variable in the likelihood (that is what makes
    the t marginal analytic), so when its groups disagree the star gets their mean
    -- the only choice that reduces to the global-nu model when they agree.
    """
    n_star = len(stars["id"])
    if nu is None:
        return np.full(n_star, np.inf)
    nu_e = np.asarray(nu, dtype="f8")[stars["g"]]
    off = stars["off"][:-1]
    out = np.add.reduceat(nu_e, off) / stars["n_entry"] if n_star else np.zeros(0)
    return out


def _null_sf(star_chi2: np.ndarray, star_dof: np.ndarray, star_nu: np.ndarray,
             cdf: bool = False) -> np.ndarray:
    """Survival function (or CDF) of the star chi2 under the model's own null.

    Gaussian: ``star_chi2 ~ chi2_dof``. Student's t: ``star_chi2 / dof ~ F(dof, nu)``,
    which is the same statistic divided by its dimension -- hence ``q`` in one branch
    and the raw chi2 in the other.
    """
    q = np.maximum(star_chi2, 0.0) / np.maximum(star_dof, 1)
    out = np.ones_like(q)
    finite = np.isfinite(star_nu)
    if (~finite).any():
        args = (np.maximum(star_chi2[~finite], 0.0), np.maximum(star_dof[~finite], 1))
        out[~finite] = stats.chi2.cdf(*args) if cdf else stats.chi2.sf(*args)
    if finite.any():
        args = (q[finite], np.maximum(star_dof[finite], 1), star_nu[finite])
        out[finite] = stats.f.cdf(*args) if cdf else stats.f.sf(*args)
    return out


def three_cornered_hat(stars: StarTable, n_cat: int, min_pairs: int = 200) -> np.ndarray:
    """Per-catalog excess variance from the stars in exactly two catalogs.

    For such a star ``Var(y_1 - y_2) = S_1 + S_2`` exactly -- the difference is a
    difference of two independent measurements, with no estimated common value in
    it -- so ``median(d^2) - median(S_1 + S_2)`` estimates ``D_1 + D_2`` per
    parameter, and the ``D`` follow by weighted least squares over the catalog
    pairs. Catalogs not in any such star are NaN, and values may be negative.

    Stars with ``K >= 3`` entries are deliberately *not* used: their entry residual
    carries ``-A^-1``, the covariance of the star's combined value, and subtracting a
    median of that makes the naive pair formula biased low (with three equally
    weighted entries and equal quoted errors it recovers a third of the true extra
    variance). Those stars are what the likelihood is for.
    """
    out = np.full((n_cat, N_PARAMS), np.nan)
    # What the current pipeline propagates for every entry (the comparison baseline).
    quoted_cov = entry_cov_np(stars["sig0"], stars["J"], measured=stars["m"])
    off = stars["off"][:-1][stars["n_entry"] == 2]
    if len(off) == 0:
        return out
    c1 = stars["c"][off].astype("i8")
    key = c1 * n_cat + stars["c"][off + 1]
    order = np.argsort(key, kind="stable")
    ukey, start = np.unique(key[order], return_index=True)
    stop = np.r_[start[1:], len(order)]
    for p in range(N_PARAMS):
        rows = []
        for k, a, b in zip(ukey, start, stop):
            idx = off[order[a:b]]
            ok = stars["m"][idx, p] & stars["m"][idx + 1, p]
            if ok.sum() < min_pairs:
                continue
            i = idx[ok]
            d = (stars["y"][i, p].astype("f8") - stars["y"][i + 1, p].astype("f8"))
            Spp = (quoted_cov[i, p, p] + quoted_cov[i + 1, p, p])
            rows.append((int(k // n_cat), int(k % n_cat),
                         robust_scale(d) ** 2 - float(np.median(Spp)), int(ok.sum())))
        if not rows:
            continue
        A = np.zeros((len(rows), n_cat))
        y = np.zeros(len(rows))
        w = np.zeros(len(rows))
        for i, (j, k, e, n) in enumerate(rows):
            A[i, j] = A[i, k] = 1.0
            y[i], w[i] = e, np.sqrt(n)
        used = A.sum(0) > 0
        sol, *_ = np.linalg.lstsq(A[:, used] * w[:, None], y * w, rcond=None)
        out[used, p] = sol
    return out


def single_partner_share(stars: StarTable, n_cat: int) -> np.ndarray:
    """Share of each catalog's entries whose star is seen by only one other catalog.

    Such stars pin the *combination* of the two catalogs' error models, not each
    separately, so a catalog with a high share is weakly identified on its own:
    ``sdssv_mdc`` on the 20-catalog run has 98 % (almost all of it paired with
    ``sdssv_apogeenet``), which is why its fitted correction should be read as a
    difference from that catalog and not as an absolute statement.
    """
    n_entry = stars["n_entry"]
    star = np.repeat(np.arange(len(n_entry)), n_entry)
    only_two = (n_entry == 2)[star]
    c = stars["c"]
    out = np.full(n_cat, np.nan)
    for k in range(n_cat):
        sel = c == k
        if sel.any():
            out[k] = float(only_two[sel].mean())
    return out


def summarize(stars: StarTable, st: Dict[str, np.ndarray], n_cat: int,
              min_entries: int = 1000) -> dict:
    """All the calibration statistics in one dictionary (JSON-friendly after ``to_json``)."""
    z, d, sd = st["z"], st["d"], st["sd"]
    c, m = st["c"], st["m"]

    # Per catalog and parameter
    excess = three_cornered_hat(stars, n_cat, min_pairs=max(min_entries // 4, 20))
    alone = single_partner_share(stars, n_cat)
    per_catalog = []
    for k in range(n_cat):
        in_cat = c == k
        row = dict(catalog=k, n=int(in_cat.sum()), scale_ratio=np.full(N_PARAMS, np.nan),
                   beyond3=np.full(N_PARAMS, np.nan), excess=excess[k],
                   single_partner=float(alone[k]))
        for p in range(N_PARAMS):
            sel = in_cat & m[:, p]
            if sel.sum() < min_entries:
                continue
            row["scale_ratio"][p] = robust_scale(z[sel, p])
            row["beyond3"][p] = float(np.mean(np.abs(z[sel, p]) > 3))
        per_catalog.append(row)

    # Pooled, over stars (independent) and over entries (for the per-parameter widths)
    ok = st["star_dof"] > 0
    pit = st["pit"][ok]
    core = np.all(np.abs(z) < 3, axis=1) & m.all(1)
    corr_obs = np.corrcoef(z[core].T) if core.sum() > 100 else np.full((N_PARAMS,) * 2, np.nan)

    teff = st["x"][:, 0]
    edges = np.quantile(teff, np.linspace(0, 1, 9))
    which = np.clip(np.digitize(teff, edges[1:-1]), 0, 7)
    pooled = dict(
        n_stars=int(ok.sum()),
        n_entries=int(len(z)),
        median_pit=float(np.median(pit)),
        frac_sf=[float(np.mean(st["sf"][ok] < t)) for t in SF_THRESHOLDS],
        beyond3=[float(np.nanmean(np.abs(z[m[:, p], p]) > 3)) for p in range(N_PARAMS)],
        scale_ratio=[robust_scale(z[m[:, p], p]) for p in range(N_PARAMS)],
        pit_hist=np.histogram(pit, bins=20, range=(0, 1))[0],
    )
    return dict(
        per_catalog=per_catalog, pooled=pooled,
        excess_variance=np.stack([r["excess"] for r in per_catalog], axis=0)
        if per_catalog else np.full((n_cat, N_PARAMS), np.nan),
        corr_observed=corr_obs, corr_predicted=st["corr_pred"],
        teff_bins=dict(
            edges=edges,
            center=np.array([np.median(teff[which == i]) for i in range(8)]),
            scale_ratio=np.array([[robust_scale(z[(which == i) & m[:, p], p]) for p in range(N_PARAMS)]
                                  for i in range(8)]),
        ),
        min_entries=min_entries,
    )


def to_json(obj):
    """Convert a result dictionary to something ``json.dump`` accepts."""
    if isinstance(obj, dict):
        return {k: to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_json(obj.tolist())
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def diagnose(stars: StarTable, n_cat: int, delta_fn: Optional[DeltaFn] = None,
             min_entries: int = 1000, nu: Optional[np.ndarray] = None,
             group_cov: Optional[np.ndarray] = None, group_map=None) -> dict:
    """Every calibration statistic, with the null of the model that produced ``nu``.

    ``group_cov`` (shared variance per group) is added to each entry's covariance so
    that the residuals describe what the merge will use. One caveat: the shared part
    cancels from a star's *own* combined value when all its entries are in one group,
    so for the members of a group the predicted residual is somewhat too wide and
    their per-catalog scale ratio comes out below 1 (APOGEE [a/Fe] reads 0.45 with a
    shared sigma of 0.014 dex). The pooled numbers are fine -- they are dominated by
    stars that span several groups -- and ``pair_chi2`` handles the distinction
    exactly (it adds the shared part only across groups).
    """
    return summarize(stars, entry_stats(stars, delta_fn, nu=nu, group_cov=group_cov),
                     n_cat, min_entries)


def format_summary(res: dict, labels: Sequence[str], n_worst: int = 12) -> str:
    p = res["pooled"]
    lines = [
        f"{p['n_stars']:,} stars, {p['n_entries']:,} entries. Pooled: median PIT "
        f"{p['median_pit']:.2f} (0.50 if calibrated); "
        + ", ".join(f"P(sf<{t:g}) = {f:.2%}" for t, f in zip(SF_THRESHOLDS, p["frac_sf"]))
        + f"  (nominal {', '.join(f'{t:.2%}' for t in SF_THRESHOLDS)})",
        "scale ratio (1 = calibrated):   " + "  ".join(f"{n} {r:.2f}" for n, r in zip(PARAM_NAMES, p["scale_ratio"])),
        "|z| > 3 (0.27% if Gaussian):   " + "  ".join(f"{n} {r:.2%}" for n, r in zip(PARAM_NAMES, p["beyond3"])),
        "",
        f"{'catalog':26s}" + "".join(f"{n:>9s}" for n in PARAM_NAMES)
        + "".join(f"{'ex ' + n:>12s}" for n in PARAM_NAMES) + f"{'1-partner':>11s}",
    ]
    for row in res["per_catalog"]:
        lines.append(
            f"{labels[row['catalog']][:26]:26s}"
            + "".join(f"{r:9.2f}" for r in row["scale_ratio"])
            + "".join(f"{np.sqrt(max(e, 0.0)):12.3f}" for e in row["excess"])
            + f"{row.get('single_partner', float('nan')):11.0%}"
        )
    worst = []
    for row in res["per_catalog"]:
        for q in range(N_PARAMS):
            r = row["scale_ratio"][q]
            if np.isfinite(r):
                worst.append((abs(np.log(r)), labels[row["catalog"]], PARAM_NAMES[q], r))
    worst.sort(reverse=True)
    lines.append("")
    lines.append(f"farthest from 1 ({n_worst} of {len(worst)}):")
    for _, lab, name, r in worst[:n_worst]:
        lines.append(f"  {lab[:26]:26s} {name:7s} {r:.2f}")
    return "\n".join(lines)


def _heatmap(ax, matrix, labels, title, fmt="{:.2f}"):
    """A catalog x parameter matrix as an image with the numbers written in."""
    v = np.asarray(matrix, dtype="f8")
    logv = np.log2(np.clip(v, 1e-3, 1e3))
    im = ax.imshow(logv, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
    ax.set_xticks(range(N_PARAMS), PARAM_NAMES, fontsize=8)
    ax.set_yticks(range(len(labels)), [lab[:20] for lab in labels], fontsize=6)
    ax.set_title(title, fontsize=9)
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            if np.isfinite(v[i, j]):
                ax.text(j, i, fmt.format(v[i, j]), ha="center", va="center", fontsize=6)
    return im


def plot_diagnostics(res: dict, labels: Sequence[str], out_dir: Path, suffix: str = "") -> Sequence[Path]:
    """Three figures: per catalog/parameter, distributions, and Teff dependence."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    ratios = np.array([r["scale_ratio"] for r in res["per_catalog"]])
    excess = np.array([np.sqrt(np.maximum(r["excess"], 0.0)) for r in res["per_catalog"]])
    p = res["pooled"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 0.22 * len(labels) + 2.4), layout="constrained")
    im = _heatmap(axes[0], ratios, labels, "scale ratio of the residual (1 = calibrated)")
    fig.colorbar(im, ax=axes[0], label=r"$\log_2$ (1/4 ... 4)")
    im = _heatmap(axes[1], excess, labels, r"excess $\sigma$ needed (harmonized units)")
    fig.colorbar(im, ax=axes[1], label=r"$\log_2$")
    axes[2].bar(np.arange(20) / 20 + 0.025, p["pit_hist"], width=0.05, color="C0")
    axes[2].axhline(p["n_stars"] * 0.05, color="C3", lw=1, label="uniform")
    axes[2].set(xlabel="PIT of the star chi2", ylabel="stars", title="star-level coverage")
    axes[2].legend(fontsize=7)
    fig.suptitle(f"calibration of the quoted errors{suffix}", fontsize=10)
    paths.append(out_dir / f"summary{suffix}.png")
    fig.savefig(paths[-1], dpi=110)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    st_teff = res["teff_bins"]
    for q, ax in enumerate(axes.flat):
        ax.plot(st_teff["center"], st_teff["scale_ratio"][:, q], "o-", color="C0")
        ax.axhline(1.0, color="C3", lw=1)
        ax.set(title=PARAM_NAMES[q], xlabel=r"harmonized $5\ln(T_{\rm eff}/5000\,{\rm K})$",
               ylabel="scale ratio")
        ax.invert_xaxis()
        ax.grid(alpha=0.2)
    fig.suptitle(f"scale ratio against Teff{suffix}", fontsize=10)
    paths.append(out_dir / f"teff{suffix}.png")
    fig.savefig(paths[-1], dpi=110)
    plt.close(fig)
    return paths


def write_diagnostics(res: dict, labels: Sequence[str], out_dir: Path, suffix: str = "") -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"summary{suffix}.json", "w") as f:
        json.dump(to_json(dict(res, labels=list(labels))), f, indent=1)
    (out_dir / f"summary{suffix}.txt").write_text(format_summary(res, labels) + "\n")
    plot_diagnostics(res, labels, out_dir, suffix)


def main(argv: Optional[Sequence[str]] = None) -> int:
    from .. import registry

    parser = argparse.ArgumentParser(
        prog="astrotune-delta-diagnose",
        description="Calibration of the quoted errors, from every observation of every star.",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None, help="default: <run-dir>/delta")
    parser.add_argument("--min-entries", type=int, default=1000)
    parser.add_argument("--rebuild", action="store_true", help="rebuild the star table")
    parser.add_argument("--model", action="store_true",
                        help="also compare the pair chi2 with and without the saved delta model")
    parser.add_argument("--max-pairs", type=int, default=None,
                        help="subsample this many pairs for the chi2 distribution")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    out_dir = args.out_dir or args.run_dir / "delta"
    stars, keys = get_star_table(args.run_dir, out_dir, args.rebuild)
    labels = [registry.get_spec(k).label for k in keys]
    res = diagnose(stars, len(keys), None, args.min_entries)
    write_diagnostics(res, labels, out_dir / "diagnostics")
    print(format_summary(res, labels))
    if args.model:
        from .model import load_delta_model, model_nu, model_shared, numpy_delta_fn

        model = load_delta_model(out_dir, keys=keys)
        nu = model_nu(model)
        shared = model_shared(model)
        if nu is not None:
            log.info("tail weight per group: nu = %s", np.round(nu, 1).tolist())
        if shared is not None:
            log.info("shared sigma per group (max over parameters): %s",
                     np.round(np.sqrt(np.maximum(shared.max(1), 0.0)), 4).tolist())
        log.info("pair chi2: quoted errors")
        before = pair_chi2(stars, None, max_pairs=args.max_pairs)
        log.info("pair chi2: with the saved delta model")
        after = pair_chi2(stars, numpy_delta_fn(model), max_pairs=args.max_pairs, nu=nu,
                          group_cov=shared)
        text = pair_chi2_text(before, after)
        print(text)
        (out_dir / "diagnostics" / "pair_chi2.txt").write_text(text + "\n")
        plot_pair_chi2(before, after, out_dir / "diagnostics" / "pair_chi2.png",
                       title="pair chi2 before and after the learned correction")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
