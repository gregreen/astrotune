"""The delta model: a correction to each catalog's own quoted uncertainties.

The correction lives in *catalog* space. For an entry whose catalog quotes
``sigma`` (in our parameter units, before our rescaling and error floors) the
model returns a log variance scale ``ln_scale`` and an additive variance
``ln_add``, and :func:`astrotune.delta.covariance.entry_cov_tf` builds

    v_p = exp(ln_scale_p) * sigma_p^2 + exp(ln_add_p),
    S   = J diag(v) J^T,

with ``J`` the bijector's Jacobian at the entry's own parameters -- the same
sandwich ``astrotune.apply`` forms for the quoted errors, with a learned diagonal
in place of ``sigma^2``. So ``exp(ln_scale / 2)`` is the factor by which the
catalog's quoted error is wrong and ``exp(ln_add / 2)`` the part it does not
report, both in the catalog's units. Three parameterisations are implemented
(:data:`MODEL_TYPES`): one additive term per group and parameter (``const_add``),
scale and additive per group and parameter (``const_scale_add``, the default),
and a FiLM network of the stellar parameters (``network_scale_add``).

The likelihood is the one of the latent-``y`` paradigm. A star has entries
``k = 1..K`` in different catalogs, all measurements of one unknown value ``z``:

    y_k = z + e_k,   e_k ~ N(0, S_k),

with the ``e_k`` independent given ``z``. Marginalising ``z`` with a flat prior
gives, per star,

    -2 ln p = sum_k d_k^T S_k^-1 d_k + ln det A + sum_k ln det S_k + n_res ln 2 pi

with ``d_k = y_k - z``, ``A = sum_k S_k^-1`` and ``z = A^-1 sum_k S_k^-1 y_k``
(the precision-weighted mean, i.e. the three-cornered hat), and ``n_res`` the
dimension of the space of differences: the number of parameters the entries measure
less the number the star has at all. For ``K = 2`` this is *identical* to the old
pair likelihood, constant included, since
``det A = det(S_1 + S_2) / (det S_1 det S_2)`` and ``n_res`` is the dimension of
``y_1 - y_2``; it differs for ``K >= 3``, where the pair version counted
``K (K-1) / 2`` correlated terms instead of one.

Two extensions are used by the fit. Catalogs that run the same pipeline on the
same spectra are grouped, and the group's value is a latent level between ``z``
and its entries (:class:`SharedSystematic`), so their common mistake cancels in
within-group differences; and a per-group tail weight ``nu`` makes the noise a
multivariate Student's t (:class:`TailWeight`). :mod:`astrotune.delta.train` fits
it and ``notes/error_calibration_model.tex`` states the model in full.

A parameter that an entry does not measure carries a placeholder variance of
``MISSING_VAR`` in ``S`` and a harmonized value of 0, so its row and column of
``S_k`` are ``1e9``: its precision ``1e-9`` contributes essentially nothing to
``A``, its residual is 0, and its ``ln det`` contribution is removed by the
mask. A parameter that *no* entry of the star measures leaves ``A`` with a
``1e-8`` diagonal and ``z = 0`` there, which is the same "no information"
answer the trainer's ``y``-ridge gives; the masked ``ln det A`` drops it too.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import sonnet as snt
import tensorflow as tf

from ..model.vector_network import VectorNetwork
from ..registry import group_index
from .covariance import entry_cov_tf
from .stars import N_PARAMS

log = logging.getLogger(__name__)

#: Smooth lower bound of ln_delta: sigma_delta = 0.003 (far below any quoted error)
LN_MIN = float(2 * np.log(3e-3))

#: Weak Gaussian prior on ln_delta: center (sigma = 0.03, the floor the merge used) and width
PRIOR_MEAN = float(2 * np.log(0.03))
PRIOR_SD = 3.0

#: Soft bounds on the covariance scale, in variance units: sigma stays within 1/8 .. 8
#: times the quoted error. Symmetric in log space, so the clamp has no preference.
SCALE_MIN = 1.0 / 64.0
SCALE_MAX = 64.0
LOG_SCALE_MIN = float(np.log(SCALE_MIN))
LOG_SCALE_MAX = float(np.log(SCALE_MAX))

#: Prior widths: "the quoted errors are right" and "nothing is added", both weak
PRIOR_SCALE_SD = 1.0
PRIOR_ADD_SD = 4.0

#: Where a fresh additive term starts: on the floor, so it can only be raised by data
ADD_INIT = LN_MIN - 10.0

#: The additive sigma at the floor (0.003): "nothing is added"
ADD_MIN_SIGMA = float(np.exp(0.5 * LN_MIN))

# --- Student's t tail weight ---------------------------------------------------
#: nu = NU_MIN + softplus(raw): the degrees of freedom of every catalog, kept above 2
#: so the variance exists. Softplus can reach any large value, so a catalog that wants
#: Gaussian tails is free to say so.
NU_MIN = 2.0

#: Prior on the unconstrained coordinate: centred on nu = 8 (a tail heavy enough to
#: matter -- excess kurtosis 6 / (nu - 4) = 1.5 -- but close enough to Gaussian that
#: "no evidence" changes nothing), with sd 2 in theta, i.e. nu in [4.1, 12.0] at 95 %.
NU_PRIOR_MEAN = 6.0
NU_PRIOR_SD = 2.0
NU_INIT_RAW = 6.0

# --- Shared systematic ---------------------------------------------------------
#: Where each group's shared variance starts: at the floor, i.e. independent errors
SHARED_INIT = -10.0

#: Width of the prior that pulls a group's shared variance back to the floor
SHARED_PRIOR_SD = 3.0


class SharedSystematic(snt.Module):
    """Variance shared by the catalogs of a group: ``G_g = diag(tau_g^2)``.

    Two catalogs that ran the same pipeline on the same spectra make the same
    mistake, so their errors have a common part. It cannot be represented by
    enlarging either covariance -- that inflates their *difference* instead of
    explaining it -- so it enters as a latent value between the star and its
    entries (see ``star_nll``). A group of one catalog has no shared part at all:
    the multiplicity mask makes ``tau^2`` exactly zero there.
    """

    def __init__(self, n_group: int, multiplicity: Sequence[int],
                 init: float = SHARED_INIT, name=None):
        super().__init__(name=name)
        multi = np.asarray([1.0 if m > 1 else 0.0 for m in multiplicity], dtype="f4")
        self.multi = tf.constant(multi[:, None])  # (n_group, 1)
        self.raw = tf.Variable(tf.fill([n_group, N_PARAMS], float(init)), name="raw_tau")

    def variance(self):
        """``(n_group, 4)``: the shared variance, exactly 0 for singleton groups."""
        return self.multi * (float(ADD_MIN_SIGMA) ** 2 + tf.nn.softplus(self.raw))

    def has_shared(self) -> bool:
        """Whether any group has more than one catalog (a static property)."""
        return bool(float(tf.reduce_max(self.multi)) > 0.0)

    def prior_penalty(self, sd: float = SHARED_PRIOR_SD):
        """Pull the shared sigma towards the floor: no shared term unless needed."""
        v = tf.maximum(self.variance(), 1e-30)
        z2 = ((tf.math.log(v) - LN_MIN) / sd) ** 2
        return 0.5 * tf.reduce_sum(z2 * self.multi) / tf.maximum(tf.reduce_sum(self.multi) * N_PARAMS, 1.0)


class TailWeight(snt.Module):
    """One degrees-of-freedom parameter per catalog, ``nu_c = 2 + softplus(raw_c)``.

    A single number per catalog is the right granularity: the outlier fraction is a
    property of the pipeline (blends, bad matches, unmodelled binaries), not of the
    star. The prior is Gaussian on ``raw``, which keeps ``nu > 2`` without a clamp and
    lets a catalog go to effectively Gaussian tails if the data ask for it.

    ``prior_penalty`` also takes a prior that is Gaussian in ``nu`` itself
    (``space="nu"``). Because ``nu = 2 + softplus(raw)`` is bounded below by 2, that
    prior is the half-Gaussian ``p(nu) ~ exp(-(nu - nu0)^2 / 2 sd^2)`` on ``nu >= 2``
    and zero below. It is the better-behaved of the two: a Gaussian in ``raw`` pushes
    ``p(nu)`` *up* as ``nu -> 2`` from above, because a shrinking stretch of ``raw``
    maps to each ``d nu`` there, so it puts a divergent spike of prior mass on
    almost-Cauchy tails that no catalog is ever shown to want.
    """

    def __init__(self, n_cat: int, raw_init: float = NU_INIT_RAW, name=None):
        super().__init__(name=name)
        self.raw = tf.Variable(tf.fill([n_cat], float(raw_init)), name="raw_nu")

    def __call__(self):
        return NU_MIN + tf.math.softplus(self.raw)

    def prior_penalty(self, mean: float = NU_PRIOR_MEAN, sd: float = NU_PRIOR_SD,
                      space: str = "raw"):
        """Mean over catalogs of ``0.5 ((x - mean) / sd)^2``, ``x`` = raw or nu."""
        if space == "nu":
            # The prior is on nu, and softplus already enforces nu > 2, so this is
            # the truncated Gaussian without any extra bookkeeping.
            return 0.5 * tf.reduce_mean(((self() - mean) / sd) ** 2)
        if space != "raw":
            raise ValueError(f"unknown nu prior space {space!r}; use 'raw' or 'nu'")
        return 0.5 * tf.reduce_mean(((self.raw - mean) / sd) ** 2)

#: Per-entry batch arrays, plus the star each entry belongs to
BATCH_KEYS = ("x", "ls", "sig", "J", "c", "g", "y", "m")

#: ``fn(x, ln_sigma, catalog) -> (ln_scale, ln_add)``, each ``(n, 4)``
DeltaFn = Callable[[tf.Tensor, tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]


def floor_ln_delta(raw):
    """``LN_MIN + softplus(raw - LN_MIN)``: equals ``raw`` well above ``LN_MIN``,
    tends smoothly to ``LN_MIN`` below it (so gradients never vanish abruptly)."""
    return LN_MIN + tf.math.softplus(raw - LN_MIN)


def soft_clamp(raw, lo, hi, sharpness: float = 8.0):
    """Identity between ``lo`` and ``hi``, saturating smoothly outside.

    The two-sided analogue of :func:`floor_ln_delta`, which is the same thing written
    for a single bound. ``sharpness`` narrows the rounding region, so that the
    identity part covers most of ``[lo, hi]``: with the default bounds (``+-ln 16``)
    the map is within 1e-3 of the identity on ``[-2, 2]`` and never leaves
    ``[lo - 0.004, hi]``.
    """
    def sp(y):
        return tf.math.softplus(sharpness * y) / sharpness

    return hi - sp(hi - lo - sp(raw - lo))


class DeltaNetwork(snt.Module):
    """FiLM network ``(x, ln_sigma_x, catalog) -> ln_delta_cov`` (4 values).

    The additive model: the covariance scale is fixed at 1. Kept so that runs trained
    before the scale term existed still load and still compare.
    """

    def __init__(self, n_cat: int, hidden_size: int = 64, n_hidden: int = 3,
                 multiplicity=None, name=None):
        super().__init__(name=name)
        self.n_cat = n_cat
        self.tail = TailWeight(n_cat)
        self.net = VectorNetwork(
            n_cat, N_PARAMS, n_hidden, hidden_size,
            output_scale=1e-3, output_bias=PRIOR_MEAN,
            use_residual=True, use_film=True, use_layer_norm=True,
        )
        self(tf.zeros([1, N_PARAMS]), tf.zeros([1, N_PARAMS]), tf.zeros([1], dtype=tf.int32))  # build

    def __call__(self, x, ln_sigma, c):
        return tf.zeros(tf.shape(x), tf.float32), floor_ln_delta(self.net(tf.concat([x, ln_sigma], axis=1), c))

    def nu(self):
        return self.tail()


class ScaleAddNetwork(snt.Module):
    """Two FiLM heads ``(theta, catalog) -> (ln_scale, ln_add)``.

    The correction is a function of the stellar parameters only: ``ln_sigma`` is part
    of the signature but is deliberately not fed to either head, so the model answers
    "what is wrong with this catalog here", not "what is wrong with this measurement".
    How much that matters is measured by the scale itself, which multiplies the quoted
    error rather than adding a fixed amount to it.

    ``ln_scale`` is the log of a *variance* scale, soft-clamped to
    ``[ln 1/64, ln 64]``; ``ln_add`` is floored at ``LN_MIN`` and starts there.
    """

    def __init__(self, n_cat: int, hidden_size: int = 64, n_hidden: int = 3,
                 multiplicity=None, name=None):
        super().__init__(name=name)
        self.n_cat = n_cat
        self.tail = TailWeight(n_cat)
        self.shared = SharedSystematic(n_cat, multiplicity or [1] * n_cat)
        kw = dict(use_residual=True, use_film=True, use_layer_norm=True)
        self.scale = VectorNetwork(n_cat, N_PARAMS, n_hidden, hidden_size,
                                   output_scale=1e-3, output_bias=0.0, **kw)
        self.add = VectorNetwork(n_cat, N_PARAMS, n_hidden, hidden_size,
                                 output_scale=1e-3, output_bias=ADD_INIT, **kw)
        self(tf.zeros([1, N_PARAMS]), tf.zeros([1, N_PARAMS]), tf.zeros([1], dtype=tf.int32))  # build

    def __call__(self, x, ln_sigma, c):
        log_scale = soft_clamp(self.scale(x, c), LOG_SCALE_MIN, LOG_SCALE_MAX)
        return log_scale, floor_ln_delta(self.add(x, c))

    def nu(self):
        return self.tail()

    def shared_variance(self):
        return self.shared.variance()


class ConstDelta(snt.Module):
    """One ln_delta per group and parameter (the additive-only baseline)."""

    def __init__(self, n_cat: int, multiplicity=None, name=None):
        super().__init__(name=name)
        self.raw = tf.Variable(tf.fill([n_cat, N_PARAMS], PRIOR_MEAN), name="raw")
        self.tail = TailWeight(n_cat)
        self.shared = SharedSystematic(n_cat, multiplicity or [1] * n_cat)

    def __call__(self, x, ln_sigma, c):
        return tf.zeros(tf.shape(x), tf.float32), floor_ln_delta(tf.gather(self.raw, c))

    def nu(self):
        return self.tail()

    def shared_variance(self):
        return self.shared.variance()


class ConstScaleAdd(snt.Module):
    """One scale and one additive term per group and parameter.

    Eight parameters per group instead of four. This is the arm that says whether
    the signal is multiplicative at all, and it is the reference the network has to
    beat; ``raw_scale`` starts at 0 (scale 1) and ``raw_add`` on the floor.
    """

    def __init__(self, n_cat: int, multiplicity=None, name=None):
        super().__init__(name=name)
        self.raw_scale = tf.Variable(tf.zeros([n_cat, N_PARAMS]), name="raw_scale")
        self.raw_add = tf.Variable(tf.fill([n_cat, N_PARAMS], ADD_INIT), name="raw_add")
        self.tail = TailWeight(n_cat)
        self.shared = SharedSystematic(n_cat, multiplicity or [1] * n_cat)

    def __call__(self, x, ln_sigma, c):
        log_scale = soft_clamp(tf.gather(self.raw_scale, c), LOG_SCALE_MIN, LOG_SCALE_MAX)
        return log_scale, floor_ln_delta(tf.gather(self.raw_add, c))

    def nu(self):
        return self.tail()

    def shared_variance(self):
        return self.shared.variance()

    def scale_sigma(self) -> np.ndarray:
        """The fitted sigma scale per group and parameter (``(n_group, 4)``)."""
        return np.exp(0.5 * soft_clamp(self.raw_scale, LOG_SCALE_MIN, LOG_SCALE_MAX).numpy())

    def add_sigma(self) -> np.ndarray:
        return np.exp(0.5 * floor_ln_delta(self.raw_add).numpy())


def fixed_delta(sigma: float) -> DeltaFn:
    """The same added sigma everywhere (``0`` for no added variance), scale 1."""
    value = -60.0 if sigma <= 0 else float(2 * np.log(sigma))

    def fn(x, ln_sigma, c):
        zero = tf.fill(tf.shape(x), tf.constant(0.0, dtype=x.dtype))
        return zero, tf.fill(tf.shape(x), tf.constant(value, dtype=x.dtype))

    return fn


def make_batch(stars, idx, seg) -> Dict[str, object]:
    """One block of whole stars as tensors: entry arrays plus the star segment."""
    return {
        **{k: tf.constant(stars[k][idx]) for k in BATCH_KEYS},
        "seg": tf.constant(np.asarray(seg, dtype="i4")),
    }


def _marginal(cov, y, m, seg, n_seg) -> dict:
    """Pieces of the Gaussian marginal over one common value per segment.

    For segments holding entries with covariance ``cov`` around a common value, this
    returns the precision sum ``A`` and its inverse, the weighted mean ``z``, the
    residual quadratic form ``quad``, the determinant terms and the dimension counts.
    Used at both levels of the group hierarchy: segments are groups at the lower
    level and stars at the upper one.
    """
    L = tf.linalg.cholesky(cov)
    Sinv = tf.linalg.inv(cov)  # symmetric positive definite; the Cholesky is for ln det
    A = tf.math.unsorted_segment_sum(Sinv, seg, n_seg)  # (S, 4, 4)
    b = tf.math.unsorted_segment_sum(tf.linalg.matvec(Sinv, y), seg, n_seg)  # (S, 4)
    LA = tf.linalg.cholesky(A)
    z = tf.linalg.cholesky_solve(LA, b[..., None])[..., 0]  # (S, 4)
    d = y - tf.gather(z, seg)  # (N, 4)
    quad = tf.reduce_sum(d * tf.linalg.matvec(Sinv, d), axis=1)  # (N,)
    # ln det cov over the parameters this member measures: the placeholder axes of an
    # unmeasured parameter are 1e9 and must not contribute ln(1e9).
    logdet = 2.0 * tf.reduce_sum(m * tf.math.log(tf.linalg.diag_part(L)), axis=1)
    # ln det A over the parameters *some* member of the segment measures.
    any_m = tf.cast(tf.math.unsorted_segment_max(m, seg, n_seg), tf.float64)  # (S, 4)
    return dict(
        A=A, Ainv=tf.linalg.inv(A), z=z, any_m=any_m,
        quad=tf.math.unsorted_segment_sum(quad, seg, n_seg),
        logdet=tf.math.unsorted_segment_sum(logdet, seg, n_seg),
        logdet_A=2.0 * tf.reduce_sum(any_m * tf.math.log(tf.linalg.diag_part(LA)), axis=1),
        n_meas=tf.math.unsorted_segment_sum(tf.reduce_sum(m, axis=1), seg, n_seg),
        d_shared=tf.reduce_sum(any_m, axis=1),
    )


def _gaussian_neg2lnp(p: dict) -> tf.Tensor:
    """``-2 ln p`` of the differences, before any tail weight is applied."""
    n_res = p["n_meas"] - p["d_shared"]
    return (p["quad"] + p["logdet_A"] + p["logdet"]
            + n_res * tf.cast(np.log(2 * np.pi), tf.float64))


def star_nll(batch: Dict[str, object], delta_fn: DeltaFn,
             nu: Optional[tf.Tensor] = None, n_group: Optional[int] = None,
             shared: Optional[tf.Tensor] = None) -> Tuple[tf.Tensor, tuple]:
    """Per-star negative log likelihood, in float64; also returns the model's parameters.

    The sum over a star's entries is a segmented reduction, so a batch is a flat
    array of entries plus the index of the star each belongs to: no loop over
    observations and no ragged tensor. Every star of the block must have at least
    one entry, which is what :func:`astrotune.delta.stars.block_indices` yields.

    The entry covariance is ``S_ij = s_i C_ij s_j + delta_ij add_i`` with
    ``s_i = exp(ln_scale_i / 2)``: a per-parameter scale applied to the whole
    propagated matrix (positive definite for any ``s > 0``, since it is a congruence
    of ``C``) plus a diagonal term. ``ln_scale = 0`` and ``ln_add -> -inf``
    reproduce the untransformed covariance exactly.

    ``n_group``/``shared`` switch on the group hierarchy. With them, entries that
    belong to the same group share a latent value (``t_g = z + u_g``,
    ``u_g ~ N(0, diag(shared[g]))``), so the group's errors are partly common. The
    likelihood then factorises into the same marginal applied twice: within each
    group, whose ``A^-1 + diag(shared)`` becomes the covariance of one effective
    observation of the star's value, and across those observations. With
    ``shared = 0`` the two levels are algebraically identical to the flat one.

    ``nu`` (one degree of freedom per group, or None for Gaussian errors) turns the
    marginal into a multivariate Student's t. The likelihood above is a Gaussian
    density in the ``n_res`` differences with quadratic form ``quad``; giving the
    star's noise a Gamma-distributed precision ``u`` and integrating ``u`` out turns
    it into ``t_nu``, because

        integral u^(nu/2 - 1) e^(-nu u / 2) u^(n_res/2) e^(-u quad/2) du

    is a Gamma integral: the result is ``(nu + n_res) ln(1 + quad/nu)`` plus constants.
    So the t costs no iteration, needs no augmentation, and keeps every other line of
    the model -- masks, determinants, priors -- unchanged. The star's ``u`` is shared
    by all its entries, because that is what "this star is a binary / a bad match"
    means; for a star whose groups have different nu we use their mean, which is the
    only choice that reduces to the global-nu model when all of them agree. Per-*entry*
    heavy tails are a different model: the product of t densities has no closed-form
    marginal over ``z`` and would need data augmentation.
    """
    seg = batch["seg"]
    n_star = tf.reduce_max(seg) + 1  # a dynamic segment count: no retracing per block
    m = tf.cast(batch["m"], tf.float64)  # (N, 4): measured by this entry
    ln_scale, ln_add = delta_fn(batch["x"], batch["ls"], batch["g"])  # (N, 4) each

    y = tf.cast(batch["y"], tf.float64)
    # The correction acts on the catalog's *own* quoted sigma, and the result is
    # propagated to the harmonized space with the bijector's Jacobian.
    S = entry_cov_tf(batch["sig"], batch["J"], ln_scale, ln_add, measured=batch["m"])
    S.set_shape([None, N_PARAMS, N_PARAMS])

    if shared is None:
        p = _marginal(S, y, m, seg, n_star)
        quad_s = p["quad"]
        det = p["logdet_A"] + p["logdet"]
        n_res = p["n_meas"] - p["d_shared"]
    else:
        # Level 1: one effective observation per (star, group).
        slot = seg * tf.constant(int(n_group), tf.int32) + tf.cast(batch["g"], tf.int32)
        uniq, inv = tf.unique(tf.reshape(slot, [-1]))  # compact (star, group) ids
        n_slot = tf.shape(uniq)[0]
        p1 = _marginal(S, y, m, inv, n_slot)
        V = p1["Ainv"] + tf.linalg.diag(
            tf.gather(tf.cast(shared, tf.float64), uniq % int(n_group)))
        # Level 2: those observations around the star's value.
        star_of_slot = uniq // int(n_group)
        p2 = _marginal(V, p1["z"], p1["any_m"], star_of_slot, n_star)
        # The two levels together are one density in the space of all the star's
        # differences: sum the quadratic forms and the determinant terms, and count
        # the dimensions of both levels.
        quad_s = p2["quad"] + tf.math.unsorted_segment_sum(p1["quad"], star_of_slot, n_star)
        det = (p2["logdet_A"] + p2["logdet"] + tf.math.unsorted_segment_sum(
            p1["logdet_A"] + p1["logdet"], star_of_slot, n_star))
        n_res = (p2["n_meas"] - p2["d_shared"] + tf.math.unsorted_segment_sum(
            p1["n_meas"] - p1["d_shared"], star_of_slot, n_star))

    if nu is None:
        neg2lnp = (quad_s + det
                   + n_res * tf.cast(np.log(2 * np.pi), tf.float64))
    else:
        # nu is per group; the star's tail weight is the mean over its entries. The
        # scale mixture multiplies every covariance by 1/u, so it scales the whole
        # difference-space quadratic form and leaves the determinant terms alone.
        nu_s = tf.math.unsorted_segment_mean(
            tf.gather(tf.cast(nu, tf.float64), batch["g"]), seg, n_star)
        k = tf.cast(n_res, tf.float64)
        neg2lnp = (det
                   + (nu_s + k) * tf.math.log1p(quad_s / nu_s)
                   + 2.0 * tf.math.lgamma(0.5 * nu_s) - 2.0 * tf.math.lgamma(0.5 * (nu_s + k))
                   + k * tf.math.log(nu_s * tf.cast(np.pi, tf.float64)))
    return 0.5 * neg2lnp, (ln_scale, ln_add)


def prior_penalty(batch: Dict[str, object], params: tuple) -> tf.Tensor:
    """Mean over measured parameters of the two Gaussian priors.

    ``ln_scale ~ N(0, PRIOR_SCALE_SD)``: the quoted errors are right unless the data
    say otherwise. ``ln_add ~ N(LN_MIN, PRIOR_ADD_SD)``: the additive term sits on the
    floor unless the data lift it, which is why the centre is the floor and not the
    0.03 the merge uses. One term per *entry*, which the star framing gives for free:
    the pair version applied it once per pair, i.e. ``K - 1`` times to the same entry.
    """
    ln_scale, ln_add = params
    m = tf.cast(batch["m"], tf.float32)
    z2_scale = (tf.cast(ln_scale, tf.float32) / PRIOR_SCALE_SD) ** 2
    # Anything at or below the floor is "nothing added": the floor is the smallest
    # term the model can return, so it is where the prior's centre belongs.
    ln_add = tf.maximum(tf.cast(ln_add, tf.float32), LN_MIN)
    z2_add = ((ln_add - LN_MIN) / PRIOR_ADD_SD) ** 2
    return 0.5 * tf.reduce_sum((z2_scale + z2_add) * m) / tf.maximum(tf.reduce_sum(m), 1.0)


def control_points(stars, keys: Sequence[str], n_per_catalog: int = 1024,
                   seed: int = 0) -> Dict[str, np.ndarray]:
    """Dispersed points in the stellar-parameter box, one set per catalog.

    The priors above act where there *is* data; these act everywhere else, which is
    where a per-catalog network is otherwise free to say anything. They are drawn
    from a low-discrepancy sequence over the box spanned by the 1st to 99th percentile
    of the training stars (per parameter, over the entries that measure it), so a
    fixed budget covers the box evenly instead of clustering in the middle the way a
    wide Gaussian would.

    A parameter that a catalog does not measure is set to 0 -- the placeholder the
    network sees in the data itself -- and masked out of the penalty, so the model is
    never pinned on an axis the catalog cannot see.
    """
    x, m, c = stars["x"], stars["m"], stars["c"]
    lo = np.zeros(N_PARAMS)
    hi = np.ones(N_PARAMS)
    for p in range(N_PARAMS):
        sel = m[:, p]
        if sel.sum() >= 10:
            lo[p], hi[p] = np.percentile(x[sel, p].astype("f8"), [1.0, 99.0])
    # Which parameters each catalog measures at all (a catalog absent from the star
    # table has no entries, so it is pinned on every axis)
    measures = np.zeros((len(keys), N_PARAMS), dtype=bool)
    for k in range(len(keys)):
        in_cat = c == k
        measures[k] = m[in_cat].any(0) if in_cat.any() else True

    unit = _dispersed_unit(n_per_catalog * len(keys), N_PARAMS, seed)
    group_of = group_index(keys)
    xs, cs, gs, ms = [], [], [], []
    for k in range(len(keys)):
        u = unit[k * n_per_catalog:(k + 1) * n_per_catalog]
        pts = lo[None, :] + (hi - lo)[None, :] * u
        pts[:, ~measures[k]] = 0.0
        xs.append(pts.astype("f4"))
        cs.append(np.full(n_per_catalog, k, dtype="i4"))
        gs.append(np.full(n_per_catalog, group_of[k], dtype="i4"))
        ms.append(np.tile(measures[k], (n_per_catalog, 1)))
    return dict(x=np.concatenate(xs), c=np.concatenate(cs), g=np.concatenate(gs),
                m=np.concatenate(ms))


def _dispersed_unit(n: int, dim: int, seed: int) -> np.ndarray:
    """``n`` points in ``[0, 1]^dim``: a Sobol sequence if scipy has one, else uniform."""
    try:
        from scipy.stats import qmc

        m = int(np.ceil(np.log2(max(n, 2))))
        return qmc.Sobol(dim, scramble=True, seed=seed).random_base2(m)[:n]
    except Exception:  # pragma: no cover - older scipy
        return np.random.default_rng(seed).random((n, dim))


def control_penalty(ctrl: Dict[str, np.ndarray], model) -> tf.Tensor:
    """The same prior as :func:`prior_penalty`, evaluated away from the data."""
    cat = ctrl["g"] if "g" in ctrl else ctrl["c"]  # the model's category is the group
    ln_scale, ln_add = model(tf.constant(ctrl["x"], tf.float32),
                             tf.zeros_like(tf.constant(ctrl["x"], tf.float32)),
                             tf.constant(cat, tf.int32))
    m = tf.constant(ctrl["m"].astype("f4"))
    z2_scale = (ln_scale / PRIOR_SCALE_SD) ** 2
    ln_add = tf.maximum(ln_add, LN_MIN)
    z2_add = ((ln_add - LN_MIN) / PRIOR_ADD_SD) ** 2
    return 0.5 * tf.reduce_sum((z2_scale + z2_add) * m) / tf.maximum(tf.reduce_sum(m), 1.0)


def l2_penalty(module: snt.Module):
    ws = [v for v in module.trainable_variables if v.shape.rank == 2]
    return tf.add_n([tf.nn.l2_loss(v) for v in ws]) if ws else tf.constant(0.0)


# --- Numpy interface (diagnostics) --------------------------------------------
def numpy_delta_fn(model, batch_size: int = 65536):
    """``f(x, ln_sigma, catalog) -> (ln_scale, ln_add)`` on NumPy arrays, in batches."""

    def fn(x, ln_sigma, c):
        scale, add = [], []
        for i in range(0, len(x), batch_size):
            sl = slice(i, i + batch_size)
            la, lb = model(tf.constant(x[sl], tf.float32),
                           tf.constant(ln_sigma[sl], tf.float32),
                           tf.constant(c[sl], tf.int32))
            scale.append(np.asarray(la.numpy()))
            add.append(np.asarray(lb.numpy()))
        if not scale:
            empty = np.zeros((0, N_PARAMS), dtype="f4")
            return empty, empty
        return np.concatenate(scale), np.concatenate(add)

    return fn


def model_nu(model) -> Optional[np.ndarray]:
    """The fitted ``nu`` per group, or None if the model is Gaussian."""
    return model.nu().numpy().astype("f8") if hasattr(model, "nu") else None


def model_shared(model) -> Optional[np.ndarray]:
    """The fitted shared variance per group and parameter, or None if there is none.

    ``(n_group, 4)``; a group of one catalog has exactly zero. This is what has to be
    added to an entry's own covariance when the entry is compared with anything
    outside its group, so it is the piece the merge needs.
    """
    if not hasattr(model, "shared_variance"):
        return None
    return model.shared_variance().numpy().astype("f8")


#: Which module a saved ``delta_model_config.json`` was written for.
#:
#: The ``const_`` models carry one scale and one additive term per group and
#: parameter and ignore ``x`` entirely, so they cannot absorb structure in the
#: stellar parameters; the ``network_`` ones take ``(x, ln sigma)`` and can.
#: ``add`` and ``scale_add`` are the names earlier runs were written under, and are
#: read back as the network forms they were.
MODEL_TYPES: Dict[str, type] = {
    "const_add": ConstDelta,
    "const_scale_add": ConstScaleAdd,
    "network_add": DeltaNetwork,
    "network_scale_add": ScaleAddNetwork,
}
LEGACY_TYPE_ALIASES: Dict[str, str] = {"add": "network_add", "scale_add": "network_scale_add"}
#: Every type that can be loaded or trained.
MODEL_CHOICES = tuple(MODEL_TYPES)
#: What ``--model-type all`` trains. ``network_add`` is loadable but not fitted by
#: default: it is the additive-only network that predates the scale term.
ALL_MODEL_TYPES = ("const_add", "const_scale_add", "network_scale_add")


def canonical_model_type(name: str) -> str:
    """The canonical name of a model type, accepting the legacy aliases."""
    key = str(name).strip().lower()
    key = LEGACY_TYPE_ALIASES.get(key, key)
    if key not in MODEL_TYPES:
        raise ValueError(
            f"unknown delta model type {name!r}; choose from {MODEL_CHOICES} "
            f"(or {sorted(LEGACY_TYPE_ALIASES)})"
        )
    return key


def build_delta_model(model_type: str, n_cat: int, hidden_size: int = 64,
                      n_hidden: int = 3, multiplicity=None):
    """Construct the module recorded under ``model_type``.

    The constant models take ``(n_cat, multiplicity)``; the networks also take the
    width and depth. One entry point so the loader and the trainer cannot disagree
    about how a type is built.
    """
    kind = canonical_model_type(model_type)
    if kind in ("const_add", "const_scale_add"):
        return MODEL_TYPES[kind](n_cat, multiplicity)
    return MODEL_TYPES[kind](n_cat, hidden_size, n_hidden, multiplicity=multiplicity)


def save_delta_model(out_dir: Path, model, config: dict) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = tf.train.Checkpoint(model=model).write(str(out_dir / "delta_model"))
    with open(out_dir / "delta_model_config.json", "w") as f:
        json.dump(config, f, indent=1)
    return Path(path)


def saved_model_config(out_dir: Path) -> dict:
    """The config a saved model was written with."""
    with open(Path(out_dir) / "delta_model_config.json") as f:
        return json.load(f)


def delta_model_type(out_dir: Path) -> str:
    """The canonical type name of the model saved in ``out_dir``."""
    return canonical_model_type(saved_model_config(out_dir).get("model_type", "add"))


def load_delta_model(out_dir: Path, keys=None):
    """Restore a model saved by :func:`save_delta_model`.

    ``n_cat`` alone would not catch a model trained on a different catalog list of
    the same length, whose category indices would silently mean other catalogs, so
    the saved keys are compared whenever ``keys`` is given. Configs written before
    the scale term existed carry no ``model_type`` and are read as additive-only.
    """
    out_dir = Path(out_dir)
    cfg = saved_model_config(out_dir)
    model_type = cfg.get("model_type", "add")
    if keys is not None and tuple(cfg.get("keys", ())) != tuple(keys):
        raise ValueError(
            f"{out_dir}/delta_model was trained on {list(cfg.get('keys', ()))}, "
            f"not {list(keys)}; retrain it."
        )
    model = build_delta_model(model_type, cfg["n_cat"], cfg.get("hidden_size", 64),
                              cfg.get("n_hidden", 3), cfg.get("multiplicity"))
    tf.train.Checkpoint(model=model).read(str(out_dir / "delta_model")).assert_existing_objects_matched()
    return model
