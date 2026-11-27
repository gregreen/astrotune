"""The per-star profile likelihood of the harmonized parameters, in one place.

For one star with observations ``a`` (one per catalog measurement) the data term is

    r_a    = x_a - f_c^-1(y)
    chi2_a = r_a^T C_a^-1 r_a
    rho_a  = (nu_c + d_a) log(1 + chi2_a / nu_c)          Student's t (default)
    rho_a  = S asinh(chi2_a / S)                          the pair loss's softening

with ``C_a`` the observation's covariance (the catalog's own when it quotes
off-diagonals, its diagonal otherwise), ``d_a`` the number of parameters that
observation measures and ``nu_c`` the catalog's degrees of freedom; see
``notes/gauss_newton_with_students_t.tex`` for the derivation of the weight
``w = d rho / d chi2`` that the Gauss-Newton step uses.

``C_a^-1`` is never formed. The observation carries the ten numbers of a
whitening factor ``U`` with ``C^-1 = D^-1 U^T U D^-1``, ``D = diag(sigma)`` and
``U`` lower triangular, so ``chi2_a = ||U r_a / sigma_a||^2`` is a matvec of four
terms. A diagonal covariance -- every catalog but ``sdssv_clam`` -- has ``U``
identically the identity, so one code path serves both kinds of catalog.

Two callers need exactly this likelihood and must not drift apart:

* :class:`astrotune.train_latent.LatentTrainer`, which fits ``y`` for every training
  star while the map is being fitted;
* :class:`astrotune.merge`, which fits ``y`` once more for every star of the merged
  catalog at the frozen final map.

So the estimator lives here, with an explicit interface and no trainer state: the
same numbers, the same weights, one implementation. A parameter that nobody measures
carries ``sigma = 1e9`` and contributes ~1e-18, so no mask is needed; it does leave
the t's dimension ``d``, which is why ``d`` counts the measured parameters rather
than the width of the vector (the marginal of a multivariate t over an axis known
exactly is a t of the same ``nu``).
"""

from __future__ import annotations

from typing import Tuple

import tensorflow as tf

from .train import arcsinh_of_exp

#: ``sigma_x`` above this marks a parameter that the catalog does not measure
MISSING = 1e6

#: The data terms that may be selected
DATA_TERMS = ("student_t", "softened")


@tf.autograph.experimental.do_not_convert
def whiten(z, u=None):
    """Apply the lower-triangular whitening ``U`` to ``z = r / sigma``.

    ``u`` holds ``U = L_R^-1`` for a catalog that quotes covariances -- its four
    diagonal entries followed by the six strict-lower ones (see
    ``prep.transforms.covariance_to_whitening``). For every other catalog it is the
    identity: ``m = 1`` on the diagonal and ``0`` off it, so this returns ``z``
    unchanged (adding 0.0 and multiplying by 1.0 are exact). The diagonal
    approximation is therefore a special case of the same arithmetic and not a
    second code path.

    ``U z`` is written out so that nothing is allocated per observation.
    """
    if u is None:
        return z
    z0 = z[:, 0]
    z1 = z[:, 1]
    z2 = z[:, 2]
    y0 = u[:, 0] * z0
    y1 = u[:, 4] * z0 + u[:, 1] * z1
    y2 = u[:, 5] * z0 + u[:, 6] * z1 + u[:, 2] * z2
    y3 = u[:, 7] * z0 + u[:, 8] * z1 + u[:, 9] * z2 + u[:, 3] * z[:, 3]
    return tf.stack([y0, y1, y2, y3], axis=1)


@tf.autograph.experimental.do_not_convert
def precision_blocks(xs, u=None):
    """``C^-1`` of every observation, ``(n, d, d)``, from sigma and the whitening.

    ``C^-1 = D^-1 U^T U D^-1`` with ``D = diag(sigma)``, which for the identity
    ``U`` is ``diag(1 / sigma^2)``.
    """
    inv_sigma = 1.0 / xs
    if u is None:
        return tf.linalg.diag(inv_sigma**2)
    zero = tf.zeros_like(u[:, 0])
    rows = [
        [u[:, 0], zero, zero, zero],
        [u[:, 4], u[:, 1], zero, zero],
        [u[:, 5], u[:, 6], u[:, 2], zero],
        [u[:, 7], u[:, 8], u[:, 9], u[:, 3]],
    ]
    matrix = tf.stack([tf.stack(row, axis=1) for row in rows], axis=1)  # (n, d, d)
    q = tf.linalg.matmul(matrix, matrix, transpose_a=True)
    return q * inv_sigma[:, :, None] * inv_sigma[:, None, :]


@tf.autograph.experimental.do_not_convert
def loss_inputs(x, xs, x_pred, c, nu_cat, u=None) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """``(chi2, d, nu)`` of every observation.

    ``chi2`` is the squared Mahalanobis residual, ``r^T C^-1 r`` with
    ``r = x - x_pred``; ``d`` is the number of parameters the catalog measures,
    which is the dimension of the Student's t; ``nu`` is the catalog's degrees of
    freedom, gathered by the observation's catalog index. ``u`` is the whitening
    correction of a catalog that quotes covariances, and zero (or ``None``) for
    the rest.
    """
    chi2_obs = tf.reduce_sum(whiten((x - x_pred) / xs, u)**2, axis=1)
    d_obs = tf.reduce_sum(tf.cast(xs < MISSING, tf.float32), axis=1)
    return chi2_obs, d_obs, tf.gather(nu_cat, c)


@tf.autograph.experimental.do_not_convert
def data_term(chi2_obs, d_obs, nu_obs, *, data_term: str = "student_t",
              chi2_turnover: float = 50.0) -> tf.Tensor:
    """The per-observation data term the loss averages.

    ``(nu + d) ln(1 + chi2 / nu)`` for the Student's t: twice the exact deviance,
    chosen so that ``nu -> inf`` gives ``chi2`` itself and the penalties keep their
    calibration. Its curvature at ``chi2 = nu`` is half what it is at zero, against
    0.71 for the softened form, so the t turns over more sharply; far beyond it both
    grow like ``ln chi2``.
    """
    if data_term == "student_t":
        return (nu_obs + d_obs) * tf.math.log1p(chi2_obs / nu_obs)
    return chi2_turnover * arcsinh_of_exp(
        tf.math.log(chi2_obs + 1e-30) - tf.math.log(chi2_turnover)
    )


@tf.autograph.experimental.do_not_convert
def data_weight(chi2_obs, d_obs, nu_obs, *, data_term: str = "student_t",
                chi2_turnover: float = 50.0) -> tf.Tensor:
    """``d rho / d chi2``: the curvature the Gauss-Newton step weights by.

    For the t that is ``(nu + d) / (nu + chi2)``, the weight derived in
    ``notes/gauss_newton_with_students_t.tex``. Freezing it at the current ``y``
    within a step is what keeps the normal matrix positive semi-definite: the exact
    Hessian of the t carries a subtracted rank-one term that turns the system
    indefinite beyond ``chi2 = nu``, and a step off an indefinite matrix overshoots
    the mode.
    """
    if data_term == "student_t":
        return (nu_obs + d_obs) / (nu_obs + chi2_obs)
    return 1.0 / tf.sqrt(1.0 + (chi2_obs / chi2_turnover) ** 2)


@tf.autograph.experimental.do_not_convert
def normal_equations(x, xs, x_pred, jac, weight, star, u=None) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Per-star ``(A, b)`` of the Gauss-Newton step in ``y``.

    The normal equations of one observation are ``J^T w C^-1 J`` and
    ``J^T w C^-1 (x - x_pred)``, where ``w`` is the curvature of the data term
    (:func:`data_weight`) so that the step is the exact minimiser of that loss and
    not merely of a quadratic stand-in, and ``C^-1`` comes from the uncertainty and
    (for a catalog that quotes one) its covariance. With a diagonal ``C`` this is
    ``J^T diag(w / sigma^2) J``, exactly as before. The weight is an argument rather
    than something recomputed here, so the loss and the step cannot drift apart.

    Returns the summed matrices ``(n_star, n_dim, n_dim)``, the vectors
    ``(n_star, n_dim)`` and the ``tf.unique`` of the star index.
    """
    precision = precision_blocks(xs, u) * weight[:, None, None]
    dx = x - x_pred
    A = tf.einsum("npa,npq,nqb->nab", jac, precision, jac)
    b = tf.einsum("npa,npq,nq->na", jac, precision, dx)
    uniq, inv = tf.unique(star)
    n = tf.size(uniq)
    return (
        tf.math.unsorted_segment_sum(A, inv, n),
        tf.math.unsorted_segment_sum(b, inv, n),
        uniq,
    )


@tf.autograph.experimental.do_not_convert
def solve(A, b, *, ridge: float = 1e-3, trust: float = 0.0,
          return_scale: bool = False):
    """Solve the per-star normal equations, with a ridge for flat directions.

    A parameter that none of a star's catalogs measures has ``sigma = 1e9``, so its
    row and column of ``A`` are ~1e-18 and the system is singular; the ridge makes
    the step in that direction ~0 instead of undefined. It is diagonal on purpose: an
    off-diagonal term would leak the other components' steps into a direction they
    say nothing about. It also damps the nearly flat directions, where the full step
    is hundreds of sigma.

    The ridge is relative to the mean of the diagonal, so it is scale free. ``trust``
    caps the largest single-parameter move of any star (0 disables it); the step is
    *scaled*, never clipped component by component, so every component keeps its
    share and the scaled step is still the exact minimiser along that line.
    """
    n_dim = tf.shape(b)[-1]
    scale = tf.reduce_mean(tf.linalg.diag_part(A), axis=1)[:, None, None]
    ridge_matrix = (ridge * scale + 1e-8) * tf.eye(n_dim, dtype=A.dtype)
    step = tf.linalg.solve(A + ridge_matrix, b[:, :, None])[:, :, 0]
    if not trust:
        allowed = tf.ones_like(step[:, 0])
        return (step, allowed) if return_scale else step
    biggest = tf.reduce_max(tf.abs(step), axis=1)
    allowed = tf.minimum(1.0, trust / tf.maximum(biggest, 1e-12))
    step = step * allowed[:, None]
    return (step, allowed) if return_scale else step
