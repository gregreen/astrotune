"""The per-star profile likelihood of the harmonized parameters, in one place.

For one star with observations ``a`` (one per catalog measurement) the data term is

    chi2_a = sum_p ( x_{a,p} - [f_c^-1(y)]_p )^2 / sigma_{a,p}^2
    rho_a  = (nu_c + d_a) log(1 + chi2_a / nu_c)          Student's t (default)
    rho_a  = S asinh(chi2_a / S)                          the pair loss's softening

with ``d_a`` the number of parameters that observation measures and ``nu_c`` the
catalog's degrees of freedom; see ``notes/gauss_newton_with_students_t.tex`` for the
derivation of the weight ``w = d rho / d chi2`` that the Gauss-Newton step uses.

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
def loss_inputs(x, xs, x_pred, c, nu_cat) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """``(chi2, d, nu)`` of every observation.

    ``chi2`` is the squared Mahalanobis residual; ``d`` is the number of parameters
    the catalog measures, which is the dimension of the Student's t; ``nu`` is the
    catalog's degrees of freedom, gathered by the observation's catalog index.
    """
    chi2_obs = tf.reduce_sum((x - x_pred) ** 2 / xs**2, axis=1)
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
def normal_equations(x, xs, x_pred, jac, weight, star) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Per-star ``(A, b)`` of the Gauss-Newton step in ``y``.

    With ``C = diag(sigma^2)`` diagonal, the normal equations of one observation are
    ``J^T W J`` and ``J^T W (x - x_pred)``, ``W = w / sigma^2`` elementwise, where
    ``w`` is the curvature of the data term (:func:`data_weight`) so that the step is
    the exact minimiser of that loss and not merely of a quadratic stand-in. The
    weight is an argument rather than something recomputed here, so the loss and the
    step cannot drift apart.

    Returns the summed matrices ``(n_star, n_dim, n_dim)``, the vectors
    ``(n_star, n_dim)`` and the ``tf.unique`` of the star index.
    """
    weight = weight[:, None] / xs**2
    dx = x - x_pred
    A = tf.einsum("npa,np,npb->nab", jac, weight, jac)
    b = tf.einsum("npa,np,np->na", jac, weight, dx)
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
