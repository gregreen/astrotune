"""The covariance of one catalog entry, built from its *own* quoted sigma.

The delta model corrects what the catalog reported, not what the harmonized
covariance happens to be: a catalog's ``sigma_x_raw`` (its quoted uncertainty, in
our parameter units, before our floors and rescalings) is scaled and inflated in
those units, and only then propagated to the harmonized space with the Jacobian,

    S = J diag(v) J^T,   v_p = exp(ln_scale_p) * sigma_p^2 + exp(ln_add_p),

which makes ``s = exp(ln_scale / 2)`` the factor by which the catalog's own error
is wrong and ``a = exp(ln_add / 2)`` its un-modelled part -- both in the catalog's
units, so they are comparable across catalogs and params. ``J`` is the bijector's
Jacobian at the entry's own parameters (``astrotune.apply``), so this is the same
sandwich ``apply`` uses, with a different (learned) diagonal.

Two conventions are shared with ``astrotune.apply``: an unmeasured parameter
contributes nothing (``v_p = 0``), and it comes out as ``(0, 1e9)`` on the
diagonal with zero off-diagonal entries, so a determinant never sees the
placeholder.

The group's *shared* systematic is deliberately not part of this: it is a
property of the group's harmonized output (two catalogs that ran one pipeline
make the same mistake), it is a single number per group and parameter, and it
enters the likelihood's second level as ``V_g = A_g^-1 + G_g``. Keeping it out of
``S`` is what makes that factorization exact.
"""

from __future__ import annotations

import numpy as np

from .stars import MISSING_VAR, N_PARAMS

#: Placeholder diagonal for a parameter the entry does not measure
_PLACEHOLDER = MISSING_VAR


def entry_cov_np(sig, jac, ln_scale=None, ln_add=None, measured=None) -> np.ndarray:
    """``S = J diag(v) J^T`` for a batch of entries, in float64.

    ``sig`` and ``jac`` are ``(n, 4)`` and ``(n, 4, 4)``; ``ln_scale``/``ln_add``
    are log *variances* (``(n, 4)``) and default to "no correction". With
    ``measured``, the unmeasured axes are masked out of ``v`` and replaced by the
    placeholder afterwards.
    """
    sig = np.asarray(sig, dtype="f8")
    jac = np.asarray(jac, dtype="f8")
    v = sig**2
    if ln_scale is not None:
        v = v * np.exp(np.asarray(ln_scale, dtype="f8"))
    if ln_add is not None:
        v = v + np.exp(np.asarray(ln_add, dtype="f8"))
    m = None
    if measured is not None:
        m = np.asarray(measured, dtype=bool)
        v = np.where(m, v, 0.0)
    cov = np.einsum("nik,nk,njk->nij", jac, v, jac)
    if m is not None:
        cov = np.where(m[:, :, None] & m[:, None, :], cov, 0.0)
        i = np.arange(N_PARAMS)
        cov[:, i, i] += np.where(m, 0.0, _PLACEHOLDER)
    return cov


def entry_cov_tf(sig, jac, ln_scale=None, ln_add=None, measured=None):
    """The same, as TensorFlow ops in float64 (differentiated inside ``star_nll``)."""
    import tensorflow as tf

    sig = tf.cast(sig, tf.float64)
    jac = tf.cast(jac, tf.float64)
    v = sig**2
    if ln_scale is not None:
        v = v * tf.exp(tf.cast(ln_scale, tf.float64))
    if ln_add is not None:
        v = v + tf.exp(tf.cast(ln_add, tf.float64))
    m = None
    if measured is not None:
        m = tf.cast(measured, tf.bool)
        v = tf.where(m, v, tf.zeros_like(v))
    cov = tf.einsum("nik,nk,njk->nij", jac, v, jac)
    if m is not None:
        inside = tf.logical_and(m[:, :, None], m[:, None, :])
        cov = tf.where(inside, cov, tf.zeros_like(cov))
        placeholder = tf.where(m, tf.zeros_like(v),
                               tf.fill(tf.shape(v), tf.constant(_PLACEHOLDER, tf.float64)))
        cov = cov + tf.linalg.diag(placeholder)
    return cov


def propagated_np(sig, jac) -> np.ndarray:
    """``diag(J diag(sig^2) J^T)``: the uncertainties the current pipeline would report."""
    sig = np.asarray(sig, dtype="f8")
    jac = np.asarray(jac, dtype="f8")
    return np.einsum("nik,nk->ni", jac**2, sig**2)


def sigma_out_np(cov) -> np.ndarray:
    """``sqrt(diag(cov))``, with the placeholder axes reported as NaN."""
    d = np.diagonal(np.asarray(cov, dtype="f8"), axis1=1, axis2=2)
    return np.sqrt(np.where(d < 0.5 * _PLACEHOLDER, np.maximum(d, 0.0), np.nan))
