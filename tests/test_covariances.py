"""Covariances in the likelihood: the whitening factor and the diagonal limit.

The point of these tests is the property that lets one code path serve both kinds
of catalog: a diagonal covariance has a whitening correction of exactly zero, so
the same arithmetic reduces to the per-parameter chi-squared of before.
"""

import numpy as np
import pytest
import tensorflow as tf

from astrotune import profile
from astrotune.prep import transforms


def _identity(n):
    """The whitening factor of a diagonal catalog: unit diagonal, zero elsewhere."""
    u = np.zeros((n, transforms.WHITENING_SIZE), dtype="f4")
    u[:, :4] = 1.0
    return u


def _random_blocks(n=20, seed=0):
    """Positive-definite 4x4 blocks with real off-diagonals.

    Built as ``R = Q diag(lambda) Q^T`` from a random orthogonal ``Q`` (so it is
    positive definite by construction, not by luck), then normalised to unit
    diagonal and given a wide spread of sigmas.
    """
    rng = np.random.default_rng(seed)
    blocks = np.empty((n, 4, 4))
    for i in range(n):
        q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
        evals = np.exp(rng.uniform(-3.0, 0.7, size=4))
        corr = (q * evals) @ q.T
        d = np.sqrt(np.diag(corr))
        blocks[i] = corr / (d[:, None] * d[None, :])
    sigma = np.exp(rng.normal(scale=0.5, size=(n, 4))) * 0.1
    return blocks * sigma[:, :, None] * sigma[:, None, :]


def test_whitening_reproduces_the_mahalanobis_norm():
    cov = _random_blocks()
    u = transforms.covariance_to_whitening(cov)
    sigma = transforms.covariance_to_sigma(cov)
    r = np.random.default_rng(1).normal(size=(len(cov), 4))
    direct = np.einsum("na,nab,nb->n", r, np.linalg.inv(cov), r)
    z = transforms.whitening_to_matrix(u) @ (r / sigma)[:, :, None]
    assert np.allclose((z[:, :, 0] ** 2).sum(1), direct, rtol=1e-5)
    # and the corrections are actually being exercised
    assert np.abs(u).max() > 0.05


def test_a_diagonal_block_has_an_identity_whitening():
    cov = np.stack([np.diag(d) for d in
                    np.random.default_rng(2).uniform(0.01, 1, (20, 4))])
    u = transforms.covariance_to_whitening(cov)
    assert u.shape == (20, transforms.WHITENING_SIZE)
    assert np.all(u[:, :4] == 1.0)
    assert np.all(u[:, 4:] == 0.0)


def test_whiten_is_the_identity_for_a_diagonal_catalog():
    z = tf.constant(np.random.default_rng(3).normal(size=(7, 4)), tf.float32)
    identity = np.zeros((7, transforms.WHITENING_SIZE), dtype="f4")
    identity[:, :4] = 1.0
    assert np.array_equal(profile.whiten(z, tf.constant(identity)).numpy(), z.numpy())
    assert np.array_equal(profile.whiten(z, None).numpy(), z.numpy())


def test_precision_blocks_match_the_inverse_covariance():
    cov = _random_blocks()
    sigma = transforms.covariance_to_sigma(cov)
    u = transforms.covariance_to_whitening(cov)
    precision = profile.precision_blocks(tf.constant(sigma, tf.float64),
                                        tf.constant(u, tf.float64)).numpy()
    assert np.allclose(precision, np.linalg.inv(cov), rtol=1e-5, atol=1e-9)


def test_loss_inputs_agree_with_the_diagonal_form_when_u_is_zero():
    rng = np.random.default_rng(4)
    n = 32
    x = rng.normal(size=(n, 4)).astype("f4")
    xs = np.abs(rng.normal(scale=0.1, size=(n, 4))).astype("f4")
    xs[5, 3] = 1e9  # an unmeasured parameter, as everywhere else
    x_pred = x + rng.normal(scale=0.01, size=(n, 4)).astype("f4")
    c = np.zeros(n, dtype="i4")
    nu_cat = np.full(1, 25.0, dtype="f4")

    chi2, d, _ = profile.loss_inputs(tf.constant(x), tf.constant(xs),
                                     tf.constant(x_pred), tf.constant(c),
                                     tf.constant(nu_cat),
                                     tf.constant(_identity(n)))
    expected = (((x - x_pred) ** 2) / xs**2).sum(1)
    # The same arithmetic up to the order of the division and the square, so this
    # is float32 rounding (a few ulp) and not a difference in the model.
    assert np.allclose(chi2.numpy(), expected, rtol=1e-6)
    assert np.allclose(d.numpy(), (xs < 1e6).sum(1))


def test_normal_equations_reduce_to_the_diagonal_weights():
    rng = np.random.default_rng(5)
    n, d = 16, 4
    x = rng.normal(size=(n, d)).astype("f4")
    xs = np.abs(rng.normal(scale=0.1, size=(n, d))).astype("f4")
    x_pred = x + rng.normal(scale=0.01, size=(n, d)).astype("f4")
    jac = np.tile(np.eye(d), (n, 1, 1)).astype("f4")
    weight = np.full(n, 0.7, dtype="f4")
    star = np.arange(n, dtype="i8")

    A, b, uniq = profile.normal_equations(
        tf.constant(x), tf.constant(xs), tf.constant(x_pred), tf.constant(jac),
        tf.constant(weight), tf.constant(star), u=tf.constant(_identity(n)))
    w = weight[:, None] / xs**2
    A = np.asarray(A)
    # The matmuls run in TF32 on the GPU (10-bit mantissa), so the tolerance here
    # is ~1e-3 rather than float32's 1e-7; that is a property of the hardware path
    # the trainer already uses, not of the covariance algebra.
    assert np.allclose(np.diagonal(A, axis1=1, axis2=2), w, rtol=3e-3)
    off = A - np.eye(d)[None] * np.diagonal(A, axis1=1, axis2=2)[:, :, None]
    assert np.allclose(off, 0.0, atol=1e-3)
    assert np.allclose(np.asarray(b), weight[:, None] * (x - x_pred) / xs**2,
                       rtol=3e-3, atol=1e-3)
