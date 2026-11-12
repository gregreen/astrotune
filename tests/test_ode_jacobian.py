"""Tests for the Jacobian transported alongside the ODE bijection.

The Jacobian solves ``dM/dt = J_v M`` through the same integration as the map, so
it must agree with ``d x(1)/d x(0)``. Two kinds of reference are used:

* the derivative of the *velocity field* itself, compared to a finite difference
  of ``conditional_velocity`` -- no ODE involved, so this isolates the network
  Jacobian and is the test that catches a broken gradient path;
* the derivative of the *map*, compared to a finite difference of the plain
  inverse. A network velocity is a stiff float32 field, and its finite
  difference is limited by round-off at the ``1e-4`` level (the error falls as
  ``~1e-4 / eps`` for large ``eps``), so the map comparison against the network
  is loose on purpose. The sharp map comparisons use an analytic velocity field
  integrated in float64, where a constant linear field also has the exact
  answer ``exp(B)``.
"""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
pytest.importorskip("sonnet")
pytest.importorskip("tensorflow_probability")
scipy_linalg = pytest.importorskip("scipy.linalg")

from astrotune.model.ode_bijector import ODEBijector  # noqa: E402

N_CAT = 5
N_DIM = 4


def _bijector(seed=0, output_scale=0.4, tol=1e-5):
    """A bijector whose velocity is well away from zero, so J is not ~I."""
    tf.random.set_seed(seed)
    return ODEBijector(
        n_dim=N_DIM,
        n_cond=N_CAT,
        hidden_size=8,
        n_hidden=2,
        rtol=tol,
        atol=tol,
        vector_network_kwargs=dict(output_scale=output_scale),
    )


def _sample(n=5, seed=1, dtype="f4"):
    rng = np.random.default_rng(seed)
    y = rng.normal(size=(n, N_DIM)).astype(dtype)
    c = rng.integers(0, N_CAT, size=n).astype("i4")
    return y, c


def _patch_velocity(bij, field):
    """Replace the velocity by ``field``, which ignores the network."""
    bij.conditional_velocity = field
    return bij


def _linear_field(b, dtype="f4"):
    b = tf.constant(np.asarray(b, dtype=dtype))

    def field(t, x, c=None):
        return tf.linalg.matmul(x, b, transpose_b=True)

    return field


def _nonlinear_field(a, dtype="f8"):
    """A smooth, time-dependent, non-conservative field: ``tanh(A x) (1 + t/2)``."""
    a = tf.constant(np.asarray(a, dtype=dtype))

    def field(t, x, c=None):
        return tf.math.tanh(tf.linalg.matmul(x, a, transpose_b=True)) * (1 + 0.5 * t)

    return field


def _finite_difference(fn, v, eps, columns=None):
    """``d fn(v)/d v`` by central differences, one column at a time."""
    jac = np.zeros((len(v), N_DIM, N_DIM))
    for j in (range(N_DIM) if columns is None else columns):
        delta = np.zeros_like(v)
        delta[:, j] = eps
        jac[:, :, j] = (fn(v + delta) - fn(v - delta)) / (2 * eps)
    return jac


@pytest.mark.filterwarnings("ignore")
def test_velocity_jacobian_matches_finite_differences_of_the_velocity():
    """The inner Jacobian: d v / d x of the network itself, no ODE involved.

    The tolerance is loose because the network is float32: its velocity carries
    ~7e-4 of noise, so a finite difference cannot do better than ``7e-4 / eps``
    (~1e-2 at ``eps = 5e-2``, and worse for smaller steps). The sharp checks of
    the same code path are the analytic-field tests below.
    """
    bij = _bijector()
    x, c = _sample()
    x, c = tf.constant(x), tf.constant(c)
    t = 0.3

    v, jac = bij.velocity_and_jacobian(t, x, c)
    # only a sanity check that the returned velocity is the plain one
    assert np.allclose(v.numpy(), bij.conditional_velocity(t, x, c).numpy(), atol=1e-4)

    jac = jac.numpy()
    assert jac.shape == (len(x), N_DIM, N_DIM)
    assert not np.allclose(jac, 0.0), "the velocity Jacobian must not be all zeros"

    fd = _finite_difference(
        lambda xx: bij.conditional_velocity(t, tf.constant(xx), c).numpy(), x.numpy(), 5e-2
    )
    assert np.allclose(jac, fd, rtol=5e-2, atol=5e-2), np.abs(jac - fd).max()


@pytest.mark.filterwarnings("ignore")
def test_velocity_jacobian_is_not_transposed():
    """A wrong convention, ``d x_j / d v_i``, would otherwise pass unnoticed."""
    bij = _bijector()
    x, c = _sample(n=2)
    x, c = tf.constant(x), tf.constant(c)

    _, jac = bij.velocity_and_jacobian(0.7, x, c)
    jac = jac.numpy()
    fd = _finite_difference(
        lambda xx: bij.conditional_velocity(0.7, tf.constant(xx), c).numpy(), x.numpy(), 1e-2
    )
    assert not np.allclose(jac, np.swapaxes(fd, 1, 2), rtol=2e-2, atol=2e-2)


@pytest.mark.filterwarnings("ignore")
def test_transported_jacobian_matches_the_map_for_a_nonlinear_field():
    """Sharp map check: analytic field, float64 integration, tight tolerance."""
    rng = np.random.default_rng(2)
    bij = _patch_velocity(
        _bijector(tol=1e-10), _nonlinear_field(0.7 * rng.normal(size=(N_DIM, N_DIM)))
    )
    y, c = _sample(seed=3, dtype="f8")
    y, c = tf.constant(y), tf.constant(c)

    _, _, jac = bij.inverse_with_s_and_jac(y, c=c)
    fd = _finite_difference(
        lambda vv: bij._inverse(tf.constant(vv), c=c).numpy(), y.numpy(), 1e-6
    )
    assert np.allclose(jac.numpy(), fd, rtol=1e-6, atol=1e-6), np.abs(jac.numpy() - fd).max()

    _, _, jac_f = bij.forward_with_s_and_jac(y, c=c)
    fd_f = _finite_difference(
        lambda vv: bij._forward(tf.constant(vv), c=c).numpy(), y.numpy(), 1e-6
    )
    assert np.allclose(jac_f.numpy(), fd_f, rtol=1e-6, atol=1e-6), np.abs(jac_f.numpy() - fd_f).max()


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("scale", [0.0, 0.5])
def test_linear_velocity_field_gives_the_exponential_jacobian(scale):
    """With ``v = B x`` the map is exactly ``exp(B)``, so ``d x/d y = exp(-B)``."""
    rng = np.random.default_rng(4)
    b = 0.5 * scale * rng.normal(size=(N_DIM, N_DIM))

    bij = _patch_velocity(_bijector(tol=1e-9), _linear_field(b, "f8"))
    y, c = _sample(seed=5, dtype="f8")
    c = tf.constant(c)

    x, s, jac = bij.inverse_with_s_and_jac(tf.constant(y), c=c)
    expected = np.broadcast_to(scipy_linalg.expm(-b), (len(y), N_DIM, N_DIM))
    assert np.allclose(jac.numpy(), expected, atol=1e-8), np.abs(jac.numpy() - expected).max()

    if scale == 0.0:  # a zero field must map every point to itself
        assert np.allclose(x.numpy(), y, atol=1e-10)
        assert np.allclose(s.numpy(), 0.0, atol=1e-10)

    _, _, jac_f = bij.forward_with_s_and_jac(tf.constant(y), c=c)
    expected = np.broadcast_to(scipy_linalg.expm(b), (len(y), N_DIM, N_DIM))
    assert np.allclose(jac_f.numpy(), expected, atol=1e-8)


@pytest.mark.filterwarnings("ignore")
def test_jacobian_variant_reproduces_the_plain_map():
    """Adding the Jacobian must not change the map or the path length.

    The tolerance is the solver's own: the augmented state has 21 components
    instead of 5, so it takes a slightly different sequence of steps and the two
    integrations agree to their accuracy, not exactly.
    """
    bij = _bijector()
    y, c = _sample(seed=6)
    y, c = tf.constant(y), tf.constant(c)

    x_ref, s_ref = bij.inverse_with_s(y, c=c)
    x, s, _ = bij.inverse_with_s_and_jac(y, c=c)
    assert np.allclose(x.numpy(), x_ref.numpy(), atol=2e-3)
    assert np.allclose(s.numpy(), s_ref.numpy(), atol=2e-3)

    out_ref, s_out = bij.forward_with_s(y, c=c)
    out, s, _ = bij.forward_with_s_and_jac(y, c=c)
    assert np.allclose(out.numpy(), out_ref.numpy(), atol=2e-3)
    assert np.allclose(s.numpy(), s_out.numpy(), atol=2e-3)


@pytest.mark.filterwarnings("ignore")
def test_forward_and_inverse_jacobians_cancel():
    """``d y/d x`` times ``d x/d y`` is the identity, as it must be."""
    bij = _bijector()
    y, c = _sample(seed=7)
    y, c = tf.constant(y), tf.constant(c)

    x, _, jac_i = bij.inverse_with_s_and_jac(y, c=c)
    _, _, jac_f = bij.forward_with_s_and_jac(x, c=c)

    product = np.matmul(jac_f.numpy(), jac_i.numpy())
    identity = np.broadcast_to(np.eye(N_DIM), product.shape)
    assert np.allclose(product, identity, atol=1e-3), np.abs(product - identity).max()


@pytest.mark.filterwarnings("ignore")
def test_map_jacobian_matches_autodiff_through_the_solver():
    """End-to-end check on the real network against a different method.

    ``batch_jacobian`` differentiates the solver itself, so it is an independent
    estimate of ``d x/d y``: no finite difference and no ``1e-4 / eps`` noise.
    The two agree to ~2e-3 here, which is the accuracy of the solver rather than
    of either Jacobian.
    """
    bij = _bijector()
    y, c = _sample(seed=8)
    y, c = tf.constant(y), tf.constant(c)

    _, _, jac = bij.inverse_with_s_and_jac(y, c=c)

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(y)
        x = bij._inverse(y, c=c)
    reference = tape.batch_jacobian(x, y, experimental_use_pfor=False).numpy()
    del tape

    assert np.allclose(jac.numpy(), reference, rtol=1e-2, atol=1e-2), np.abs(
        jac.numpy() - reference
    ).max()
