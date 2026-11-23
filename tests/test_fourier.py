"""Tests for the separable Fourier features of the spacetime coordinate."""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")


def test_the_ladders_are_separable_and_the_time_ladder_is_the_harmonics():
    """One log-spaced ladder per parameter axis; 1, 1/2, 1/4 ... along the flow."""
    from astrotune.model.fourier import FourierFeatures, mode_wavelengths, time_wavelengths

    assert np.allclose(time_wavelengths(1), [1.0])
    assert np.allclose(time_wavelengths(2), [1.0, 0.5])
    assert np.allclose(time_wavelengths(3), [1.0, 0.5, 0.25])
    assert time_wavelengths(0).size == 0

    # both ends exact, in between geometric
    lam = mode_wavelengths(5, 0.25, 8.0)
    assert lam.size == 5
    assert lam[0] == pytest.approx(8.0) and lam[-1] == pytest.approx(0.25)
    assert np.allclose(np.diff(np.log(lam)), -np.log(lam[0] / lam[-1]) / 4)
    assert np.allclose(mode_wavelengths(1, 0.25, 8.0), [8.0])  # one mode: the long end
    assert mode_wavelengths(0, 0.25, 8.0).size == 0

    # 5 columns = (t, 4 parameters) -> 2 harmonics on t, 5 modes on each of the rest
    f = FourierFeatures(n_in=5, n_modes=5, time_harmonics=2)
    assert f.n_waves == 2 + 4 * 5
    assert f.n_features == 2 * f.n_waves
    lam = f.wavelengths()
    axes = f.axes.numpy()
    assert np.allclose(lam[axes == 0], [1.0, 0.5])  # the flow axis is the harmonics
    for axis in range(1, 5):  # every other axis carries the full ladder
        assert np.allclose(lam[axes == axis], mode_wavelengths(5, 0.25, 8.0))
    # without a flow axis every column gets the ladder (the delta model's (x, ln s))
    assert FourierFeatures(n_in=8, n_modes=5, time_axis=None).n_waves == 8 * 5
    # and one ladder can be switched off on its own
    assert FourierFeatures(n_in=5, n_modes=0, time_harmonics=2).n_waves == 2


def test_each_mode_is_periodic_in_its_own_axis_only():
    from astrotune.model.fourier import FourierFeatures

    f = FourierFeatures(n_in=5, n_modes=1, lam_min=0.25, lam_max=8.0, time_harmonics=2)
    u = np.array([[0.37, -1.2, 0.4, 2.1, -0.3]], dtype="f4")
    base = f(tf.constant(u)).numpy()
    for axis in range(5):
        period = 1.0 if axis == 0 else 8.0  # the harmonics are 1 and 1/2 in t
        shifted = u.copy()
        shifted[0, axis] += period
        assert np.allclose(f(tf.constant(shifted)).numpy(), base, atol=1e-5)
        # a shift that is not a period has to change the block
        shifted[0, axis] = u[0, axis] + 0.37 * period
        assert not np.allclose(f(tf.constant(shifted)).numpy(), base, atol=1e-5)


def test_the_block_is_bounded_and_deterministic():
    """No seed, no training: the same input always gives the same features."""
    from astrotune.model.fourier import FourierFeatures

    f = FourierFeatures(n_in=5, n_modes=5, time_harmonics=2)
    u = np.random.default_rng(0).uniform(-3, 3, size=(5000, 5)).astype("f4")
    a = f(tf.constant(u)).numpy()
    assert a.shape == (5000, 44)
    assert np.abs(a).max() <= 1.0 / np.sqrt(f.n_waves) + 1e-6
    assert np.allclose(a, f(tf.constant(u)).numpy())
    # the longest-wavelength modes are nearly constant across a narrow data range
    narrow = np.zeros((1, 5), dtype="f4")
    long_mode = int(np.flatnonzero(np.isclose(f.wavelengths(), 8.0))[0])
    zero = f(tf.constant(narrow)).numpy()[0]
    assert abs(zero[long_mode]) == pytest.approx(f.scale)  # cos(0) = 1, scaled
    assert abs(zero[long_mode + f.n_waves]) < 1e-6  # sin(0) = 0


def test_the_network_widens_only_its_first_layer():
    from astrotune.model.vector_network import VectorNetwork

    plain = VectorNetwork(3, 4, 2, 8)
    plain(tf.zeros([2, 5]), tf.zeros([2], dtype=tf.int32))
    assert tuple(plain.hidden[0].w.shape) == (5, 8)  # unchanged
    assert plain.n_features == 0

    wide = VectorNetwork(3, 4, 2, 8,
                         fourier=dict(n_in=5, n_modes=5, time_harmonics=2))
    wide(tf.zeros([2, 5]), tf.zeros([2], dtype=tf.int32))  # the raw coordinate
    assert wide.n_features == 44
    assert tuple(wide.hidden[0].w.shape) == (5 + 44, 8)
    assert tuple(wide.hidden[1].w.shape) == (8, 8)  # later layers unchanged
    y = wide(tf.ones([3, 5]), tf.zeros([3], dtype=tf.int32))
    assert y.shape == (3, 4) and np.all(np.isfinite(y.numpy()))


def test_the_bijector_with_features_is_still_invertible():
    """Any velocity field gives a diffeomorphism; check this one numerically."""
    from astrotune.model.ode_bijector import ODEBijector

    b = ODEBijector(4, 3, 8, 2, fourier=dict(n_modes=3, time_harmonics=2))
    rng = np.random.default_rng(0)
    x = tf.constant(rng.normal(scale=0.5, size=(200, 4)), dtype=tf.float32)
    c = tf.constant(rng.integers(0, 3, size=200), dtype=tf.int32)
    y, _ = b.forward_with_s(x, c=c)
    back, _ = b.inverse_with_s(y, c=c)
    assert np.all(np.isfinite(back.numpy()))
    assert np.abs(back.numpy() - x.numpy()).max() < 1e-3
