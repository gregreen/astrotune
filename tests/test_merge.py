"""Tests for combining harmonized estimates (astrotune.merge) and applying a model."""

import numpy as np
import pytest

from astrotune import merge

MISSING = merge.MISSING_VAR
EPS = 0.03


def _est(index, ids, y, var, measured=None):
    """Estimates with diagonal covariances."""
    y = np.asarray(y, dtype="f8").reshape(len(ids), -1)
    n, d = y.shape
    var = np.broadcast_to(np.asarray(var, dtype="f8"), (n, d))
    cov = np.zeros((n, d, d))
    for p in range(d):
        cov[:, p, p] = var[:, p]
    meas = np.ones((n, d), dtype=bool) if measured is None else np.asarray(measured, dtype=bool)
    y = np.where(meas, y, 0.0)
    for p in range(d):
        cov[~meas[:, p], p, p] = MISSING
    return merge.Estimates(index, np.asarray(ids, dtype="i8"), y, cov, meas)


def test_two_entries_are_a_product_of_gaussians():
    a = _est(0, [7], [[0.0, 1.0, 2.0, 0.0]], 0.01)
    b = _est(1, [7], [[0.2, 1.0, 2.0, 0.1]], 0.04)
    m = merge.combine([a, b], chi2_max=1e9)
    va, vb = 0.01 + EPS**2, 0.04 + EPS**2  # the floor is part of the weights
    w = (1 / va) / (1 / va + 1 / vb)
    assert m.params[0, 0] == pytest.approx(w * 0.0 + (1 - w) * 0.2)
    assert m.cov[0, 0, 0] == pytest.approx(1 / (1 / va + 1 / vb))
    assert m.n_used[0] == 2
    assert m.params_source[0] == 0b11


def test_err_is_sqrt_variance_plus_floor_not_variance_squared():
    # Variance 4 (sigma 2): the notebook's sqrt(C^2 + e^2) gave ~4.0
    m = merge.combine([_est(0, [1], [[0, 0, 0, 0]], 4.0)])
    assert m.err[0, 0] == pytest.approx(np.sqrt(4.0 + EPS**2))
    m = merge.combine([_est(0, [1], [[0, 0, 0, 0]], 0.01)])
    assert m.err[0, 0] == pytest.approx(np.sqrt(0.01 + EPS**2))


def test_single_entry_is_copied_without_the_floor():
    m = merge.combine([_est(0, [1], [[0.5, -1, 3, 0.1]], 0.02)])
    assert np.allclose(m.params[0], [0.5, -1, 3, 0.1])
    assert m.cov[0, 0, 0] == pytest.approx(0.02)


def test_outlier_is_rejected():
    ids = [5]
    good = [_est(i, ids, [[0.0, 0.0, 0.0, 0.0]], 0.01) for i in range(3)]
    bad = _est(3, ids, [[3.0, 0.0, 0.0, 0.0]], 0.01)  # drags the first mean to 0.75
    m = merge.combine(good + [bad])
    assert m.n_used[0] == 3 and m.params_source[0] == 0b0111
    assert abs(m.params[0, 0]) < 1e-6
    assert not m.used[3][0] and m.used[0][0]
    assert m.chi2[3][0] > merge.CHI2_MAX


def test_star_left_with_one_entry_keeps_it():
    """The notebook returned zeros with zero covariance here."""
    # Entry 1 is far from entry 0 but its chi2 against the *combined* estimate is
    # large only if entry 0 is much tighter, so make entry 0 very precise.
    a = _est(0, [9], [[0.0, 0.0, 0.0, 0.0]], 1e-6)
    b = _est(1, [9], [[8.0, 0.0, 0.0, 0.0]], 1.0)
    m = merge.combine([a, b])
    assert m.n_used[0] == 1 and m.params_source[0] == 0b01
    assert np.allclose(m.params[0], 0.0, atol=1e-6)
    assert m.cov[0, 0, 0] == pytest.approx(1e-6)
    assert np.all(np.isfinite(m.lndet_cov))


def test_unmeasured_parameter_encoding_and_lndet():
    a = _est(0, [1, 2], [[0.1, 0, 1, 0.0]] * 2, 0.01, measured=[[1, 1, 1, 0]] * 2)
    b = _est(1, [1], [[0.1, 0, 1, 0.3]], 0.01)
    m = merge.combine([a, b])
    i1, i2 = 0, 1  # sorted ids 1, 2
    # star 1: alpha comes from b only
    assert m.measured[i1, 3] and m.params[i1, 3] == pytest.approx(0.3, abs=1e-3)
    # star 2: nobody measured alpha
    assert not m.measured[i2, 3]
    assert m.params[i2, 3] == 0.0 and m.cov[i2, 3, 3] == MISSING
    assert np.all(m.cov[i2, 3, :3] == 0.0)
    # ln det over the three measured parameters: no ln(1e9)
    assert m.lndet_cov[i2] == pytest.approx(3 * np.log(0.01))


def test_unmeasured_axis_does_not_move_the_others():
    a = _est(0, [1], [[0.0, 0.0, 0.0, 0.0]], 0.01, measured=[[1, 1, 1, 0]])
    b = _est(1, [1], [[0.2, 0.0, 0.0, 0.0]], 0.01, measured=[[1, 1, 1, 0]])
    m = merge.combine([a, b], chi2_max=1e9)
    assert m.params[0, 0] == pytest.approx(0.1)


def test_bit_mask_has_room_for_19_catalogs():
    ests = [_est(i, [1], [[0.0, 0.0, 0.0, 0.0]], 0.01) for i in (0, 16, 18)]
    m = merge.combine(ests)
    assert m.params_source.dtype == np.uint32
    assert m.params_source[0] == (1 | 1 << 16 | 1 << 18)
    assert merge.catalogs_in(m.params_source, 18)[0] and not merge.catalogs_in(m.params_source, 17)[0]


def test_per_entry_results_follow_the_input_order():
    a = _est(0, [3, 1, 2], np.zeros((3, 4)), 0.01)
    b = _est(1, [2, 3], np.zeros((2, 4)), 0.01)
    m = merge.combine([a, b])
    assert [len(x) for x in m.chi2] == [3, 2] and [len(x) for x in m.used] == [3, 2]
    assert list(m.ids) == [1, 2, 3]
    assert list(m.n_used) == [1, 2, 2]


def test_transform_propagates_covariance_with_the_jacobian():
    tf = pytest.importorskip("tensorflow")
    from astrotune import apply as ap
    from astrotune.model import ode_bijector  # noqa: F401  (ensures the package imports)

    A = np.array([[1.0, 0.5, 0, 0], [0, 2.0, 0, 0.3], [0.1, 0, 1.0, 0], [0, 0, 0.2, 1.5]], dtype="f4")
    shift = np.array([0.1, 0.2, 0.3, 0.4], dtype="f4")

    class Linear:
        def _forward(self, x, c=None):
            return tf.linalg.matvec(A, x) + shift[None] * tf.cast(c, tf.float32)[:, None]

    transform = ap.make_transform(Linear())
    from astropy.table import Table

    t = Table({
        "id": np.array([1, 2]),
        "x": np.array([[0.1, 0.2, 0.3, 0.4], [1.0, 1.0, 1.0, 0.0]], dtype="f4"),
        "sigma_x": np.array([[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 1e9]], dtype="f4"),
    })
    est = ap.transform_catalog(transform, t, index=2, batch_size=2)
    s = np.array([0.1, 0.2, 0.3, 0.4])
    expect = A @ np.diag(s**2) @ A.T
    assert np.allclose(est.cov[0], expect, rtol=5e-3)  # TF32 matmuls on A100
    assert np.allclose(est.y[0], A @ np.array([0.1, 0.2, 0.3, 0.4]) + 2 * shift, rtol=5e-3)
    # row 2: alpha unmeasured -> its 1e9 must not leak into the other covariances
    s2 = np.array([0.1, 0.2, 0.3, 0.0])
    expect2 = A @ np.diag(s2**2) @ A.T
    assert np.allclose(est.cov[1][:3, :3], expect2[:3, :3], rtol=5e-3)
    assert not est.measured[1, 3] and est.y[1, 3] == 0.0 and est.cov[1, 3, 3] == MISSING
    assert np.all(est.cov[1][3, :3] == 0.0)


def test_all_entries_failing_chi2_does_not_empty_the_star():
    """Three consistent entries and one wild one: the best entry always survives."""
    ids = [5]
    ests = [_est(i, ids, [[0.0, 0.0, 0.0, 0.0]], 0.01) for i in range(3)]
    ests.append(_est(3, ids, [[3.0, 0.0, 0.0, 0.0]], 0.01))
    for n_iter in (1, 2, 3, 4):
        m = merge.combine(ests, n_iter=n_iter)
        assert m.n_used[0] >= 1 and np.all(np.isfinite(m.params))


def test_chi2_max_used_flags_disagreeing_entries():
    a = _est(0, [1, 2], [[0.0] * 4, [0.0] * 4], 0.01)
    b = _est(1, [1, 2], [[0.0] * 4, [0.3, 0, 0, 0]], 0.01)
    m = merge.combine([a, b])  # tie: nobody can be rejected, but the star is flagged
    assert m.chi2_max_used[0] < 1e-9 and m.chi2_max_used[1] > 1.5


def test_transform_cache_round_trip_and_mismatch(tmp_path):
    pytest.importorskip("tensorflow")
    from astrotune import apply as ap

    e = _est(3, [10, 11, 12], np.arange(12).reshape(3, 4), 0.01, measured=[[1, 1, 1, 0]] * 3)
    ap.save_cached(tmp_path, e)
    got = ap.load_cached(tmp_path, 3, e.ids)
    assert np.array_equal(got.ids, e.ids) and np.allclose(got.y, e.y) and np.array_equal(got.measured, e.measured)
    assert ap.load_cached(tmp_path, 3, e.ids[::-1]) is None  # different row order: not reused
    assert ap.load_cached(tmp_path, 4, e.ids) is None


def test_transform_cache_is_discarded_when_the_table_changes(tmp_path):
    pytest.importorskip("tensorflow")
    from astropy.table import Table

    from astrotune import apply as ap

    t = Table({"id": np.array([10, 11, 12]), "x": np.zeros((3, 4), "f4"), "sigma_x": np.full((3, 4), 0.1, "f4")})
    e = _est(3, [10, 11, 12], np.arange(12).reshape(3, 4), 0.01)
    ap.save_cached(tmp_path, e, t)
    assert ap.load_cached(tmp_path, 3, e.ids, t) is not None
    t2 = t.copy()
    t2["sigma_x"][:, 3] = 0.05  # e.g. a different error floor, with the same ids
    assert ap.load_cached(tmp_path, 3, e.ids, t2) is None
    ap.save_cached(tmp_path, e)  # a cache made before the check existed
    assert ap.load_cached(tmp_path, 3, e.ids, t) is None
    assert ap.load_cached(tmp_path, 3, e.ids) is not None  # still readable without a table
