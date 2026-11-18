"""Tests for the Student's-t likelihood and the per-catalog tail weight."""

import numpy as np
import pytest
from scipy import stats

from astrotune.delta import diagnostics as dg

tf = pytest.importorskip("tensorflow")


def _stars(n_star=200, n_cat=3, sigma=(0.02, 0.05, 0.1), c_reported=0.03, seed=0,
           outlier_frac=0.0, outlier_scale=8.0):
    """Stars seen by every catalog, optionally with a fraction of 'bad' stars.

    A bad star has all of its entries inflated by ``outlier_scale`` -- the signature
    of a binary or a mismatched id, which is what a per-star shared tail weight is
    meant to absorb.
    """
    rng = np.random.default_rng(seed)
    n, K = n_star, n_cat
    z = rng.normal(size=(n, 1, 4))
    total = np.sqrt(c_reported**2 + np.asarray(sigma, "f8") ** 2)[None, :, None]
    noise = rng.normal(size=(n, K, 4)) * total
    bad = rng.random(n) < outlier_frac
    noise[bad] *= outlier_scale
    y = np.tile(z, (1, K, 1)) + noise
    cats = np.tile(np.arange(K, dtype="i4"), n)
    flat = dict(
        x=rng.normal(size=(n, K, 4)),
        ls=np.full((n, K, 4), np.log(c_reported)),
        c=cats[:, None],
        y=y,
        C=np.broadcast_to(np.eye(4) * c_reported**2, (n, K, 4, 4)),
    )
    out = {k: np.ascontiguousarray(v.reshape(n * K, *v.shape[2:]).astype("f4"))
           for k, v in flat.items()}
    out["m"] = np.ones((n * K, 4), dtype=bool)
    out["c"] = out["c"].astype("i4")
    out["id"] = np.arange(n, dtype="i8") + 1
    out["n_entry"] = np.full(n, K, dtype="i4")
    out["off"] = np.r_[0, np.cumsum(out["n_entry"])].astype("i8")
    out["seg"] = np.repeat(np.arange(n, dtype="i4"), K)
    return out


def _batch(s):
    from astrotune.delta.model import make_batch

    return make_batch(s, np.arange(int(s["off"][-1])), s["seg"])


def test_a_huge_tail_weight_is_the_gaussian_likelihood():
    """nu -> infinity must give back the Gaussian numbers, not merely similar ones."""
    from astrotune.delta.model import fixed_delta, star_nll

    s = _stars(n_star=40)
    batch = _batch(s)
    gauss, _ = star_nll(batch, fixed_delta(0.02))
    heavy, _ = star_nll(batch, fixed_delta(0.02), nu=tf.fill([3], 1e9))
    assert np.allclose(heavy.numpy(), gauss.numpy(), rtol=1e-5, atol=1e-6)


def test_the_two_catalog_marginal_is_a_multivariate_t():
    """K = 2: the difference is t_nu with scale S_1 + S_2, exactly as scipy has it."""
    from astrotune.delta.model import fixed_delta, star_nll

    nu = 5.0
    s = _stars(n_star=12, n_cat=2, sigma=(0, 0), c_reported=0.03)
    rng = np.random.default_rng(5)  # correlated covariances, so nothing is diagonal
    A = rng.normal(size=(len(s["c"]), 4, 4)) * 0.01
    s["C"] = (np.einsum("nij,nkj->nik", A, A) + np.eye(4) * 0.0025).astype("f4")
    nll, _ = star_nll(_batch(s), fixed_delta(0.0), nu=tf.fill([2], nu))
    for i in range(12):
        W = s["C"][2 * i].astype("f8") + s["C"][2 * i + 1].astype("f8")
        d = (s["y"][2 * i] - s["y"][2 * i + 1]).astype("f8")
        ref = -stats.multivariate_t(np.zeros(4), W, df=nu).logpdf(d)
        # scipy and this formula differ at the 1e-8 level (float32 C in, different
        # algebra for the same quantity), which is far below any physical meaning.
        assert float(nll[i].numpy()) == pytest.approx(ref, rel=1e-6, abs=1e-6)


def test_the_star_tail_weight_is_the_mean_over_its_catalogs():
    from astrotune.delta.diagnostics import _star_nu

    s = _stars(n_star=3, n_cat=3)
    nu = np.array([4.0, 6.0, 11.0])
    assert np.allclose(_star_nu(s, nu), 7.0)  # every star sees all three
    s["c"][0] = 1  # star 0 now sees catalogs 1, 1, 2 (edge case: repetition cannot
    # happen in a real table, but the mean must still be well defined)
    assert _star_nu(s, nu)[0] == pytest.approx((6 + 6 + 11) / 3)
    assert np.all(np.isinf(_star_nu(s, None)))


def test_tail_weight_starts_at_the_prior_and_is_penalised_away_from_it():
    from astrotune.delta.model import NU_INIT_RAW, NU_MIN, NU_PRIOR_MEAN, TailWeight

    t = TailWeight(4)
    assert np.allclose(t().numpy(), NU_MIN + np.log1p(np.exp(NU_INIT_RAW)), rtol=1e-4)
    assert float(t.prior_penalty(NU_PRIOR_MEAN, 2.0)) == pytest.approx(0.0, abs=1e-6)
    assert float(t.prior_penalty(NU_PRIOR_MEAN + 2.0, 2.0)) == pytest.approx(0.5, rel=1e-5)
    t.raw.assign(tf.fill([4], -50.0))  # theta -> -inf: nu -> 2 from above
    assert np.allclose(t().numpy(), NU_MIN, atol=1e-3)


def test_a_learnt_tail_weight_absorbs_outliers_without_inflating_the_core():
    """The point of the exercise: heavy tails instead of a larger scale."""
    from astrotune.delta.model import ConstScaleAdd
    from astrotune.delta.train import DeltaConfig, fit

    s = _stars(n_star=20000, outlier_frac=0.15, outlier_scale=8.0)
    base = DeltaConfig(batch_size=16384, prior_weight=0.0, l2_weights=0.0, n_control=0,
                       nu_prior_weight=0.0, const_epochs=40)
    t = ConstScaleAdd(3)
    fit(t, s, base, epochs=40, lr=0.1, desc="t")
    gauss = ConstScaleAdd(3)
    fit(gauss, s, DeltaConfig(**{**base.__dict__, "learn_nu": False}), epochs=40, lr=0.1,
        desc="gauss")
    nu = t.nu().numpy()
    # The tail weight must have *moved*: reading it outside the gradient tape leaves it
    # at its initial value, which is exactly the bug this asserts against.
    assert np.all(np.abs(nu - 8.0) > 0.5), nu
    assert np.all(nu < 15.0), nu  # the t is preferred over Gaussian tails
    assert np.all(t.scale_sigma() < gauss.scale_sigma() * 0.8)  # and the scale is smaller
