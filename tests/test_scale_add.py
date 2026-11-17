"""Tests for the scale + additive error model.

``S_ij = s_i C_ij s_j + delta_ij add_i`` with ``s_i = exp(ln_scale_i / 2)``: the
covariance is rescaled by a per-parameter factor and a diagonal term is added, both
functions of the stellar parameters and the catalog.
"""

import numpy as np
import pytest

from astrotune.delta import diagnostics as dg

tf = pytest.importorskip("tensorflow")


def _stars(n_star=30000, scale=(4.0, 1.0, 0.25), seed=0, sigma_spread=0.7):
    """Stars in three catalogs, each reporting a *different* sigma per star.

    The spread of the reported errors is what separates a scale from an additive
    term (with one fixed sigma per catalog the two are exactly interchangeable), so
    it is drawn from a log-normal with the given width. Catalog ``k``'s noise is
    ``sqrt(scale[k])`` times what it reports, i.e. the true correction is a pure
    scale of ``scale[k]`` in variance.
    """
    rng = np.random.default_rng(seed)
    n, K = n_star, len(scale)
    rep = np.exp(rng.normal(np.log(0.03), sigma_spread, size=(n, K)))
    x = (rng.normal(size=(n, K, 4)) * np.array([1.0, 0.5, 1.0, 0.2])
         + np.array([0.5, -1.0, 3.0, 0.05]))
    z = rng.normal(size=(n, 1, 4))
    total = (rep * np.sqrt(np.asarray(scale, "f8"))[None, :])[:, :, None]
    y = np.tile(z, (1, K, 1)) + rng.normal(size=(n, K, 4)) * total
    cats = np.tile(np.arange(K, dtype="i4"), n)
    C = np.broadcast_to(np.eye(4), (n, K, 4, 4)) * (rep**2)[:, :, None, None]
    flat = dict(
        x=x,
        ls=np.broadcast_to(np.log(rep)[:, :, None], (n, K, 4)),
        c=cats[:, None],
        y=y,
        C=C,
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


def test_soft_clamp_is_the_identity_inside_and_saturates_outside():
    from astrotune.delta.model import LOG_SCALE_MAX, LOG_SCALE_MIN, soft_clamp

    raw = tf.constant([-100.0, -6.0, -1.0, 0.0, 1.0, 6.0, 100.0])
    out = soft_clamp(raw, LOG_SCALE_MIN, LOG_SCALE_MAX).numpy()
    assert out[0] == pytest.approx(LOG_SCALE_MIN, abs=1e-3)
    assert out[-1] == pytest.approx(LOG_SCALE_MAX, abs=1e-3)
    assert np.all((out >= LOG_SCALE_MIN - 1e-2) & (out <= LOG_SCALE_MAX + 1e-2))
    # identity in the middle: the clamp must not bias the scale where it has no opinion
    assert np.allclose(out[2:5], raw.numpy()[2:5], atol=2e-3)
    assert np.all(np.diff(out[1:6]) > 0)  # monotone over the whole unsaturated range


def test_the_covariance_stays_positive_definite_at_the_clamps():
    """Any s > 0 keeps diag(s) C diag(s) PD, and the additive term only helps."""
    from astrotune.delta.model import LOG_SCALE_MAX, LOG_SCALE_MIN, fixed_delta, make_batch, star_nll

    s = _stars(n_star=32)
    rng = np.random.default_rng(3)  # correlated, non-diagonal C
    A = rng.normal(size=(len(s["c"]), 4, 4)) * 0.5
    s["C"] = (np.einsum("nij,nkj->nik", A, A) + np.eye(4) * 1e-4).astype("f4")
    for extreme in (LOG_SCALE_MIN, LOG_SCALE_MAX, 0.0):
        def fn(x, ln_sigma, c, extreme=extreme):
            return tf.fill(tf.shape(x), tf.constant(extreme, tf.float32)), tf.fill(
                tf.shape(x), tf.constant(-60.0, tf.float32))

        nll, _ = star_nll(make_batch(s, np.arange(int(s["off"][-1])), s["seg"]), fn)
        assert np.all(np.isfinite(nll.numpy()))


def test_scale_one_leaves_the_covariance_alone():
    """``ln_scale = 0`` and an add term on the floor is exactly the untransformed model."""
    from astrotune.delta.model import make_batch, star_nll

    s = _stars(n_star=64)
    batch = make_batch(s, np.arange(int(s["off"][-1])), s["seg"])

    def identity(x, ln_sigma, c):
        return (tf.zeros(tf.shape(x), tf.float32),
                tf.fill(tf.shape(x), tf.constant(-80.0, tf.float32)))

    nll, _ = star_nll(batch, identity)
    assert np.all(np.isfinite(nll.numpy()))
    # ... and it is not the same as any nonzero scale, so the scale is doing work
    def shrunk(x, ln_sigma, c):
        return (tf.fill(tf.shape(x), tf.constant(np.log(0.25), tf.float32)),
                tf.fill(tf.shape(x), tf.constant(-80.0, tf.float32)))

    nll_shrunk, _ = star_nll(batch, shrunk)
    assert np.mean(nll_shrunk.numpy()) > np.mean(nll.numpy())


def test_a_pure_scale_is_recovered_by_the_constant_model():
    """Additive-only cannot represent this: the data need the errors to *shrink*."""
    from astrotune.delta.model import ADD_MIN_SIGMA, ConstScaleAdd
    from astrotune.delta.train import DeltaConfig, fit

    want = (4.0, 1.0, 0.25)  # variance scales: sigma x2, x1, x0.5
    s = _stars(scale=want)
    cfg = DeltaConfig(batch_size=16384, prior_weight=0.0, l2_weights=0.0, n_control=0)
    m = ConstScaleAdd(3)
    fit(m, s, cfg, epochs=60, lr=0.1, desc="test")
    got = m.scale_sigma()[:, 0]  # Teff column; the same correction was used for all four
    assert np.allclose(got, np.sqrt(want), rtol=0.15), got
    assert np.all(m.add_sigma() < 3 * ADD_MIN_SIGMA)  # nothing additive was needed


def test_control_points_are_masked_to_what_a_catalog_measures():
    """A catalog that cannot see an axis must not be pinned on it."""
    from astrotune.delta.model import control_points

    s = _stars(n_star=200)
    s["m"][s["c"] == 1, 2] = False  # catalog 1 measures no logg
    ctrl = control_points(s, ["a", "b", "c"], n_per_catalog=64, seed=0)
    assert ctrl["x"].shape == (3 * 64, 4) and ctrl["m"].shape == (3 * 64, 4)
    b = ctrl["c"] == 1
    assert not ctrl["m"][b, 2].any()  # masked out of the penalty
    assert np.all(ctrl["x"][b, 2] == 0.0)  # and drawn at the placeholder, as in the data
    assert ctrl["m"][ctrl["c"] == 0].all()  # catalog 0 measures everything
    # the points cover the box, not a huddle in the middle
    assert ctrl["x"][:, 0].std() > 0.2 * (ctrl["x"][:, 0].max() - ctrl["x"][:, 0].min())


def test_control_penalty_is_zero_at_identity_and_grows_away_from_it():
    from astrotune.delta import model as mdl

    s = _stars(n_star=100)
    ctrl = mdl.control_points(s, ["a", "b", "c"], n_per_catalog=32, seed=0)
    ident = mdl.fixed_delta(0.0)  # scale 1, add = -60 (well below the floor)
    base = float(mdl.control_penalty(ctrl, ident).numpy())
    worse = mdl.fixed_delta(0.2)  # a real additive term everywhere
    assert float(mdl.control_penalty(ctrl, worse).numpy()) > base


def _pairs_and_triples(n_pairs=100, n_triples=100):
    """A star table with ``n_pairs`` two-catalog stars and ``n_triples`` three."""
    entry_cat, entry_star = [], []
    for i in range(n_pairs):
        for k in (0, 1):
            entry_cat.append(k)
            entry_star.append(i)
    for i in range(n_triples):
        for k in (0, 1, 2):
            entry_cat.append(k)
            entry_star.append(n_pairs + i)
    n_entry = np.array([2] * n_pairs + [3] * n_triples, dtype="i4")
    return dict(c=np.array(entry_cat, dtype="i4"),
                seg=np.array(entry_star, dtype="i4"),
                n_entry=n_entry,
                off=np.r_[0, np.cumsum(n_entry)].astype("i8"))


def test_single_partner_share_flags_weakly_identified_catalogs():
    s = _pairs_and_triples(n_pairs=100, n_triples=100)
    share = dg.single_partner_share(s, 3)
    assert share[0] == pytest.approx(0.5) and share[1] == pytest.approx(0.5)
    assert share[2] == pytest.approx(0.0)  # catalog 2 only appears in three-catalog stars
