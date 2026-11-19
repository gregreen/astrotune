"""Tests for the group hierarchy: catalogs from one pipeline share a systematic.

A group of several catalogs gets one extra variance per parameter, shared by the
entries of that group; a catalog on its own gets exactly zero. Two things have to
hold for the code to be right, and both are tested here: the shared part cancels
between two entries of the same group (so grouping two catalogs must *not* widen
their mutual disagreement), and it does not cancel across groups (so it does widen
the disagreement with everything else). The second is what makes it identifiable.
"""

import numpy as np
import pytest

from astrotune.delta import diagnostics as dg

tf = pytest.importorskip("tensorflow")

SIGMA = 0.03


def _groups_stars(n_star=2000, groups=(0, 0, 1), sigma=SIGMA, shared=None, seed=0,
                  spread=0.5):
    """Stars seen by every catalog, optionally with a systematic shared per group.

    ``groups[k]`` is the group of catalog ``k``. ``shared`` (one sigma per group) is
    drawn once per star and added to every entry of that group: the signature of two
    catalogs that ran the same pipeline and therefore make the same mistake.
    """
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups, dtype="i4")
    K = len(groups)
    n_group = int(groups.max()) + 1
    rep = np.exp(rng.normal(np.log(sigma), spread, size=(n_star, K)))
    z = rng.normal(size=(n_star, 1, 4))
    y = np.tile(z, (1, K, 1)) + rng.normal(size=(n_star, K, 4)) * rep[..., None]
    if shared is not None:
        # one offset per (star, group), reused by every catalog of that group
        u = rng.normal(size=(n_star, n_group, 4)) * np.asarray(shared, "f8")[None, :, None]
        y = y + u[:, groups]
    x = rng.normal(size=(n_star, K, 4))
    C = np.broadcast_to(np.eye(4), (n_star, K, 4, 4)) * (rep**2)[..., None, None]
    flat = dict(x=x, y=y, C=C, ls=np.broadcast_to(np.log(rep)[..., None],
                                                  (n_star, K, 4)),
                sig=np.broadcast_to(rep[..., None], (n_star, K, 4)),
                sig0=np.broadcast_to(rep[..., None], (n_star, K, 4)),
                J=np.broadcast_to(np.eye(4), (n_star, K, 4, 4)).copy())
    out = {k: np.ascontiguousarray(v.reshape(n_star * K, *v.shape[2:]).astype("f4"))
           for k, v in flat.items()}
    out["c"] = np.tile(np.arange(K, dtype="i4"), n_star)
    out["g"] = np.tile(groups, n_star)
    out["m"] = np.ones((n_star * K, 4), dtype=bool)
    out["id"] = np.arange(n_star, dtype="i8") + 1
    out["n_entry"] = np.full(n_star, K, dtype="i4")
    out["off"] = np.r_[0, np.cumsum(out["n_entry"])].astype("i8")
    out["seg"] = np.repeat(np.arange(n_star, dtype="i4"), K)
    return out


def _batch(s):
    from astrotune.delta.model import make_batch

    return make_batch(s, np.arange(int(s["off"][-1])), s["seg"])


def test_a_group_of_one_catalog_has_no_shared_variance_at_all():
    from astrotune.delta.model import ADD_MIN_SIGMA, SharedSystematic

    m = SharedSystematic(4, multiplicity=(2, 1, 1, 1))
    v = m.variance().numpy()
    assert m.has_shared()
    assert v[0, 0] > ADD_MIN_SIGMA**2        # the grouped one can grow off the floor
    assert np.all(v[1:] == 0.0)              # the others are exactly zero, not small
    alone = SharedSystematic(3, multiplicity=(1, 1, 1))
    assert not alone.has_shared()
    assert np.all(alone.variance().numpy() == 0.0)
    # the prior only acts on the groups that have a shared term
    assert float(alone.prior_penalty()) == 0.0


def test_the_two_level_likelihood_is_the_flat_one_when_shared_is_zero():
    """Grouping must be a no-op when tau = 0: same numbers, not merely close ones."""
    from astrotune.delta.model import fixed_delta, star_nll

    s = _groups_stars(n_star=60, groups=(0, 0, 1))
    batch = _batch(s)
    flat, _ = star_nll(batch, fixed_delta(0.0))
    two, _ = star_nll(batch, fixed_delta(0.0), n_group=2,
                      shared=tf.zeros([2, 4], tf.float32))
    assert np.allclose(two.numpy(), flat.numpy(), rtol=1e-10, atol=1e-10)


def test_the_shared_part_cancels_inside_a_group():
    """Two entries of one group disagree no more than their own errors allow."""
    tau = 0.03
    s = _groups_stars(groups=(0, 0), shared=(tau,), seed=1)
    plain = dg.pair_chi2(s)
    grouped = dg.pair_chi2(s, group_cov=np.array([[tau**2] * 4]))
    # the mean of chi2/dof is exactly 1 (the median is not: 3.36/4 for four dof)
    r_plain = np.mean(plain["chi2"] / plain["nu"])
    r_grouped = np.mean(grouped["chi2"] / grouped["nu"])
    assert r_plain == pytest.approx(1.0, abs=0.08)
    assert r_grouped == pytest.approx(r_plain, abs=0.02)  # cancels exactly


def test_the_shared_part_does_not_cancel_across_groups():
    """With another catalog present the difference is wider, and tau explains it."""
    tau = 0.03
    s = _groups_stars(groups=(0, 1), shared=(tau, 0.0), seed=2)
    plain = dg.pair_chi2(s)
    grouped = dg.pair_chi2(s, group_cov=np.array([[tau**2] * 4, [0.0] * 4]))
    ratio_plain = np.mean(plain["chi2"] / plain["nu"])
    ratio_grouped = np.mean(grouped["chi2"] / grouped["nu"])
    # two quoted sigmas of 0.03 plus one shared tau: the difference is 1.5x wider
    assert ratio_plain == pytest.approx(1.5, abs=0.12)
    assert ratio_grouped == pytest.approx(1.0, abs=0.08)


def test_a_shared_systematic_is_recovered_by_the_constant_model():
    """Three catalogs in two groups: the shared tau of the pair is identifiable."""
    from astrotune.delta.model import ConstScaleAdd
    from astrotune.delta.train import DeltaConfig, fit

    tau = 0.02
    s = _groups_stars(n_star=20000, groups=(0, 0, 1), shared=(tau, 0.0))
    cfg = DeltaConfig(batch_size=16384, prior_weight=0.0, l2_weights=0.0, n_control=0,
                      learn_nu=False)
    m = ConstScaleAdd(2, multiplicity=(2, 1))
    fit(m, s, cfg, epochs=80, lr=0.1, desc="test")
    got = np.sqrt(m.shared_variance().numpy()[:, 0])  # Teff column, as usual
    assert got[1] == 0.0  # the single-catalog group stays at exactly zero
    assert got[0] == pytest.approx(tau, rel=0.35), got
    # ... and the entries' own errors are left alone: nothing else was wrong
    assert np.allclose(m.scale_sigma()[:, 0], 1.0, atol=0.12), m.scale_sigma()[:, 0]


def test_the_group_index_puts_the_aspcap_catalogs_together():
    from astrotune import registry

    keys = registry.keys()
    g = registry.group_index(keys)
    i, j = keys.index("apogee_dr17"), keys.index("sdssv_aspcap")
    assert g[i] == g[j]
    assert len(set(g)) == len(keys) - 1  # one pair, everything else on its own
    assert registry.group_multiplicity(keys)[g[i]] == 2
    label = registry.group_labels(keys)[g[i]]
    assert "APOGEE" in label and "+" in label
