"""Tests for the post-hoc delta model (star table, diagnostics, likelihood, recovery)."""

import numpy as np
import pytest
from scipy import stats

from astrotune.delta import diagnostics as dg
from astrotune.delta import stars as st


def _synthetic_stars(n_star=4000, sigma_true=(0.02, 0.05, 0.1), c_reported=0.03,
                     seed=0, n_cat=3, k=None):
    """Stars with ``k`` entries (default: one in every catalog).

    Every catalog reports ``C = c_reported^2 I`` and carries an extra variance
    ``sigma_true[c]^2`` (the same for the four parameters), so the correct
    calibration is ``exp(ln_delta) = sigma_true[c]^2``.
    """
    rng = np.random.default_rng(seed)
    n, K = n_star, (n_cat if k is None else k)
    if K == n_cat:
        cats = np.tile(np.arange(K, dtype="i4"), n)
    else:  # distinct random catalogs per star
        cats = np.argsort(rng.random((n, n_cat)), axis=1)[:, :K].astype("i4").ravel()
    z = rng.normal(size=(n, 1, 4))
    total = np.sqrt(c_reported**2 + np.asarray(sigma_true, "f8") ** 2)[cats].reshape(n, K, 1)
    y = np.tile(z, (1, K, 1)) + rng.normal(size=(n, K, 4)) * total
    flat = {
        "x": rng.normal(size=(n, K, 4)),
        "ls": np.full((n, K, 4), np.log(c_reported)),
        "c": cats[:, None],
        "y": y,
        "C": np.broadcast_to(np.eye(4) * c_reported**2, (n, K, 4, 4)),
        "m": np.ones((n, K, 4), dtype=bool),
    }
    out = {k: np.ascontiguousarray(v.reshape(n * K, *v.shape[2:]).astype("f4"))
           for k, v in flat.items() if k != "m"}
    out["m"] = np.ascontiguousarray(flat["m"].reshape(n * K, 4))
    out["c"] = out["c"].astype("i4")
    out["id"] = np.arange(n, dtype="i8") * 7 + 1
    out["n_entry"] = np.full(n, K, dtype="i4")
    out["off"] = np.r_[0, np.cumsum(out["n_entry"])].astype("i8")
    out["seg"] = np.repeat(np.arange(n, dtype="i4"), K)
    return out


def test_split_is_by_star_and_about_one_eighth():
    s = _synthetic_stars(n_star=20000, n_cat=2)
    v = st.val_mask(s["id"])
    assert 0.11 < v.mean() < 0.14
    assert np.array_equal(v, st.val_mask(s["id"]))
    train, val = st.split(s)
    assert len(train["id"]) + len(val["id"]) == len(s["id"])
    assert not set(train["id"]) & set(val["id"])
    assert int(train["off"][-1]) + int(val["off"][-1]) == int(s["off"][-1])
    assert np.unique(train["seg"]).size == len(train["id"])


def test_blocks_cover_every_entry_once_and_keep_stars_whole():
    s = _synthetic_stars(n_star=1000, n_cat=3)
    seen, n_total = [], 0
    for idx, seg, n in st.block_indices(s, None, target_entries=512):
        assert len(idx) == len(seg)
        assert np.all(np.diff(seg) >= 0)
        assert np.array_equal(np.unique(seg), np.arange(n))  # every star has entries here
        seen.append(idx)
        n_total += n
    idx = np.concatenate(seen)
    assert n_total == len(s["id"])
    assert np.array_equal(np.sort(idx), np.arange(int(s["off"][-1])))


def test_diagnostics_are_calibrated_when_the_covariance_is_right():
    s = _synthetic_stars(sigma_true=(0, 0, 0))
    res = dg.diagnose(s, 3, None, min_entries=1000)
    p = res["pooled"]
    assert abs(p["median_pit"] - 0.5) < 0.03
    assert all(abs(r - 1.0) < 0.05 for r in p["scale_ratio"])
    assert p["frac_sf"][0] == pytest.approx(0.05, abs=0.015)


def test_diagnostics_fill_every_star_across_blocks():
    """Every block numbers its stars from 0, so the per-star arrays must use the table's
    index; a single block hides the bug (each block used to overwrite the last)."""
    s = _synthetic_stars(sigma_true=(0, 0, 0), n_star=2000, n_cat=3)
    res = dg.entry_stats(s, chunk=600)  # 2000 stars x 3 entries -> 10 blocks
    assert res["star_dof"].min() > 0  # no star left at the initial zero
    assert res["star_chi2"].min() > 0
    pit = stats.chi2.cdf(res["star_chi2"], res["star_dof"])
    assert pit.shape == (2000,)
    assert abs(np.median(pit) - 0.5) < 0.04


def test_excess_variance_recovers_the_per_catalog_extra():
    s = _synthetic_stars(sigma_true=(0.02, 0.05, 0.1), n_star=8000, k=2)
    res = dg.diagnose(s, 3, None, min_entries=1000)
    sig = np.sqrt(np.clip(res["excess_variance"], 0.0, None))
    assert np.allclose(sig[:, 0], [0.02, 0.05, 0.1], atol=0.012)
    assert np.all(np.nan_to_num(res["excess_variance"]) > -1e-3)  # sampling noise only


def test_excess_variance_uses_only_the_two_catalog_stars():
    """With three entries the naive residual variance is biased, so it is skipped."""
    s = _synthetic_stars(sigma_true=(0.02, 0.05, 0.1), n_star=8000, k=3)
    exc = dg.three_cornered_hat(s, 3, min_pairs=1000)
    assert np.all(np.isnan(exc))


def test_diagnostics_handle_unmeasured_parameters():
    s = _synthetic_stars(sigma_true=(0, 0, 0), n_star=2000)
    entry = np.arange(2, 3 * 2000, 3)  # the third entry of every star: no [a/Fe]
    s["m"][entry, 3] = False
    s["y"][entry, 3] = 0.0
    s["C"][entry, 3, :] = 0.0
    s["C"][entry, :, 3] = 0.0
    s["C"][entry, 3, 3] = 1e9
    res = dg.entry_stats(s)
    assert np.all(np.isnan(res["z"][entry, 3]))
    assert np.all(res["dof"][entry] == 3)
    # 4 + 4 + 3 measured parameters, less the one parameter that all three share
    assert np.all(res["star_dof"] == 4 + 4 + 3 - 4)
    pit = stats.chi2.cdf(res["star_chi2"], res["star_dof"])
    assert abs(np.median(pit) - 0.5) < 0.04


tf = pytest.importorskip("tensorflow")


def _batch(s):
    from astrotune.delta.model import make_batch

    return make_batch(s, np.arange(int(s["off"][-1])), s["seg"])


def test_star_nll_for_two_catalogs_is_the_pair_likelihood():
    """K = 2 is the case the old pair likelihood covered: it must agree exactly."""
    from astrotune.delta.model import fixed_delta, star_nll

    s = _synthetic_stars(n_star=40, sigma_true=(0.0, 0.0, 0.0), n_cat=2)
    nll, _ = star_nll(_batch(s), fixed_delta(0.05))
    for i in range(40):
        S = s["C"][2 * i].astype("f8") + s["C"][2 * i + 1].astype("f8") + np.eye(4) * 2 * 0.05**2
        d = (s["y"][2 * i] - s["y"][2 * i + 1]).astype("f8")
        ref = -stats.multivariate_normal(np.zeros(4), S).logpdf(d)
        assert float(nll[i].numpy()) == pytest.approx(ref, rel=1e-6)


def test_star_nll_matches_the_difference_space_density():
    """K >= 3: the marginal is a Gaussian density in the ``K - 1`` differences.

    With the star's value integrated out, the likelihood depends on the entries only
    through ``u_k = y_k - y_K``, which are jointly Gaussian with
    ``Cov(u_a, u_b) = S_a [a == b] + S_K``. That is a proper density of dimension
    ``n_res = (K - 1) n_params`` -- exactly the dimension the constant of
    :func:`star_nll` counts -- and scipy can evaluate it, so this checks the
    quadratic form, both determinants and the normalisation at once.
    """
    from astrotune.delta.model import fixed_delta, star_nll

    K = 4
    s = _synthetic_stars(n_star=6, sigma_true=(0.02, 0.05, 0.1, 0.07), c_reported=0.05, n_cat=K)
    rng = np.random.default_rng(7)  # correlated covariances, so the test is not diagonal
    A = rng.normal(size=(len(s["c"]), 4, 4)) * 0.01
    s["C"] = (np.einsum("nij,nkj->nik", A, A) + np.eye(4) * 0.0025).astype("f4")
    nll, _ = star_nll(_batch(s), fixed_delta(0.0))
    for i in range(6):
        S = [s["C"][K * i + k].astype("f8") for k in range(K)]
        y = [s["y"][K * i + k].astype("f8") for k in range(K)]
        u = np.concatenate([y[k] - y[-1] for k in range(K - 1)])
        W = np.zeros(((K - 1) * 4,) * 2)
        for a in range(K - 1):
            for b in range(K - 1):
                W[a * 4:(a + 1) * 4, b * 4:(b + 1) * 4] = S[-1] + (S[a] if a == b else 0)
        ref = -stats.multivariate_normal(np.zeros(len(u)), W).logpdf(u)
        assert float(nll[i].numpy()) == pytest.approx(ref, rel=1e-9, abs=1e-9)


def test_three_catalog_star_nll_is_not_the_product_of_its_pairs():
    """The sum over observations replaces three correlated pairwise terms."""
    from astrotune.delta.model import fixed_delta, star_nll

    s = _synthetic_stars(n_star=1, sigma_true=(0.0, 0.0, 0.0), n_cat=3)
    nll, _ = star_nll(_batch(s), fixed_delta(0.0))
    pair_sum = 0.0
    for a, b in ((0, 1), (0, 2), (1, 2)):
        S = s["C"][a].astype("f8") + s["C"][b].astype("f8")
        d = (s["y"][a] - s["y"][b]).astype("f8")
        pair_sum += -stats.multivariate_normal(np.zeros(4), S).logpdf(d)
    assert abs(float(nll[0].numpy()) - pair_sum) > 1e-6


def test_floor_is_smooth_and_bounded_below():
    from astrotune.delta.model import LN_MIN, floor_ln_delta

    raw = tf.constant([-100.0, LN_MIN, -7.0, 0.0])
    out = floor_ln_delta(raw).numpy()
    assert out[0] == pytest.approx(LN_MIN, abs=1e-6) and np.all(out >= LN_MIN)
    assert out[2] == pytest.approx(-7.0, abs=0.02) and out[3] == pytest.approx(0.0, abs=1e-4)


def test_constant_model_recovers_the_true_extra_variance():
    """The three-catalog 'hat' is identifiable through the star likelihood too."""
    from astrotune.delta.model import ConstDelta
    from astrotune.delta.train import DeltaConfig, fit

    s = _synthetic_stars(sigma_true=(0.02, 0.05, 0.1), n_star=20000)
    cfg = DeltaConfig(batch_size=8192, prior_weight=0.0, l2_weights=0.0)
    m = ConstDelta(3)
    fit(m, s, cfg, epochs=40, lr=0.2, desc="test")  # flat split direction: many steps
    sig = np.exp(0.5 * m.raw.numpy())  # (3, 4)
    assert np.allclose(sig[:, 0], [0.02, 0.05, 0.1], atol=0.015)


def test_prior_is_applied_once_per_entry():
    """The pair version charged the prior K-1 times for the same entry."""
    from astrotune.delta.model import PRIOR_MEAN, prior_penalty

    s = _synthetic_stars(n_star=5, n_cat=3)
    la = tf.constant(np.full((15, 4), PRIOR_MEAN + 3.0), tf.float32)
    pen = float(prior_penalty(_batch(s), la).numpy())
    assert pen == pytest.approx(0.5, rel=1e-5)  # one sigma^2 per measured parameter


def test_delta_model_refuses_a_different_catalog_list(tmp_path):
    from astrotune.delta.model import DeltaNetwork, load_delta_model, save_delta_model

    net = DeltaNetwork(3)
    save_delta_model(tmp_path, net, dict(n_cat=3, hidden_size=64, n_hidden=3,
                                         keys=["a", "b", "c"], config={}))
    load_delta_model(tmp_path, keys=["a", "b", "c"])
    with pytest.raises(ValueError):
        load_delta_model(tmp_path, keys=["a", "b", "d"])


def test_build_star_table_groups_entries_by_star(tmp_path, monkeypatch):
    """The assembly step: sort by star, drop single-catalog stars, keep catalog order."""
    import json

    import astrotune.apply as ap
    import astrotune.training_data as td
    from astrotune import registry
    from astrotune.merge import Estimates

    keys = list(registry.keys()[:3])
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "config.json").write_text(json.dumps({"catalogs": keys}))
    cache = tmp_path / "cache"
    cache.mkdir()

    # star 1 is in all three catalogs, star 2 in catalogs 0 and 2, stars 3 and 9 in one
    per_catalog = {0: np.array([1, 2, 9]), 1: np.array([1, 3]), 2: np.array([1, 2])}

    def fake_tables(keys_arg=None, dedupe="none"):
        from astropy.table import Table

        return [
            Table({"id": per_catalog[i], "x": np.zeros((len(per_catalog[i]), 4), "f4"),
                   "sigma_x": np.ones((len(per_catalog[i]), 4), "f4")})
            for i in range(len(keys))
        ]

    def fake_cached(cache_dir, index, ids, table=None):
        n = len(ids)
        y = np.full((n, 4), index + 1.0, "f4")  # tagged with the catalog index
        cov = np.broadcast_to(np.eye(4, dtype="f4"), (n, 4, 4)).copy()
        return Estimates(index, ids, y, cov, np.ones((n, 4), bool))

    monkeypatch.setattr(td, "load_tables", fake_tables)
    monkeypatch.setattr(ap, "load_cached", fake_cached)

    stars, out_keys = st.build_star_table(tmp_path / "run", cache)
    assert tuple(out_keys) == tuple(keys)
    assert np.array_equal(stars["id"], [1, 2])  # the single-catalog stars are dropped
    assert np.array_equal(stars["n_entry"], [3, 2])
    assert np.array_equal(stars["off"], [0, 3, 5])
    assert np.array_equal(stars["seg"], [0, 0, 0, 1, 1])
    assert np.array_equal(stars["c"], [0, 1, 2, 0, 2])  # star-major, catalog ascending
    assert np.allclose(stars["y"][:, 0], [1, 2, 3, 1, 3])  # rows follow their catalog
