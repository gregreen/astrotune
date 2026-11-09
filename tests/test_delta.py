"""Tests for the post-hoc delta model (pair table, diagnostics, likelihood, recovery)."""

import numpy as np
import pytest
from scipy import stats

from astrotune.delta import diagnostics as dg
from astrotune.delta import pairs as pr


def _synthetic_pairs(n_per=6000, sigma_true=(0.02, 0.05, 0.1), c_reported=0.03, seed=0, n_cat=3):
    """Pairs for three catalogs. Each reports C = c_reported^2 I but has an extra
    variance sigma_true[c]^2 (the same for all four parameters)."""
    rng = np.random.default_rng(seed)
    out = {k: [] for k in pr.FIELDS}
    for j in range(n_cat):
        for k in range(j + 1, n_cat):
            n = n_per
            y0 = rng.normal(size=(n, 4)).astype("f4")
            tot = lambda c: np.sqrt(c_reported**2 + sigma_true[c] ** 2)
            ya = y0 + rng.normal(size=(n, 4)) * tot(j)
            yb = y0 + rng.normal(size=(n, 4)) * tot(k)
            C = np.broadcast_to(np.eye(4) * c_reported**2, (n, 4, 4)).astype("f4")
            out["id"].append(rng.integers(1, 10**12, n))
            out["ca"].append(np.full(n, j, "i4"))
            out["cb"].append(np.full(n, k, "i4"))
            for s, yy in (("a", ya), ("b", yb)):
                out["x" + s].append(y0.astype("f4"))
                out["ls" + s].append(np.full((n, 4), np.log(c_reported), "f4"))
                out["y" + s].append(yy.astype("f4"))
                out["C" + s].append(C)
            out["m"].append(np.ones((n, 4), bool))
    return {k: np.concatenate(v) for k, v in out.items()}


def test_split_is_by_star_and_about_one_eighth():
    ids = np.random.default_rng(0).integers(1, 10**15, 200000)
    v = pr.val_mask(ids)
    assert 0.11 < v.mean() < 0.14
    assert np.array_equal(v, pr.val_mask(ids))
    p = {"id": np.r_[ids, ids]}
    assert np.array_equal(pr.val_mask(p["id"])[:len(ids)], pr.val_mask(p["id"])[len(ids):])


def test_diagnostics_are_calibrated_when_the_covariance_is_right():
    pairs = _synthetic_pairs(sigma_true=(0, 0, 0))
    res = dg.diagnose(pairs, 3, None, min_pairs=1000)
    p = res["pooled"]
    assert abs(p["median_pit"] - 0.5) < 0.02
    assert all(abs(r - 1.0) < 0.05 for r in p["scale_ratio"])
    assert p["frac_sf"][0] == pytest.approx(0.05, abs=0.01)


def test_three_cornered_hat_recovers_the_excess():
    pairs = _synthetic_pairs(sigma_true=(0.02, 0.05, 0.1))
    res = dg.diagnose(pairs, 3, None, min_pairs=1000)
    sig = np.sqrt(np.abs(res["excess_variance"]))
    assert np.allclose(sig[:, 0], [0.02, 0.05, 0.1], atol=0.01)
    assert np.all(np.nan_to_num(res["excess_variance"]) > -1e-4)  # sampling noise only


def test_diagnostics_handle_unmeasured_parameters():
    pairs = _synthetic_pairs(sigma_true=(0, 0, 0), n_per=2000)
    pairs["m"][:, 3] = False
    pairs["ya"][:, 3] = pairs["yb"][:, 3] = 0
    pairs["Ca"][:, 3, 3] = pairs["Cb"][:, 3, 3] = 1e9
    st = dg.pair_stats(pairs)
    assert np.all(np.isnan(st["z"][:, 3])) and np.all(st["dof"] == 3)
    assert abs(np.median(st["pit"]) - 0.5) < 0.03


tf = pytest.importorskip("tensorflow")


def _batch(pairs, n=64):
    return {k: tf.constant(pairs[k][:n]) for k in ("xa", "xb", "lsa", "lsb", "ca", "cb", "ya", "yb", "Ca", "Cb", "m")}


def test_pair_nll_matches_scipy_with_masking_and_delta():
    from astrotune.delta.model import fixed_delta, pair_nll

    rng = np.random.default_rng(3)
    pairs = _synthetic_pairs(n_per=40, sigma_true=(0.0, 0.0, 0.0))
    n = len(pairs["id"])
    A = rng.normal(size=(n, 4, 4)) * 0.01
    for s in "ab":  # full, correlated covariances
        pairs["C" + s] = (np.einsum("nij,nkj->nik", A, A) + np.eye(4) * 1e-3).astype("f4")
    pairs["m"][::3, 2] = False  # logg unmeasured by one catalog
    for s in "ab":
        pairs["C" + s][::3, 2, :] = 0
        pairs["C" + s][::3, :, 2] = 0
        pairs["C" + s][::3, 2, 2] = 1e9
    b = _batch(pairs, 60)
    nll, _, _ = pair_nll(b, fixed_delta(0.05))
    for i in range(60):
        m = pairs["m"][i]
        S = (pairs["Ca"][i].astype("f8") + pairs["Cb"][i].astype("f8") + np.eye(4) * 2 * 0.05**2)[np.ix_(m, m)]
        d = (pairs["ya"][i] - pairs["yb"][i]).astype("f8")[m]
        assert nll[i].numpy() == pytest.approx(-stats.multivariate_normal(np.zeros(m.sum()), S).logpdf(d), rel=1e-5)


def test_floor_is_smooth_and_bounded_below():
    from astrotune.delta.model import LN_MIN, floor_ln_delta

    raw = tf.constant([-100.0, LN_MIN, -7.0, 0.0])
    out = floor_ln_delta(raw).numpy()
    assert out[0] == pytest.approx(LN_MIN, abs=1e-6) and np.all(out >= LN_MIN)
    assert out[2] == pytest.approx(-7.0, abs=0.02) and out[3] == pytest.approx(0.0, abs=1e-4)


def test_constant_model_recovers_the_true_extra_variance():
    """The three-catalog 'hat' is identifiable through the likelihood too."""
    from astrotune.delta.model import ConstDelta
    from astrotune.delta.train import DeltaConfig, fit

    pairs = _synthetic_pairs(sigma_true=(0.02, 0.05, 0.1), n_per=20000)
    cfg = DeltaConfig(batch_size=4096, prior_weight=0.0, l2_weights=0.0)
    m = ConstDelta(3)
    fit(m, pairs, cfg, epochs=60, lr=0.2, desc="test")  # flat split direction: many steps
    sig = np.exp(0.5 * m.raw.numpy())  # (3, 4)
    assert np.allclose(sig.mean(1), [0.02, 0.05, 0.1], atol=0.012), sig


def test_network_runs_and_saves_and_loads(tmp_path):
    from astrotune.delta.model import DeltaNetwork, load_delta_model, numpy_delta_fn, save_delta_model

    net = DeltaNetwork(3, hidden_size=16, n_hidden=2)
    pairs = _synthetic_pairs(n_per=50)
    fn = numpy_delta_fn(net)
    out = fn(pairs["xa"], pairs["lsa"], pairs["ca"])
    assert out.shape == (len(pairs["id"]), 4) and np.all(np.isfinite(out))
    save_delta_model(tmp_path, net, dict(n_cat=3, hidden_size=16, n_hidden=2))
    net2 = load_delta_model(tmp_path)
    assert np.allclose(numpy_delta_fn(net2)(pairs["xa"], pairs["lsa"], pairs["ca"]), out, atol=1e-6)
