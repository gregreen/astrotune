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
        "sig": np.full((n, K, 4), c_reported),
        "sig0": np.full((n, K, 4), c_reported),
        "J": np.broadcast_to(np.eye(4), (n, K, 4, 4)).copy(),
        "c": cats[:, None],
        "y": y,
        "C": np.broadcast_to(np.eye(4) * c_reported**2, (n, K, 4, 4)),
        "m": np.ones((n, K, 4), dtype=bool),
    }
    out = {k: np.ascontiguousarray(v.reshape(n * K, *v.shape[2:]).astype("f4"))
           for k, v in flat.items() if k != "m"}
    out["m"] = np.ascontiguousarray(flat["m"].reshape(n * K, 4))
    out["c"] = out["c"].astype("i4")
    out["g"] = out["c"].copy()  # one group per catalog unless a test says otherwise
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
    s["sig"][entry, 3] = 0.0
    s["sig0"][entry, 3] = 0.0
    res = dg.entry_stats(s)
    assert np.all(np.isnan(res["z"][entry, 3]))
    assert np.all(res["dof"][entry] == 3)
    # 4 + 4 + 3 measured parameters, less the one parameter that all three share
    assert np.all(res["star_dof"] == 4 + 4 + 3 - 4)
    pit = stats.chi2.cdf(res["star_chi2"], res["star_dof"])
    assert abs(np.median(pit) - 0.5) < 0.04


def test_pair_chi2_is_calibrated_when_the_covariance_is_right():
    s = _synthetic_stars(n_star=4000, sigma_true=(0, 0, 0), n_cat=3)
    r = dg.pair_chi2(s, None)
    assert r["chi2"].size == 3 * 4000  # 3 pairs per three-entry star
    assert np.all(r["nu"] == 4)  # every entry measures all four parameters
    assert np.median(r["chi2"]) == pytest.approx(stats.chi2.median(4), rel=0.06)


def test_pair_chi2_rejects_a_quoted_error_that_is_half_the_truth():
    """Every catalog reports sigma but really scatters by 2 sigma: chi2 quadruples."""
    from astrotune.delta.model import fixed_delta

    extra = np.sqrt(3.0) * 0.03  # sqrt(0.03^2 + extra^2) = 2 x 0.03
    s = _synthetic_stars(n_star=4000, sigma_true=(extra, extra), n_cat=2)
    bad = dg.pair_chi2(s, None)
    assert np.median(bad["chi2"]) == pytest.approx(4 * stats.chi2.median(4), rel=0.10)
    # the additive term that fixes it exactly (as a numpy callable)
    from astrotune.delta.model import fixed_delta

    inner = fixed_delta(extra)

    def good_fn(x, ln_sigma, c):
        a, b = inner(tf.constant(x), tf.constant(ln_sigma), tf.constant(c, tf.int32))
        return a.numpy(), b.numpy()

    good = dg.pair_chi2(s, good_fn)
    assert np.median(good["chi2"]) == pytest.approx(stats.chi2.median(4), rel=0.08)


def test_pair_chi2_counts_only_the_parameters_both_entries_share():
    s = _synthetic_stars(sigma_true=(0, 0, 0), n_star=200, n_cat=2)
    s["m"][1::2, 3] = False  # every second entry has no [alpha/Fe]
    s["sig"][1::2, 3] = 0.0
    s["sig0"][1::2, 3] = 0.0
    r = dg.pair_chi2(s, None)
    assert np.all(r["nu"] == 3)
    assert np.median(r["chi2"]) == pytest.approx(stats.chi2.median(3), rel=0.12)


def test_the_ideal_curve_is_a_normalised_mixture():
    from astrotune.delta.diagnostics import chi2_mixture_pdf

    nu = np.array([3] * 3 + [4] * 7)
    x = np.linspace(0, 60, 3001)
    assert np.trapz(chi2_mixture_pdf(x, nu), x) == pytest.approx(1.0, rel=2e-3)
    # 70 % of the pairs share all four parameters, so the mixture follows chi2_4 there
    assert chi2_mixture_pdf(np.array([3.357]), nu)[0] == pytest.approx(
        0.7 * stats.chi2.pdf(3.357, 4) + 0.3 * stats.chi2.pdf(3.357, 3), rel=1e-6)


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
    # J diag(sig^2) J^T is the covariance: J carries the correlations, sig the size
    s["C"] = (np.einsum("nij,nkj->nik", A, A) + np.eye(4) * 0.0025).astype("f4")
    s["J"] = (np.linalg.cholesky(s["C"].astype("f8")) / 0.05).astype("f4")
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
        assert float(nll[i].numpy()) == pytest.approx(ref, rel=1e-6, abs=1e-6)


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
    cfg = DeltaConfig(batch_size=8192, prior_weight=0.0, l2_weights=0.0, n_control=0)
    m = ConstDelta(3)
    fit(m, s, cfg, epochs=40, lr=0.2, desc="test")  # flat split direction: many steps
    sig = np.exp(0.5 * m.raw.numpy())  # (3, 4)
    assert np.allclose(sig[:, 0], [0.02, 0.05, 0.1], atol=0.015)


def test_prior_is_applied_once_per_entry():
    """The pair version charged the prior K-1 times for the same entry."""
    from astrotune.delta.model import LN_MIN, PRIOR_ADD_SD, PRIOR_SCALE_SD, prior_penalty

    s = _synthetic_stars(n_star=5, n_cat=3)
    zero = tf.zeros((15, 4), tf.float32)
    one_sigma_up = tf.fill((15, 4), LN_MIN + float(PRIOR_ADD_SD))
    pen = float(prior_penalty(_batch(s), (zero, one_sigma_up)).numpy())
    assert pen == pytest.approx(0.5, rel=1e-5)  # one z^2, once per measured parameter
    pen_scale = float(prior_penalty(_batch(s), (tf.fill((15, 4), 2.0 * PRIOR_SCALE_SD),
                                              tf.fill((15, 4), LN_MIN))).numpy())
    assert pen_scale == pytest.approx(2.0, rel=1e-5)  # (2 sigma)^2 / 2, add term at rest
    at_rest = float(prior_penalty(_batch(s), (zero, tf.fill((15, 4), LN_MIN))).numpy())
    assert at_rest == pytest.approx(0.0, abs=1e-6)  # scale 1 and add on the floor


def test_delta_model_refuses_a_different_catalog_list(tmp_path):
    from astrotune.delta.model import ScaleAddNetwork, load_delta_model, save_delta_model

    net = ScaleAddNetwork(3)
    save_delta_model(tmp_path, net, dict(n_cat=3, hidden_size=64, n_hidden=3,
                                         keys=["a", "b", "c"], model_type="scale_add",
                                         config={}))
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

    def fake_tables(keys_arg=None, dedupe="none", **kwargs):
        from astropy.table import Table

        return [
            Table({"id": per_catalog[i], "x": np.zeros((len(per_catalog[i]), 4), "f4"),
                   "sigma_x": np.ones((len(per_catalog[i]), 4), "f4"),
                   "sigma_x_raw": np.ones((len(per_catalog[i]), 4), "f4")})
            for i in range(len(keys))
        ]

    def fake_jac(bij):
        def f(xx, cc):
            n = int(len(xx))
            # tag the harmonized value with the catalog index, as the real table would
            # carry the catalog's own value there
            y = np.repeat(np.asarray(cc, "f4")[:, None] + 1.0, 4, axis=1)
            return y, np.broadcast_to(np.eye(4, dtype="f4"), (n, 4, 4)).copy()

        return f

    monkeypatch.setattr(td, "load_tables", fake_tables)
    monkeypatch.setattr(ap, "make_forward_and_jacobian", fake_jac)
    # the real bijector loader needs a trained model; the fake ignores it
    monkeypatch.setattr("astrotune.model.load_bijector", lambda *a, **k: None)
    # and the transform cache must not be consulted at all: the star table is a
    # function of the prepared tables and the bijector (see _catalog_entries)
    monkeypatch.setattr(
        ap, "load_cached",
        lambda *a, **k: pytest.fail("build_star_table must not read apply's cache"),
    )

    stars, out_keys = st.build_star_table(tmp_path / "run", cache)
    assert tuple(out_keys) == tuple(keys)
    assert np.array_equal(stars["id"], [1, 2])  # the single-catalog stars are dropped
    assert np.array_equal(stars["n_entry"], [3, 2])
    assert np.array_equal(stars["off"], [0, 3, 5])
    assert np.array_equal(stars["seg"], [0, 0, 0, 1, 1])
    assert np.array_equal(stars["c"], [0, 1, 2, 0, 2])  # star-major, catalog ascending
    assert np.allclose(stars["y"][:, 0], [1, 2, 3, 1, 3])  # rows follow their catalog


def test_training_history_plot_survives_partial_histories(tmp_path):
    """The three fits record different things; the figure must handle each."""
    from astrotune.delta.train import plot_training_history

    # a const arm: no nu, no shared term, no validation set
    const = [dict(epoch=e, train_nll=1.0 / e, loss=1.0 / e, prior=0.0, l2=0.0,
                  nu_prior=0.0, shared_prior=0.0, control=0.0, grad_norm=10.0 / e)
             for e in range(1, 4)]
    # a full arm: everything, including the per-group nu vector
    net = [dict(epoch=e, train_nll=1.0 / e, val_nll=1.1 / e, loss=1.2 / e,
                data=1.0 / e, prior=0.1 / e, l2=0.1 / e, nu_prior=0.01 / e,
                shared_prior=0.001 / e, control=0.0001 / e, grad_norm=5.0 / e,
                nu=[5.0 + e, 8.0 - e], nu_median=6.5, nu_min=5.0 + e,
                nu_max=8.0 - e, shared_sigma_median=1e-3 / e,
                shared_sigma_max=2e-3 / e)
           for e in range(1, 4)]
    path = plot_training_history({"const add": const, "network": net},
                                 tmp_path / "history.png", title="test")
    assert path.exists() and path.stat().st_size > 10000
    # an empty history is not an error either (a fit that was skipped)
    assert plot_training_history({}, tmp_path / "empty.png") == tmp_path / "empty.png"


def test_nu_prior_density_is_normalised(tmp_path):
    """p(nu) is the prior on raw moved to nu, so it must integrate to one."""
    from astrotune.delta.train import nu_prior_curve, plot_nu_prior

    # A geometric grid: the Jacobian diverges at nu = 2, so uniform sampling there
    # would both miss the mass and cost a fortune.
    nu = 2.0 + np.logspace(-12, 4, 400_000)
    for mean, sd in ((6.0, 2.0), (14.0, 6.0), (30.0, 16.0)):
        dens = nu_prior_curve(nu, mean, sd)
        assert np.all(dens >= 0)
        assert np.isclose(np.trapz(dens, nu), 1.0, atol=5e-3)
        # the far tail has to die away, or the curve would not be plottable
        assert dens[-1] < 1e-8 * dens.max()

        # The nu-space prior is the same statement made directly: a Gaussian cut off
        # at 2, so it stays small there instead of piling up at the boundary.
        flat = np.linspace(2.0, 60.0, 400_000)
        gauss = nu_prior_curve(flat, 8.0, 2.0, "nu")
        assert gauss[0] < 0.05 * gauss.max()
        assert np.isclose(np.trapz(gauss, flat), 1.0, atol=5e-3)
        # ... whereas the raw-space prior rises towards nu = 2, above its own peak
        raw_near_two = nu_prior_curve(np.array([2.001, 2.01, 2.1]), 14.0, 6.0)
        interior = nu_prior_curve(flat[flat > 6.0], 14.0, 6.0).max()
        assert raw_near_two[0] > interior
        assert raw_near_two[0] > raw_near_two[-1]

    keys = ("apogee_dr17", "sdssv_aspcap", "galah_dr4")
    nu_values = np.array([7.0, 7.0, 9.5])  # the first two share a group
    path = plot_nu_prior(nu_values, keys, 8.0, 2.0, tmp_path / "nu_prior.png", space="nu")
    assert path.exists() and path.stat().st_size > 10000


def test_the_volume_change_is_reported_per_parameter(tmp_path):
    """A catalog reporting three parameters must not look smaller than one with four.

    Same scale, same additive floor, so the only difference between the catalogs is
    how many parameters they measure. Divided by that number, the two agree.
    """
    from astrotune.delta.model import ADD_MIN_SIGMA, ConstScaleAdd
    from astrotune.delta.train import catalog_summary, plot_nu_det_c

    s = _synthetic_stars(n_star=500, n_cat=3)
    s["m"] = np.array(s["m"], copy=True)
    s["m"][s["c"] == 2, 3] = False  # catalog 2 does not measure the fourth parameter
    out = catalog_summary(ConstScaleAdd(3), s, ("a", "b", "c"))
    assert np.isfinite(out["det"]).all()
    assert np.allclose(out["det"][2], out["det"][0], rtol=1e-3)
    # and the change is the floor's, per parameter: ln(1 + add^2 / C)
    expected = np.log1p(float(ADD_MIN_SIGMA) ** 2 / 0.03 ** 2)
    assert np.allclose(out["det"], expected, rtol=0.05)

    path = plot_nu_det_c(out, tmp_path / "nu_det_c.png")
    assert path.exists() and path.stat().st_size > 10000


def test_label_positions_leave_the_dots_and_each_other_clear():
    """The layout of the point labels: no overlaps, leaders on the box edges."""
    from astrotune.delta.train import _label_positions

    pts = np.array([[0.5, 0.5], [0.7, 0.4], [1.4, 1.1], [1.8, 0.6]])
    sizes = np.tile([0.9, 0.12], (len(pts), 1))  # 11 characters or so, at 6.5pt
    centres, ends = _label_positions(pts, sizes, (0.0, 2.5, 0.0, 1.5))
    assert np.all(np.isfinite(centres)) and np.all(np.isfinite(ends))
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            # boxes are disjoint in at least one direction
            dx = abs(centres[i, 0] - centres[j, 0]) - (sizes[i, 0] + sizes[j, 0]) / 2
            dy = abs(centres[i, 1] - centres[j, 1]) - (sizes[i, 1] + sizes[j, 1]) / 2
            assert dx > 0 or dy > 0
        # no label sits on any dot
        for p in pts:
            assert (abs(p[0] - centres[i, 0]) > sizes[i, 0] / 2
                    or abs(p[1] - centres[i, 1]) > sizes[i, 1] / 2)
        # the leader stops on the label's boundary
        assert np.allclose(ends[i], np.clip(pts[i],
                                            centres[i] - sizes[i] / 2,
                                            centres[i] + sizes[i] / 2))


def test_nu_prior_in_nu_space_is_quadratic_in_nu():
    """The penalty is a Gaussian in nu, not in the unconstrained coordinate."""
    from astrotune.delta.model import TailWeight
    from astrotune.delta.train import DeltaConfig

    cfg = DeltaConfig(nu_prior_space="nu", nu_prior_mean=16.0, nu_prior_sd=4.0)
    assert cfg.nu_prior_space == "nu"
    t = TailWeight(3)
    t.raw.assign(np.array([np.log(np.expm1(16.0 - 2.0))] * 3, dtype="f4"))  # nu = 16
    at_mean = float(t.prior_penalty(cfg.nu_prior_mean, cfg.nu_prior_sd, "nu"))
    assert at_mean == pytest.approx(0.0, abs=1e-5)
    assert float(t().numpy()[0]) == pytest.approx(16.0, rel=1e-4)
    # one prior sd away in nu is a penalty of 0.5, whatever nu it is
    t.raw.assign(np.array([np.log(np.expm1(12.0 - 2.0))] * 3, dtype="f4"))  # nu = 12
    assert float(t.prior_penalty(16.0, 4.0, "nu")) == pytest.approx(0.5, rel=1e-3)
    with pytest.raises(ValueError):
        t.prior_penalty(16.0, 4.0, "nope")


def test_a_fit_starts_its_tail_weight_at_the_prior_mean():
    """Not at the arbitrary nu = 8 TailWeight is built with, whatever the prior."""
    from astrotune.delta.model import TailWeight
    from astrotune.delta.train import DeltaConfig, nu_init

    for space, mean, expected in (("nu", 25.0, 25.0), ("nu", 4.0, 4.0),
                                  ("raw", 6.0, 8.0), ("raw", 30.0, 32.0)):
        nu0, raw0 = nu_init(DeltaConfig(nu_prior_space=space, nu_prior_mean=mean))
        # in raw space the prior mean is a softplus away from the mean of nu
        assert nu0 == pytest.approx(expected, rel=1e-3)
        t = TailWeight(2)
        t.raw.assign(np.full(2, raw0, dtype="f4"))
        # the raw value is the exact inverse of nu = 2 + softplus(raw)
        assert np.allclose(t().numpy(), nu0, rtol=1e-5)
        # and it is the prior's own minimum, so the fit starts unpenalised
        assert float(t.prior_penalty(mean, 3.0, space)) == pytest.approx(0.0, abs=1e-6)


