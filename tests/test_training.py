"""Tests for the training-data builder and the training step."""

import numpy as np
import pytest
from astropy.table import Table

from astrotune import training_data as td


def _table(ids, d, seed):
    rng = np.random.default_rng(seed)
    n = len(ids)
    return Table(
        {
            "id": np.asarray(ids, dtype="i8"),
            "x": rng.normal(size=(n, d)).astype("f4"),
            "sigma_x": np.full((n, d), 0.1, dtype="f4"),
            "x_err": np.full((n, d), 0.1, dtype="f4"),
        }
    )


def test_extract_observations_pads_missing_params():
    t = _table([1, 2], 3, 0)
    obs = td.extract_observations(t, 2, np.array([1, 2]))
    assert obs["x"].shape == (2, 4)
    assert np.all(obs["x"][:, 3] == 0)
    assert np.all(obs["xs"][:, 3] == 1e9)
    assert np.all(obs["xe"][:, 3] == 1e9)
    assert list(obs["id"]) == [1, 2] and list(obs["c"]) == [2, 2]
    assert np.array_equal(obs["x"][:, :3], np.asarray(t["x"]))


def test_observations_keep_only_stars_in_two_catalogs():
    # 2 appears twice, but only in catalog 0: it carries no cross-calibration
    t0 = _table([1, 2, 2, 3], 4, 0)
    t1 = _table([3, 4], 3, 1)
    assert list(td.ids_in_several_catalogs([t0, t1])) == [3]
    obs = td.build_observations([t0, t1])
    assert list(obs["id"]) == [3, 3]
    assert list(obs["c"]) == [0, 1]
    # the 3-parameter catalog's observation is padded
    assert list(obs["x"][1, :3]) == pytest.approx(list(t1["x"][0]))
    assert obs["x"][1, 3] == 0 and obs["xs"][1, 3] == 1e9


def test_observations_are_grouped_by_star():
    tables = [_table([1, 2, 3], 4, i) for i in range(3)]
    obs = td.build_observations(tables)
    assert list(obs["id"]) == [1, 1, 1, 2, 2, 2, 3, 3, 3]
    assert list(obs["c"]) == [0, 1, 2] * 3
    assert np.all(np.diff(obs["id"]) >= 0)


def test_dedupe_keeps_first_occurrence():
    t = _table([5, 5, 7, 5, 9], 3, 0)
    d = td.dedupe_by_id(t)
    assert list(d["id"]) == [5, 7, 9]
    assert np.array_equal(d["x"][0], t["x"][0])


def test_control_points_cover_every_catalog():
    tables = [_table(np.arange(i, i + 3000), 4 if i % 2 == 0 else 3, i) for i in range(3)]
    rng = np.random.default_rng(3)
    obs = td.build_observations(tables)
    ctrl = td.build_control_points(tables, obs, rng)
    assert set(np.unique(ctrl["c_control"])) == {0, 1, 2}
    assert ctrl["x_control"].shape[1] == 4
    assert np.all(np.isfinite(ctrl["x_control"]))


def test_control_point_budget_tracks_observations():
    # Three catalogs with the same 3000 stars: 9000 observations, 3000 per catalog.
    tables = [_table(np.arange(3000), 4, i) for i in range(3)]
    obs, ctrl = td.build_training_data(tables, seed=1)
    assert len(obs["x"]) == 9000
    # n_max = 9000 // (3 - 1), each catalog contributes min(4500, 2 * 3000, 3000)
    # real points, plus n_max // 16 dispersed ones.
    assert np.sum(ctrl["from_catalog"]) == 9000
    assert len(ctrl["x_control"]) == 9000 + 3 * (9000 // 2 // 16)


def test_save_load_observations_roundtrip(tmp_path):
    tables = [_table(np.arange(500), 4 if i < 2 else 3, i) for i in range(3)]
    obs, ctrl = td.build_training_data(tables, seed=1)
    path = td.save_observations(obs, ctrl, tmp_path / "o.h5", attrs=dict(seed=1))
    obs2, ctrl2 = td.load_observations(path)
    assert np.array_equal(obs["id"], obs2["id"])
    assert np.array_equal(obs["x"], obs2["x"])
    assert np.array_equal(obs["xs"], obs2["xs"])  # statistical errors are not touched
    assert obs2["n_stars"] == 500
    assert np.array_equal(obs2["star"], np.repeat(np.arange(500), 3))
    assert np.all(obs2["xs"][obs2["c"] == 2, 3] == 1e9)  # "not measured" survives
    assert ctrl2["xe_control"].max() < 1e6  # placeholders replaced on load
    assert ctrl2["missing_control"].any()  # ... after being recorded
    assert td.read_attrs(path)["seed"] == 1


@pytest.mark.filterwarnings("ignore")
def test_training_step_runs_and_reduces_loss(tmp_path):
    pytest.importorskip("tensorflow")
    pytest.importorskip("sonnet")
    from astrotune.train import TrainConfig, Trainer

    n_cat = 16
    rng = np.random.default_rng(0)
    n = 4096
    # Catalog 1 is catalog 0 offset by 0.5 in [Fe/H]
    x = rng.normal(size=(n, 4)).astype("f4")
    pairs = dict(
        x0=x,
        x1=x + np.array([0, 0.5, 0, 0], dtype="f4"),
        xe0=np.full((n, 4), 0.1, "f4"),
        xe1=np.full((n, 4), 0.1, "f4"),
        xs0=np.full((n, 4), 0.05, "f4"),
        xs1=np.full((n, 4), 0.05, "f4"),
        c0=np.zeros(n, "i4"),
        c1=np.ones(n, "i4"),
    )
    ctrl = dict(
        x_control=rng.normal(size=(n, 4)).astype("f4"),
        xe_control=np.full((n, 4), 0.1, "f4"),
        c_control=rng.integers(0, n_cat, n).astype("i4"),
        from_catalog=np.ones(n, bool),
    )
    cfg = TrainConfig(batch_size=256, n_epochs=4, max_steps=40, plot_every_seconds=None,
                      plot_every_steps=20)
    tr = Trainer(pairs, ctrl, cfg)
    metrics = tr.train(tmp_path)
    assert len(metrics["loss"]) == 40
    assert np.all(np.isfinite(metrics["loss"]))
    assert metrics["loss"][-1] < metrics["loss"][0]
    # Figures have increasing numerical suffixes
    names = sorted(p.name for p in (tmp_path / "progress").iterdir())
    assert names[:2] == ["progress_0001.png", "progress_0002.png"]
    tr.save(tmp_path, metrics)
    assert (tmp_path / "bijector" / "saved_model.pb").exists()


def test_video_frame_times_and_interpolation():
    from astrotune import videos

    for ease in ("linear", "sigmoid", "sine"):
        t = videos.get_frame_times(30, ease)
        assert t[0] == pytest.approx(0) and t[-1] == pytest.approx(1)
        assert np.all(np.diff(t) >= 0)
    traj = np.stack([np.zeros((3, 2)), np.ones((3, 2)), 2 * np.ones((3, 2))])
    assert np.allclose(videos.interpolate_at_time(traj, 0.25), 0.5)
    assert np.allclose(videos.interpolate_at_time(traj, 1.0), 2.0)


def _xerr_tables():
    """Three catalogs: all four parameters; no [alpha/Fe]; a parameter-dependent x_err."""
    def table(ids, d, xerr, seed):
        rng = np.random.default_rng(seed)
        n = len(ids)
        return Table({
            "id": np.asarray(ids, dtype="i8"),
            "x": rng.normal(size=(n, d)).astype("f4"),
            "sigma_x": np.full((n, d), 0.1, dtype="f4"),
            "x_err": xerr,
        })

    n = 3000
    ids = np.arange(n)
    a = table(ids, 4, np.tile(np.array([0.1, 0.1, 0.1, 0.4], "f4"), (n, 1)), 0)
    b = table(ids, 3, np.tile(np.array([0.3, 0.2, 0.1], "f4"), (n, 1)), 1)
    xe = np.tile(np.array([0.1, 0.1, 0.1, 0.05], "f4"), (n, 1))
    xe[:, 0] = np.where(np.arange(n) % 2 == 0, 0.1, 0.5)  # median 0.3 (half 0.1, half 0.5)
    xe[:, 3] = 1e9  # [alpha/Fe] missing for some stars
    xe[::10, 3] = 0.2
    c = table(ids, 4, xe, 2)
    return [a, b, c]


def test_dispersed_x_err_per_catalog():
    tables = _xerr_tables()
    a, b, c = td.dispersed_x_err(tables)
    assert a == pytest.approx([0.1, 0.1, 0.1, 0.4])
    # A catalog without [alpha/Fe] keeps the "not measured" value, which the loader pins
    assert b[:3] == pytest.approx([0.3, 0.2, 0.1]) and b[3] == td.MISSING_XE
    # Median over the stars that measure a parameter; only those count for [alpha/Fe]
    assert c[3] == pytest.approx(0.2)


def test_dispersed_x_err_override(monkeypatch):
    class Module:
        X_ERR_DISPERSED = (0.7, 0.7)

    monkeypatch.setattr(td.registry, "get_module", lambda key: Module if key == "k0" else object())
    a, b, c = td.dispersed_x_err(_xerr_tables(), keys=["k0", "k1", "k2"])
    assert a == pytest.approx([0.7, 0.7, 0.1, 0.4])  # overrides the leading parameters only
    assert b[:3] == pytest.approx([0.3, 0.2, 0.1])


def test_dispersed_modes_change_only_the_errors(tmp_path):
    tables = _xerr_tables()
    per, uni = td.build_training_data(tables, seed=1), td.build_training_data(
        tables, seed=1, dispersed_xerr="uniform"
    )
    for key in td.OBS_KEYS:
        assert np.array_equal(per[0][key], uni[0][key])
    # Same points in the same order; the real control points keep their own x_err
    for key in ("x_control", "c_control", "from_catalog"):
        assert np.array_equal(per[1][key], uni[1][key])
    real = per[1]["from_catalog"]
    assert np.array_equal(per[1]["xe_control"][real], uni[1]["xe_control"][real])
    assert np.all(uni[1]["xe_control"][~real] == td.LEGACY_DISPERSED_XE)

    # Per catalog: the dispersed values differ between catalogs and match dispersed_x_err
    expected = td.dispersed_x_err(tables)
    for k in range(3):
        sel = (~real) & (per[1]["c_control"] == k)
        assert np.all(per[1]["xe_control"][sel] == expected[k].astype("f4"))

    # The file records how it was made, and the loader pins the missing [alpha/Fe]
    path = td.save_observations(*per, tmp_path / "o.h5", attrs=dict(dispersed_xerr="per-catalog"))
    assert td.read_attrs(path)["dispersed_xerr"] == "per-catalog"
    _, ctrl = td.load_observations(path)
    sel = (~real) & (ctrl["c_control"] == 1)
    assert np.all(ctrl["xe_control"][sel][:, 3] == td.UNKNOWN_XE)


def test_progress_plot_skips_missing_parameters(tmp_path):
    from astrotune import training_plots as tp

    rng = np.random.default_rng(0)
    n = 50
    x = rng.normal(size=(n, 4)).astype("f4")
    x[:, 3] = 0.0  # placeholder for an unmeasured [alpha/Fe]
    ctrl = dict(
        c_control=np.zeros(n, dtype="i4"),
        from_catalog=np.ones(n, dtype=bool),
        x_control=x,
        missing_control=np.zeros((n, 4), dtype=bool),
    )
    ctrl["missing_control"][:, 3] = True
    valid = tp.catalog_sample_valid(ctrl, 1)
    assert valid[0].shape == (n, 4) and not valid[0][:, 3].any() and valid[0][:, :3].all()

    class _Bij:
        def return_path(self, x, c, n_frames):
            import tensorflow as tf

            return tf.stack([x, x + 0.1])

    import matplotlib.pyplot as plt

    fig = plt.figure()
    tp.plot_transformed_params(_Bij(), tp.catalog_samples(ctrl, 1), ["a"], [(0, 0, 1)], fig, valid)
    for ax in fig.axes[2:]:  # the [Fe/H]-[alpha/Fe] row
        assert len(ax.collections[0].get_offsets().compressed()) == 0
    for ax in fig.axes[:2]:
        assert ax.collections[0].get_offsets().compressed().size == 2 * n
    plt.close(fig)


@pytest.mark.parametrize("plot_diff", [False, True])
def test_param1v1_video(tmp_path, plot_diff):
    from astrotune import videos as v

    rng = np.random.default_rng(1)
    n, nf = 60, 5
    a = rng.normal(size=(nf, n, 4))
    a[..., 0] -= 1.0
    b = a + rng.normal(scale=0.1, size=a.shape) * np.linspace(1, 0, nf)[:, None, None]
    valid = np.ones((n, 4), dtype=bool)
    valid[:, 2] = False  # logg not measured by both
    pair = ((0, 1), (v.Trajectories(a, valid), v.Trajectories(b, valid)))
    out = v.make_param1v1_video(
        tmp_path / "x.mp4", pair, ["A", "B"], plot_diff=plot_diff, n_video_frames=4
    )
    assert out.exists() and out.stat().st_size > 0


def test_teff_axis_labels_a_narrow_range():
    import matplotlib.pyplot as plt
    from astrotune import videos as v

    lim = (5 * np.log(3.0 / 5), 5 * np.log(4.0 / 5))  # 3000-4000 K
    fig, ax = plt.subplots()
    ax.set_xlim(lim)
    v.format_teff_axis(ax.xaxis, lim)
    labels = [ax.xaxis.get_major_formatter()(x) for x in ax.xaxis.get_major_locator()()]
    assert len(labels) >= 3 and "3.4" in labels
    plt.close(fig)


def test_observation_pairs_build_every_catalog_pair():
    from astrotune.evaluate import PAIR_FIELDS, observation_pairs

    tables = [_table([1, 2], 4, i) for i in range(3)]
    obs = td.build_observations(tables)
    pairs = observation_pairs(obs, max_stars=10)
    assert set(pairs) == set(PAIR_FIELDS)
    assert len(pairs["c0"]) == 2 * 3  # two stars, three catalog pairs each
    assert np.all(pairs["c0"] < pairs["c1"])
    # A repeated observation of one catalog does not create extra pairs
    repeated = [_table([1, 1, 2], 4, 0), _table([1, 2], 4, 1)]
    pairs = observation_pairs(td.build_observations(repeated), max_stars=10)
    assert len(pairs["c0"]) == 2


def test_initial_y_is_the_precision_weighted_mean():
    from astrotune.train_latent import initial_y

    t0 = _table([1, 2], 4, 0)
    t1 = _table([1, 2], 4, 1)
    t1["sigma_x"][1] = 10.0  # star 2 is measured much less precisely here
    y = initial_y(td.build_observations([t0, t1]))
    x0, x1 = np.asarray(t0["x"]), np.asarray(t1["x"])
    s0, s1 = np.asarray(t0["sigma_x"]), np.asarray(t1["sigma_x"])
    for k in range(2):
        w0, w1 = 1.0 / s0[k] ** 2, 1.0 / s1[k] ** 2
        assert np.allclose(y[k], (w0 * x0[k] + w1 * x1[k]) / (w0 + w1), atol=1e-6)


def test_initial_y_only_uses_the_catalogs_that_measure_an_axis():
    from astrotune.train_latent import initial_y

    t0 = _table([1], 4, 0)
    y = initial_y(td.build_observations([t0, _table([1], 3, 1)]))
    # Catalog 1 has no [alpha/Fe] (sigma = 1e9), so the axis is catalog 0's value
    assert np.allclose(y[0, 3], np.asarray(t0["x"])[0, 3], atol=1e-4)
    # Neither catalog measures it: the axis stays at 0
    y = initial_y(td.build_observations([_table([1], 3, 0), _table([1], 3, 1)]))
    assert y[0, 3] == 0.0


@pytest.mark.filterwarnings("ignore")
def test_latent_training_moves_y_and_saves(tmp_path):
    pytest.importorskip("tensorflow")
    pytest.importorskip("sonnet")
    from astrotune.train_latent import LatentTrainConfig, LatentTrainer

    tables = [_table(np.arange(1200), 4, i) for i in range(2)]
    tables[1]["x"] = tables[1]["x"] + np.array([0, 0.5, 0, 0], dtype="f4")
    obs, ctrl = td.build_training_data(tables, seed=1)
    cfg = LatentTrainConfig(
        hidden_size=16, n_hidden=1, batch_size=256, n_epochs=2, max_steps=8,
        plot_every_seconds=None, plot_every_steps=None,
        checkpoint_every_steps=None, agreement_n_stars=256,
    )
    tr = LatentTrainer(obs, ctrl, cfg)
    y0 = tr.y.numpy().copy()
    metrics = tr.train(tmp_path)
    assert len(metrics["loss"]) == 8
    assert np.all(np.isfinite(metrics["loss"]))
    assert not np.allclose(tr.y.numpy(), y0)  # the free parameters moved
    tr.save(tmp_path, metrics)
    assert (tmp_path / "bijector" / "saved_model.pb").exists()
    saved = np.load(tmp_path / "latent_y.npz")
    assert saved["y"].shape == (1200, 4) and saved["id"].shape == (1200,)


def test_held_out_stars_is_a_random_sample_not_a_prefix():
    """The held-out set must not be a block of the id order.

    Gaia ids encode a HEALPix index, so the lowest ids are a patch of sky; cutting
    on the sorted ids made the validation set a region (LAMOST-heavy, DESI-poor by
    a factor of 7) rather than a sample.
    """
    from astrotune.train_latent import held_out_stars

    n_stars = 20000
    is_val = held_out_stars(n_stars, 0.125, seed=3)

    assert is_val.dtype == bool and len(is_val) == n_stars
    assert is_val.sum() == int(n_stars * 0.125)
    # not the lowest ids
    assert not np.array_equal(np.flatnonzero(is_val), np.arange(is_val.sum()))
    # spread evenly over the index range instead
    bins = np.array_split(np.arange(n_stars), 10)
    for b in bins:
        assert 0.10 < is_val[b].mean() < 0.15


def test_held_out_stars_is_reproducible_and_seed_dependent():
    from astrotune.train_latent import held_out_stars

    a = held_out_stars(5000, 0.2, seed=1)
    assert np.array_equal(a, held_out_stars(5000, 0.2, seed=1))
    assert not np.array_equal(a, held_out_stars(5000, 0.2, seed=2))
    assert a.sum() == 1000


def test_shuffled_split_keeps_the_catalog_mix():
    """A split on a sky-correlated id must not change which surveys are held out.

    The observations are laid out like the real file: a low-id region dominated by
    one catalog and a high-id region by another.
    """
    from astrotune.train_latent import held_out_stars

    n_stars = 40000
    star = np.repeat(np.arange(n_stars), 2)  # two observations per star
    c = np.where(star < n_stars // 2, 8, 5)  # "DESI" low ids, "LAMOST" high ids

    old = star < int(n_stars * 0.125)  # the previous, id-ordered split
    new = held_out_stars(n_stars, 0.125, seed=3)[star]

    def share(mask, cat):
        return (c[mask] == cat).mean()

    # the old split holds out the low-id region only, so its held-out set is a
    # single catalog while training keeps both
    assert share(old, 8) > 0.99 and share(old, 5) < 0.01
    # the shuffled one keeps both sides representative
    for cat in (8, 5):
        assert abs(share(new, cat) - share(~new, cat)) < 0.02


def test_validation_schedule_covers_the_held_out_set_once_per_epoch():
    """The default must sweep every held-out observation exactly once per epoch."""
    from astrotune.train_latent import validation_schedule

    n_batches, n_train, n_val = 52, 3_391_987, 484_492
    every, batch, batches = validation_schedule(n_batches, n_train, n_val)

    assert every == 7  # round(3.39e6 / 4.84e5)
    calls_per_epoch = n_batches // every
    assert batches == calls_per_epoch  # one pass per epoch, not more
    assert batch * batches >= n_val  # and it is a full pass
    assert batch * (batches - 1) < n_val  # with no wasted batch
    # close to the training batch size, so chi2 and chi2_val are comparable
    assert 0.8 < batch / 65_536 < 1.2


def test_validation_schedule_is_overridable_and_always_covers():
    from astrotune.train_latent import validation_schedule

    cases = [
        (52, 3_391_987, 484_492, 1),
        (52, 3_391_987, 484_492, 13),
        (52, 3_391_987, 4_000, None),
        (52, 3_391_987, 4_000, 100),  # less often than once per epoch
        (52, 3_391_987, 100, 1),  # fewer val observations than training batches
        (3, 1000, 30, None),
    ]
    for n_batches, n_train, n_val, every in cases:
        got_every, batch, batches = validation_schedule(n_batches, n_train, n_val, every)
        assert got_every >= 1 and batch >= 1
        assert batch <= n_val
        assert batch * batches >= n_val, (n_batches, n_val, every)
        # an epoch of validation calls must cover the held-out set
        assert (n_batches // got_every) * batch >= n_val or batch == n_val, (
            n_batches, n_val, every,
        )


def test_validation_curve_is_plotted_against_its_own_steps():
    """There is no longer one validation point per step, so the x-axis must come
    from ``step_val``: plotting the series against its index would compress it into
    the first sliver of the panel, where with ``val_every=1`` it looked right."""
    from matplotlib.figure import Figure

    pytest.importorskip("matplotlib")
    from astrotune.train_latent import LATENT_PENALTIES
    from astrotune.training_plots import plot_loss_history

    n, every = 1040, 7
    step = np.arange(n)
    metrics = {
        "loss": 4.2 + 6.0 * np.exp(-step / 250.0),
        "chi2": 4.1 + 6.0 * np.exp(-step / 250.0),
        "penalty_obs": np.full(n, 2e-3),
        "penalty_ctrl": np.full(n, 2.5e-2),
        "penalty_div": np.full(n, 4e-2),
        "penalty_l2": np.full(n, 2.5e-3),
        "norm": np.full(n, 20.0),
        "step_val": list(range(every, n + 1, every)),
    }
    metrics["loss_val"] = [3.2 + 6.0 * np.exp(-s / 250.0) for s in metrics["step_val"]]

    fig = Figure(figsize=(12, 3))
    plot_loss_history(metrics, fig, penalties=LATENT_PENALTIES)
    ax = fig.axes[1]
    val = [ln for ln in ax.get_lines() if ln.get_label() == "validation loss"][0]

    x = np.asarray(val.get_xdata())
    assert len(x) == len(metrics["step_val"]) // 2 or len(x) == len(metrics["step_val"]) // 2 + 1
    # in step units, on the same axis as the training curve -- not an index
    assert x.max() == pytest.approx(max(metrics["step_val"]) / 1000.0)
    assert x.max() > 1.0
    train_x = np.asarray(ax.get_lines()[0].get_xdata())
    assert x.min() >= train_x.min()
