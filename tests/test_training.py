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


@pytest.mark.skip(reason="the trainer is being ported to the latent-y loss")
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
