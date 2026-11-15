"""Tests for the video pipeline: which data it reads, and what it integrates.

The rendering is not exercised (that needs ffmpeg); what is tested is the part
that was rebuilt: choosing the training file from the run's config, deciding
which observations every requested video needs, deduplicating them, and running
one integration per batch instead of one per catalog and per pair.
"""

import json

import numpy as np
import pytest

from astrotune import registry
from astrotune import videos as vd
from astrotune.training_data import save_observations

tf = pytest.importorskip("tensorflow")


def _write_observations(path, n_star=6, n_cat=3, n_ctrl=6, seed=0):
    """An observations + control points file with every star in every catalog."""
    rng = np.random.default_rng(seed)
    n_obs = n_star * n_cat
    star = np.repeat(np.arange(n_star), n_cat)
    obs = {
        "id": (star.astype("i8") + 1) * 11,  # one Gaia id per star
        "c": np.tile(np.arange(n_cat), n_star).astype("i4"),
        "x": rng.normal(size=(n_obs, 4)).astype("f4"),
        "xe": np.full((n_obs, 4), 0.1, "f4"),
        "xs": np.full((n_obs, 4), 0.05, "f4"),
    }
    n_ctrl_rows = n_cat * n_ctrl
    ctrl = {
        "x_control": rng.normal(size=(n_ctrl_rows, 4)).astype("f4"),
        "xe_control": np.full((n_ctrl_rows, 4), 0.1, "f4"),
        # the third parameter is unmeasured for catalog 2: tests the valid mask
        "c_control": np.repeat(np.arange(n_cat), n_ctrl).astype("i4"),
        "from_catalog": np.ones(n_ctrl_rows, dtype=bool),
    }
    ctrl["xe_control"][ctrl["c_control"] == 2, 3] = 1e9
    save_observations(obs, ctrl, path, attrs={"seed": 0})
    return obs, ctrl


def _run_dir(tmp_path, keys, observations_path=None, pairs_path=None):
    run = tmp_path / "run"
    run.mkdir(parents=True, exist_ok=True)
    cfg = {"catalogs": list(keys)}
    if observations_path is not None:
        cfg["observations_path"] = str(observations_path)
    if pairs_path is not None:
        cfg["pairs_path"] = str(pairs_path)
    (run / "config.json").write_text(json.dumps(cfg))
    return run


class _StubBijector:
    """A bijector whose flow does nothing: the path is the starting point."""

    def __init__(self):
        self.calls = []

    def return_path(self, x, c, n):
        self.calls.append((int(x.shape[0]), int(n)))
        return tf.tile(x[None, :, :], [n, 1, 1])


def test_source_is_chosen_from_the_run_config(tmp_path):
    keys = registry.keys()[:3]
    path = tmp_path / "obs.h5"
    _write_observations(path)
    run = _run_dir(tmp_path, keys, observations_path=path)
    src = vd.load_source(run)
    assert src.kind == "observations" and src.path == path
    assert src.n_cat == 3 and src.keys == tuple(keys)
    assert len(src.rows()) == 3 * 6  # one row per (star, catalog)
    assert np.array_equal(src.stars(0), np.arange(6))
    assert src.shared_count(0, 1) == 6
    # --data-path overrides what the config says
    other = tmp_path / "other_observations.h5"
    _write_observations(other, n_star=4)
    assert vd.load_source(run, other).obs["id"].max() == 4 * 11


def test_source_falls_back_when_the_config_names_no_file(tmp_path):
    keys = registry.keys()[:3]
    run = _run_dir(tmp_path, keys)
    with pytest.raises(FileNotFoundError, match="--data-path"):
        vd.load_source(run)
    # a legacy run records pairs_path, and a missing file says so
    run2 = _run_dir(tmp_path / "legacy", keys, pairs_path=tmp_path / "nope.h5")
    with pytest.raises(FileNotFoundError, match="does not exist"):
        vd.load_source(run2)


def test_plan_integrates_control_points_and_one_copy_of_each_pair_star(tmp_path):
    keys = registry.keys()[:3]
    path = tmp_path / "obs.h5"
    _write_observations(path, n_star=6, n_ctrl=4)
    src = vd.load_source(_run_dir(tmp_path, keys, observations_path=path))

    plan = vd.build_plan(src, [(0, 1)], n_plot=5)
    assert plan.n_ctrl == 3 * 4  # every catalog's control points
    assert plan.n_pair == 10  # 5 shared stars x 2 sides, no duplicates yet
    assert len(plan) == plan.n_ctrl + plan.n_pair
    # the control-point valid mask keeps the unmeasured parameter out
    assert not plan.valid[plan.ctrl_idx[2]][:, 3].any()
    assert plan.valid[plan.ctrl_idx[0]][:, 3].all()

    # two pairs that share a catalog share its stars: they must be integrated once
    plan2 = vd.build_plan(src, [(0, 1), (1, 2)], n_plot=10)
    assert plan2.n_pair_before_dedup == 4 * 6  # 6 stars x 2 sides x 2 pairs
    assert plan2.n_pair == 3 * 6  # catalog 0, 1 and 2 rows, catalog 1 shared
    a0, b0, valid = plan2.pair_idx[(0, 1)]
    a1, b1, _ = plan2.pair_idx[(1, 2)]
    assert np.array_equal(b0, a1)  # the same catalog-1 stars, the same paths
    assert np.array_equal(plan2.c[b0], np.full(len(b0), 1))
    assert valid.all()


def test_one_large_batch_per_batch_size_and_the_slices_line_up(tmp_path):
    keys = registry.keys()[:3]
    path = tmp_path / "obs.h5"
    _write_observations(path, n_star=6, n_ctrl=4)
    src = vd.load_source(_run_dir(tmp_path, keys, observations_path=path))
    plan = vd.build_plan(src, [(0, 1)], n_plot=6)

    bij = _StubBijector()
    xt = vd.integrate(bij, plan, n_frames=3, batch_size=7)
    assert xt.shape == (3, len(plan), 4)
    n = len(plan)  # 12 control points + 12 pair rows
    assert [c[0] for c in bij.calls] == [7] * (n // 7) + [n % 7]
    assert all(c[1] == 3 for c in bij.calls)

    catalogs, pairs = vd.make_trajectories(plan, xt)
    assert len(catalogs) == 3
    assert catalogs[0].xt.shape == (3, 4, 4)  # the catalog's 4 control points
    (i0, i1), (t0, t1) = next(iter(pairs.items()))
    assert (i0, i1) == (0, 1)
    assert t0.xt.shape == t1.xt.shape == (3, 6, 4)
    assert np.allclose(t0.xt[0], plan.x[plan.pair_idx[(0, 1)][0]])


def _capture(monkeypatch):
    calls = []
    monkeypatch.setattr(vd, "make_plane_video",
                        lambda path, *a, **k: calls.append(path.name) or path)
    monkeypatch.setattr(vd, "make_param1v1_video",
                        lambda path, *a, **k: calls.append(path.name) or path)
    monkeypatch.setattr("astrotune.model.load_bijector", lambda run_dir: _StubBijector())
    return calls


def test_make_videos_selects_the_requested_videos(tmp_path, monkeypatch):
    keys = registry.keys()[:3]
    path = tmp_path / "obs.h5"
    _write_observations(path, n_star=40, n_ctrl=25)
    run = _run_dir(tmp_path, keys, observations_path=path)
    calls = _capture(monkeypatch)
    common = dict(run_dir=run, n_video_frames=2, n_plot=20)

    vd.make_videos(planes=["kiel"], highlights=[keys[1]], pairs=[], **common)
    assert calls == ["kiel.mp4", "kiel_control_1.mp4"]

    calls.clear()
    vd.make_videos(planes=["kiel", "param1v1"], highlights=[], pairs=[(keys[0], keys[1])], **common)
    assert calls == ["kiel.mp4", "kiel_pair_0_1.mp4", "param1v1_pair_0_1.mp4"]

    calls.clear()  # a pair-only plane with no pair is skipped, not an error
    vd.make_videos(planes=["param1v1"], highlights=[], pairs=[], plain=False, **common)
    assert calls == []

    calls.clear()  # --all-pairs-min picks the pairs up from the data itself
    vd.make_videos(planes=["kiel"], highlights=[], pairs=[], all_pairs_min=10,
                   plain=False, **common)
    assert sorted(calls) == ["kiel_pair_0_1.mp4", "kiel_pair_0_2.mp4", "kiel_pair_1_2.mp4"]


def test_cli_takes_lists_for_planes_and_highlights(tmp_path, monkeypatch):
    keys = registry.keys()[:3]
    path = tmp_path / "obs.h5"
    _write_observations(path)
    run = _run_dir(tmp_path, keys, observations_path=path)
    seen = {}
    monkeypatch.setattr(vd, "make_videos", lambda *a, **k: seen.update(k, run_dir=a[0], highlights=a[2]))

    vd.main(["--run-dir", str(run), "--plane", "kiel", "feh_alpha",
             "--highlight", keys[0], keys[1], "--all-pairs-min", "5"])
    assert seen["planes"] == ["kiel", "feh_alpha"]
    assert seen["highlights"] == [keys[0], keys[1]]
    assert seen["all_pairs_min"] == 5

    vd.main(["--run-dir", str(run), "--all-planes", "--highlight-all"])
    assert seen["planes"] == list(vd.ALL_PLANES)
    assert seen["highlights"] == list(keys)

    vd.main(["--run-dir", str(run)])  # the defaults are filtered to this run's catalogs
    assert seen["planes"] == ["kiel", "feh_alpha"]
    assert seen["highlights"] == []

    for bad in (["--plane", "kiel", "--all-planes"],
                ["--highlight", keys[0], "--highlight-all"],
                ["--highlight", "not_a_catalog"],
                ["--plane", "nonsense"]):
        with pytest.raises(SystemExit):
            vd.main(["--run-dir", str(run)] + bad)
