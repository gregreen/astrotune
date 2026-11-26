"""Rebuilding a trained bijector from a run directory."""

import json

import numpy as np
import pytest

pytest.importorskip("tensorflow")

import tensorflow as tf  # noqa: E402

from astrotune.model.io import fourier_kwargs, load_bijector  # noqa: E402
from astrotune.model.ode_bijector import ODEBijector  # noqa: E402

CATALOGS = ["apogee_dr17", "galah_dr4"]


def _config(**fourier):
    cfg = dict(hidden_size=8, n_hidden=1, ode_rtol=1e-5, ode_atol=1e-5)
    cfg.update(fourier)
    return cfg


def _save_run(tmp_path, cfg, step=7):
    """Write a run directory holding one checkpoint of a fresh bijector."""
    run_dir = tmp_path / "run"
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    bij = ODEBijector(
        4, len(CATALOGS),
        hidden_size=cfg["hidden_size"], n_hidden=cfg["n_hidden"],
        rtol=cfg["ode_rtol"], atol=cfg["ode_atol"],
        vector_network_kwargs=dict(use_residual=True, use_film=True, use_layer_norm=True),
        fourier=fourier_kwargs(cfg),
    )
    x = np.random.default_rng(0).normal(scale=0.1, size=(4, 4)).astype("f4")
    c = np.array([0, 1, 0, 1], dtype="i4")
    # A call first, so that lazily-created variables exist when the checkpoint is
    # written (the trainer builds them long before it checkpoints).
    expected = bij.return_path(tf.constant(x), tf.constant(c), 2).numpy()
    tf.train.Checkpoint(bij=bij).write(str(ckpt_dir / f"ckpt-{step}"))
    with open(run_dir / "config.json", "w") as f:
        json.dump({"config": cfg, "catalogs": CATALOGS}, f)
    return run_dir, x, c, expected


def test_fourier_kwargs_comes_from_the_saved_config():
    assert fourier_kwargs({"fourier_n_modes": 0, "fourier_time_harmonics": 0}) is None
    assert fourier_kwargs({}) is None  # a run from before the block existed
    assert fourier_kwargs(
        {"fourier_n_modes": 0, "fourier_time_harmonics": 2}
    ) == dict(n_modes=0, lam_min=0.25, lam_max=8.0, time_harmonics=2)
    assert fourier_kwargs(
        {"fourier_n_modes": 6, "fourier_lam_min": 0.5, "fourier_lam_max": 4.0,
         "fourier_time_harmonics": 0}
    ) == dict(n_modes=6, lam_min=0.5, lam_max=4.0, time_harmonics=0)


def test_a_fourier_run_is_restored_with_its_features(tmp_path):
    """Without the block the first layer has the wrong width and the map is wrong."""
    cfg = _config(fourier_n_modes=0, fourier_time_harmonics=2)
    run_dir, x, c, expected = _save_run(tmp_path, cfg)

    bij = load_bijector(run_dir)
    assert bij.v.n_features == 4
    got = bij.return_path(tf.constant(x), tf.constant(c), 2).numpy()
    assert np.allclose(got, expected, atol=1e-6)


def test_a_run_without_fourier_features_still_loads(tmp_path):
    cfg = _config()
    run_dir, x, c, expected = _save_run(tmp_path, cfg)

    bij = load_bijector(run_dir)
    assert not bij.v.n_features
    got = bij.return_path(tf.constant(x), tf.constant(c), 2).numpy()
    assert np.allclose(got, expected, atol=1e-6)
