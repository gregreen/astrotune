"""Rebuild a trained bijector from a run directory."""

from __future__ import annotations

import json
from pathlib import Path

import tensorflow as tf

from .. import registry
from .ode_bijector import ODEBijector


def load_bijector(run_dir) -> ODEBijector:
    """Restore the bijector from the latest checkpoint in ``run_dir``.

    Unlike the exported SavedModel, which only exposes ``to_harmonized`` and
    ``from_harmonized``, this returns the full :class:`ODEBijector`
    (``return_path``, ``forward_with_s``, ...).
    """
    run_dir = Path(run_dir)
    with open(run_dir / "config.json") as f:
        cfg = json.load(f)["config"]

    bij = ODEBijector(
        n_dim=4,
        n_cond=len(registry.run_keys(run_dir)),
        hidden_size=cfg["hidden_size"],
        n_hidden=cfg["n_hidden"],
        rtol=cfg["ode_rtol"],
        atol=cfg["ode_atol"],
        vector_network_kwargs=dict(use_residual=True, use_film=True, use_layer_norm=True),
    )
    path = tf.train.latest_checkpoint(str(run_dir / "checkpoints"))
    if path is None:
        raise FileNotFoundError(f"No checkpoint in {run_dir}/checkpoints")
    tf.train.Checkpoint(bij=bij).restore(path).assert_existing_objects_matched()
    return bij
