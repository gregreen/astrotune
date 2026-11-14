"""Rebuild a trained bijector from a run directory."""

from __future__ import annotations

import json
from pathlib import Path

import tensorflow as tf

from .. import registry
from .ode_bijector import ODEBijector


def enable_memory_growth() -> None:
    """Make TensorFlow allocate GPU memory as needed instead of reserving
    nearly all of it. Call before the first TensorFlow operation; the GPU is
    shared (a process without this holds ~95% of it, whatever it really uses)."""
    for device in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(device, True)
        except RuntimeError:  # already initialized
            pass


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
    # The checkpoint directory is scanned rather than read through
    # ``tf.train.latest_checkpoint``: the trainer writes ``Checkpoint`` objects
    # directly so that it can keep old milestones alongside recent checkpoints
    # (see ``train_latent.checkpoints_to_keep``), and no longer maintains the
    # ``checkpoint`` proto that ``latest_checkpoint`` needs. Scanning also picks
    # the numerically newest step, which the proto's order would not guarantee
    # once checkpoints are deleted out of order.
    index_files = list((run_dir / "checkpoints").glob("ckpt-*.index"))
    if not index_files:
        raise FileNotFoundError(f"No checkpoint in {run_dir}/checkpoints")
    path = str(
        max(index_files, key=lambda p: int(p.name.split("-")[1].split(".")[0]))
    )[: -len(".index")]
    # ``expect_partial``: the checkpoints also hold ``latent_y``/``latent_y_val``,
    # which this object graph does not ask for, and ``Checkpoint`` always carries
    # a ``save_counter`` of its own -- both would fail a strict assertion in
    # opposite directions. What matters is that every bijector variable is bound,
    # and ``restore`` binds by object path.
    tf.train.Checkpoint(bij=bij).restore(path).expect_partial()
    return bij
