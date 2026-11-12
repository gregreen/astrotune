"""Train the bijector with a free harmonized parameter vector for every star.

    astrotune-train-latent --run-name my_run

The training data are single catalog observations (:mod:`astrotune.training_data`):
one row per measurement, grouped by Gaia id. This model adds one free parameter
vector ``y`` per star and fits

    L_data = mean over observations of rho( sum_p (x_p - f_c^-1(y)_p)^2 / sigma_p^2 ),

with ``rho`` the softened chi^2 of the pair loss. Every observation of a star is
compared with the *same* ``y``, so no catalog is regressed onto another and the
inverse-variance weighting cannot shrink a map. A parameter that a catalog does
not measure carries ``sigma = 1e9``, so it drops out of the chi^2 without a
mask.

``y`` starts at the precision-weighted mean of the star's observations. Each step
then fits it by solving the star's Gauss-Newton normal equations with the
Jacobian of ``f_c^-1`` transported through the ODE, and only afterwards takes the
map's gradient step at the new ``y``: the two blocks are optimised in turn, so
the map never trains against harmonized values that are about to move, and the
solve that produces the Jacobian stays outside the tape that differentiates the
map. All observations of a star are in the same batch (the batches are blocks of
a random permutation of the stars), so one solve sees all of its data and lands
on the optimum. ``y_solver="gradient"`` keeps the older variant, which updates
``y`` from the same tape as the map and needs no Jacobian. ``y`` is state of the
run, not of the exported SavedModel: inference for a star observed by one catalog
goes through ``to_harmonized`` as before.

The held-out stars have a second array, ``y_val``, fitted by the same rule on the
validation batches and remembered between calls. Without it the validation loss
would profile a cold ``y`` (one sweep from the precision-weighted mean) while the
training loss uses a ``y`` already at its running fixed point, and the two would
differ by about 0.3 in chi^2 for that reason alone.

A run can start with the maps alone: ``pretrain_steps`` leaves ``y`` at its
initial guess for the first that many steps, so the bijections are fitted
against the precision-weighted means before ``y`` is allowed to move, and a
value at least as large as the run is exactly "pre-train the bijections and
stop". Held-out stars keep their own initial guess during that phase, so the
validation loss measures the same objective as the training loss.

The epoch is ``block_shuffles`` independent permutations of the star blocks
concatenated, i.e. ``block_shuffles`` passes over the training set, so that a
``drop_remainder`` cut only costs the stars in the tail of one permutation one
of their ``block_shuffles`` updates. With ``drop_remainder=False`` (the default)
nothing is skipped at all and the last batch of an epoch is short.

The training loop runs in a graph: :meth:`LatentTrainer.train` calls a
``tf.function`` that advances the dataset iterator and the optimizer for a chunk
of steps inside one ``while_loop``, and does in Python only what needs the host
(the progress figure, the catalog agreement, the checkpoints).

The pair-loss path (:mod:`astrotune.train`, ``astrotune-train``) is untouched and
still reads a legacy ``data/pairs_and_ctrl_points.h5``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import signal
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tqdm.auto import tqdm

from . import config, registry
from .evaluate import (
    PARAM_SHORT,
    evaluate_bijector,
    format_summary,
    observation_pairs,
    quick_agreement,
)
from .model import ODEBijector
from .model.io import enable_memory_growth
from .train import arcsinh_of_exp
from .training_data import load_observations, read_attrs
from .training_plots import (
    catalog_colors,
    catalog_sample_valid,
    catalog_samples,
    save_progress_figure,
)

log = logging.getLogger(__name__)

#: ``sigma_x`` above this marks a parameter that the catalog does not measure
MISSING = 1e6

#: Penalty metrics drawn by the progress figure
LATENT_PENALTIES = (
    ("penalty_obs", "penalty (obs)"),
    ("penalty_ctrl", "penalty (control)"),
    ("penalty_div", "penalty (div)"),
    ("penalty_l2", "penalty (l2)"),
)

#: Per-step metrics recorded by the training loop
METRIC_KEYS = (
    "loss",
    "chi2",
    "penalty_obs",
    "penalty_ctrl",
    "penalty_div",
    "penalty_l2",
    "norm",
    "y_trust",
)


@dataclass
class LatentTrainConfig:
    # Network
    hidden_size: int = 64
    n_hidden: int = 3
    ode_rtol: float = 1e-5
    ode_atol: float = 1e-5

    # Optimization
    batch_size: int = 16384  # observations per batch
    n_epochs: int = 16
    #: Independent block permutations concatenated into one epoch. The data of an
    #: epoch are ``block_shuffles`` passes over the training set, so with
    #: ``drop_remainder`` the rows lost at the end of an epoch are those of one
    #: permutation only, and the stars in them lose one update out of
    #: ``block_shuffles`` instead of all of them.
    block_shuffles: int = 1
    #: Whether the short final batch of an epoch is discarded (constant batch
    #: shape) or kept (no row is ever skipped).
    drop_remainder: bool = False
    val_frac: float = 1 / 8  # fraction of *stars* held out
    learning_rate: float = 1e-3
    #: Number of piecewise-constant segments the rate is spread over; the rate
    #: is halved from one segment to the next, so it covers a factor
    #: ``2 ** (lr_schedule_phases - 1)``. The segments are equal fractions of
    #: the run, which with ``n_epochs`` fixed is also equal fractions of its
    #: data, so the schedule is comparable across batch sizes. Set it to 1 for
    #: a constant rate, which is what an lr comparison wants.
    lr_schedule_phases: int = 8
    #: Fraction of the run over which the rate ramps linearly from
    #: ``learning_rate / warmup_steps`` up to ``learning_rate`` before the decay
    #: takes over. Adam's step is ``~lr`` whatever the gradient norm is, so a run
    #: that starts at the full rate can travel a long way before the first drop;
    #: ``latent_gn_5h`` diverged at step ~930, still inside the first (full-rate)
    #: segment. Zero disables the warmup.
    lr_warmup_frac: float = 0.05
    #: Ceiling on the global norm of the map's gradient. Note that Adam's step is
    #: close to ``lr`` irrespective of this: clipping a gradient that is 1e9 only
    #: keeps the *relative* weights of the coordinates sane, it does not bound the
    #: step, which is why the warmup and the ``y`` trust region carry the load.
    clip_norm: float = 100.0
    seed: int = 3
    #: How often to validate, in training steps. ``None`` uses the data-size
    #: ratio, ``round(n_train_obs / n_val_obs)``, which makes one epoch's worth of
    #: validation calls sweep the entire held-out set exactly once (see
    #: ``_make_data``, which also sizes the validation batch to make that exact).
    val_every: Optional[int] = None
    max_steps: Optional[int] = None  # stop early (mainly for testing)

    # The free harmonized parameters
    #: How ``y`` is fitted. ``"gauss_newton"`` solves the per-star normal
    #: equations with the Jacobian transported through the ODE (one exact step,
    #: and nothing to tune); ``"gradient"`` takes the step along the loss's own
    #: gradient divided by the star's total precision, the older behaviour that
    #: needs no Jacobian and is several times cheaper per step.
    y_solver: str = "gauss_newton"
    #: Damping of the ``y`` step (1 is the full step), for either solver.
    y_learning_rate: float = 1.0
    #: Ridge of the per-star normal equations, relative to the mean of their
    #: diagonal: a direction of curvature ``c`` is followed by ``c / (c + ridge
    #: * cbar)``. Some stars constrain one direction very poorly, where the
    #: unregularised step can be hundreds of sigma, and those are exactly the
    #: steps that fling a star and drag the maps with it. At 1e-3 a direction
    #: with a tenth of the mean curvature loses 1% of its step, so balanced stars
    #: are unaffected to within a part in 1e3.
    y_ridge: float = 1e-3
    #: Trust region on the ``y`` step, in absolute parameter value: the largest
    #: single-parameter change a star may take in one sweep is capped at this.
    #: ``None`` or 0 disables it. The step is scaled as a whole, never clipped
    #: per dimension, so its direction is preserved. See
    #: ``LatentTrainer._trust_region``.
    y_trust_abs: Optional[float] = 0.5
    #: Steps at the start of the run during which ``y`` is left at its initial
    #: guess, so the bijections are pre-trained against the precision-weighted
    #: means before ``y`` is allowed to move; the maps then start out consistent
    #: with a reasonable ``y``. Zero disables the phase, and setting it to at
    #: least the whole run is exactly "pre-train the bijections and stop".
    pretrain_steps: int = 0

    # Observation loss
    chi2_turnover: float = 50.0
    #: Whether the path length of the observations' inverse map enters the loss
    #: (the pair trainer's ``pair_path_in_loss``, off by default there too).
    obs_path_in_loss: bool = False

    # Path-length penalty (control points)
    l1_path: float = 1e-1
    l2_path: float = 1e-1

    # Distance of control points from their images
    l1_total: float = 1e-1
    l2_total: float = 1e-1
    sigma_ctrl: float = 0.3  # scatter added to control points

    # Stretching / contraction of space
    eps_div: float = 0.1
    weight_div: float = 3.0

    # Weight decay
    l2_weights: float = 1e-4

    # Freeze variables whose names contain these strings once this fraction of
    # training is done (see the pair trainer).
    freeze_after_frac: Optional[float] = None
    freeze_vars: Tuple[str, ...] = ("embedding", "film")

    # Progress figures
    plot_every_seconds: Optional[float] = 5 * 60.0
    plot_every_steps: Optional[int] = None
    n_plot: int = 2048  # stars per catalog in the progress figure
    #: Held-out stars used for the catalog agreement at each progress figure
    agreement_n_stars: int = 16384

    # Resumable state (model, optimizer, metrics) is written at each progress
    # figure and at this interval
    checkpoint_every_seconds: Optional[float] = 10 * 60.0


#: Settings that may differ when resuming a run; everything else must match
RESUME_MUTABLE = (
    "plot_every_seconds",
    "plot_every_steps",
    "checkpoint_every_seconds",
    "agreement_n_stars",
    "n_plot",
)


# --- Data -----------------------------------------------------------------------
def initial_y(obs_dict) -> np.ndarray:
    """Precision-weighted mean of every star's observations, ``(n_stars, 4)``.

    Dimensions that no observation of the star measures are left at 0: with
    ``sigma = 1e9`` their weight is zero, and they have no gradient in the loss.
    """
    x, xs, star = obs_dict["x"], obs_dict["xs"], obs_dict["star"]
    n_stars = int(obs_dict["n_stars"])
    w = np.where(xs < MISSING, 1.0 / np.maximum(xs, 1e-12) ** 2, 0.0).astype("f8")
    num, den = [], []
    for p in range(x.shape[1]):
        num.append(np.bincount(star, weights=w[:, p] * x[:, p], minlength=n_stars))
        den.append(np.bincount(star, weights=w[:, p], minlength=n_stars))
    return (np.stack(num, axis=1) / np.maximum(np.stack(den, axis=1), 1e-30)).astype("f4")


def held_out_stars(n_stars: int, val_frac: float, seed: int) -> np.ndarray:
    """Which stars are held out, as a boolean array over the star index.

    A random choice per *star*, not a cut on the sorted ids. A Gaia id encodes a
    HEALPix index, so the lowest ids are a patch of sky: taking the first
    ``val_frac`` of them makes the held-out set a region, not a sample. On the
    19-catalog file that region was LAMOST-heavy and DESI-poor by a factor of 7
    (1.4% of its observations against 9.7% of the training ones), RAVE-poor by
    100, and had no Pristine star at all, so the validation loss was measuring an
    easier subset of the sky.

    Args:
      n_stars: number of stars.
      val_frac: fraction of stars to hold out.
      seed: reproducible choice.
    """
    n_val = int(n_stars * val_frac)
    is_val = np.zeros(n_stars, dtype=bool)
    if n_val:
        is_val[np.random.default_rng(seed).permutation(n_stars)[:n_val]] = True
    return is_val


def validation_schedule(
    n_batches: int, n_train_obs: int, n_val_obs: int, val_every: Optional[int] = None
) -> Tuple[int, int, int]:
    """How often to validate, and with what batch size: ``(every, batch, batches)``.

    Validating every step at the training batch size costs about as much time as
    training; validating less often at a smaller batch instead makes ``chi2_val``
    an average over fewer observations than ``chi2``, so the two stop being
    comparable. So validate once every ``val_every`` steps -- by default the
    data-size ratio, which makes one epoch's validation calls a single pass over
    the held-out set -- and size the batch so that pass is exactly one pass:

    .. code::

        calls_per_epoch = batches_per_epoch // val_every
        val_batch       = ceil(n_val_obs / calls_per_epoch)

    Sizing the batch to the training value instead leaves a fixed tail that a
    cycling stream never shows: 52 batches of 65,536 cover 94.7% of 484,492, and
    the missing 5.3% is dropped every epoch.
    """
    if val_every is None:
        val_every = max(1, round(n_train_obs / max(1, n_val_obs)))
    val_every = max(1, int(val_every))
    calls_per_epoch = max(1, n_batches // val_every)
    val_batch = min(n_val_obs, -(-n_val_obs // calls_per_epoch))
    n_val_batches = -(-n_val_obs // val_batch)
    return val_every, val_batch, n_val_batches


def block_shuffled_rows(star: np.ndarray, rows: np.ndarray, rng) -> np.ndarray:
    """``rows`` reordered by a random permutation of their star, stars kept whole.

    ``rows`` must be sorted by star (true of the observation file) and a star's
    rows stay contiguous, so at most one star straddles a batch boundary.
    """
    if len(rows) == 0:
        return rows
    stars = star[rows]
    perm = rng.permutation(np.unique(stars))
    rank = np.empty(int(stars.max()) + 1, dtype="i8")
    rank[perm] = np.arange(len(perm))
    return rows[np.argsort(rank[stars], kind="stable")]


class _WarmupAndDecay(keras.optimizers.schedules.LearningRateSchedule):
    """``decay`` with a linear warmup from ``peak / warmup_steps`` in front of it.

    Adam normalises by the running gradient RMS, so its step is roughly ``lr``
    per coordinate no matter how large the gradient is: the shape of the early
    trajectory is set by the rate alone. Starting at the full rate and holding it
    for a whole segment is what let ``latent_gn_5h`` walk into a region where the
    maps were distorted enough to punish themselves further. The ramp is
    ``min``-ed against the decay so that it never raises the rate.

    This has to be a ``LearningRateSchedule`` and not a plain function: Keras
    calls a callable rate without arguments and only passes the step to objects
    of that type.
    """

    def __init__(self, decay, peak: float, warmup_steps: int):
        super().__init__()
        self.decay = decay
        self.peak = float(peak)
        self.warmup_steps = int(warmup_steps)

    def __call__(self, step):
        rate = self.decay(step)
        ramp = self.peak * (tf.cast(step, rate.dtype) + 1.0) / float(self.warmup_steps)
        return tf.minimum(rate, ramp)

    def get_config(self):
        config = super().get_config()
        config.update(decay=self.decay, peak=self.peak, warmup_steps=self.warmup_steps)
        return config


def _warmup_and_decay(decay, peak: float, warmup_steps: int):
    """``decay`` unchanged when there is no warmup, wrapped in the ramp otherwise."""
    if warmup_steps <= 0:
        return decay
    return _WarmupAndDecay(decay, peak, warmup_steps)


def _make_ctrl_dataset(arrays, batch_size: int, seed: int, n_repeat: Optional[int] = None):
    ds = tf.data.Dataset.from_tensor_slices(tuple(arrays))
    # A buffer far larger than the control catalogue is pure waste: at 256k
    # observations per step the old batch_size * 64 asked for 16.7M slots to
    # shuffle 2.9M points, and refilling it stalled for seconds every few steps.
    ds = ds.shuffle(buffer_size=min(batch_size * 64, 1 << 20), seed=seed)
    ds = ds.repeat(n_repeat)
    return ds.batch(batch_size, drop_remainder=True)


class LatentTrainer:
    """Holds the bijector, the free harmonized parameters and the data pipelines."""

    def __init__(self, obs_dict, ctrl_dict, cfg: LatentTrainConfig):
        if cfg.y_solver not in ("gauss_newton", "gradient"):
            raise ValueError(
                f"y_solver must be 'gauss_newton' or 'gradient', not {cfg.y_solver!r}"
            )
        self.cfg = cfg
        self.n_dim = obs_dict["x"].shape[1]
        self.n_cat = int(ctrl_dict["c_control"].max()) + 1

        tf.random.set_seed(cfg.seed)

        self._make_data(obs_dict, ctrl_dict)
        self.samples = catalog_samples(ctrl_dict, self.n_cat, cfg.n_plot)
        self.sample_valid = catalog_sample_valid(ctrl_dict, self.n_cat, cfg.n_plot)

        self.bij = ODEBijector(
            self.n_dim,
            self.n_cat,
            cfg.hidden_size,
            cfg.n_hidden,
            rtol=cfg.ode_rtol,
            atol=cfg.ode_atol,
            vector_network_kwargs=dict(use_residual=True, use_film=True, use_layer_norm=True),
        )
        self.variables = list(self.bij.trainable_variables)
        n_vars = sum(int(np.prod(v.shape)) for v in self.variables)
        log.info("%d variables in bijector model.", n_vars)

        self.y = tf.Variable(initial_y(obs_dict), name="latent_y", dtype=tf.float32)
        log.info(
            "%d free harmonized vectors (%.0f MB).",
            int(self.y.shape[0]),
            self.y.numpy().nbytes / 2**20,
        )
        # The held-out stars' y, fitted the same way and remembered between
        # validation calls; see ``_fit_y_val``.
        self.y_val = tf.Variable(
            initial_y(obs_dict), name="latent_y_val", dtype=tf.float32
        )

        # The learning-rate budget is set by the number of steps, so the data
        # pipeline must be built first.
        n = max(8, self.n_steps)
        k = max(1, min(int(cfg.lr_schedule_phases), n))
        if k == 1:  # a constant rate; the decay needs one boundary to exist
            boundaries, values = [n], [cfg.learning_rate, cfg.learning_rate]
        else:
            boundaries = [i * n // k for i in range(1, k)]
            values = [cfg.learning_rate / 2**i for i in range(k)]
        decay = keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
        warmup_steps = int(cfg.lr_warmup_frac * n)
        self.learning_rate = _warmup_and_decay(decay, cfg.learning_rate, warmup_steps)
        if warmup_steps:
            log.info(
                "Learning rate ramps to %.3g over the first %d steps (%d%% of the run).",
                cfg.learning_rate, warmup_steps, round(100 * cfg.lr_warmup_frac),
            )
        self.opt = keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.opt.build(self.variables)

        # Graph-compiled steps: ``train`` creates the dataset iterators and
        # these functions capture them, so a chunk of steps is one graph
        # execution (see ``_run_steps_graph``).
        self._it: Optional[tf.data.Iterator] = None
        self._val_it: Optional[tf.data.Iterator] = None
        self._run_steps = tf.function(self._run_steps_graph, reduce_retracing=True)
        self._val_step = tf.function(self._val_step_graph)

        self.to_harmonized = tf.function(
            self.bij._forward,
            input_signature=(
                tf.TensorSpec([None, self.n_dim], tf.float32),
                tf.TensorSpec([None], tf.int32),
            ),
        )

    # -- data ---------------------------------------------------------------
    def _make_data(self, obs_dict, ctrl_dict) -> None:
        cfg = self.cfg
        star = np.asarray(obs_dict["star"])
        n_stars = int(obs_dict["n_stars"])
        # A random choice of stars (see ``held_out_stars``), applied per
        # observation: every observation of a held-out star is held out.
        is_val_star = held_out_stars(n_stars, cfg.val_frac, cfg.seed)
        is_val = is_val_star[star]
        n_val_stars = int(is_val_star.sum())
        rng = np.random.default_rng(cfg.seed)

        # ``block_shuffles`` independent permutations of the star blocks, one after
        # another: shuffling once fixes both the order of the batches and the tail
        # that a ``drop_remainder`` cut removes, so the same stars sit in that tail
        # in every epoch. Concatenating N permutations spreads the cut over N sets.
        base_rows = np.flatnonzero(~is_val)
        train_rows = np.concatenate(
            [
                block_shuffled_rows(star, base_rows, rng)
                for _ in range(max(1, cfg.block_shuffles))
            ]
        )
        # The validation rows are shuffled in star blocks too, with their own
        # generator so that ``block_shuffles`` cannot change the validation
        # order. In id order a validation batch is a contiguous block of sorted
        # Gaia ids, i.e. a patch of sky: at fixed weights the seven batches of
        # ``models/_check_yval`` ranged from 5.2 to 8.8 in chi^2, five times the
        # offset the ``y_val`` change removed. Shuffled, each batch is a random
        # sample of the sky, while a cycle still covers every held-out
        # observation exactly once.
        val_rows = block_shuffled_rows(
            star, np.flatnonzero(is_val), np.random.default_rng([cfg.seed, 1])
        )

        if cfg.drop_remainder:
            n_batches = len(train_rows) // cfg.batch_size
        else:
            n_batches = -(-len(train_rows) // cfg.batch_size)  # keep the last partial batch
        if n_batches == 0:
            raise ValueError(
                f"there are no training observations (batch_size {cfg.batch_size})"
            )
        self.n_steps = n_batches * cfg.n_epochs
        if cfg.max_steps is not None:
            self.n_steps = min(self.n_steps, cfg.max_steps)

        ds = tf.data.Dataset.from_tensor_slices(train_rows.astype("i4"))
        ds = ds.batch(cfg.batch_size, drop_remainder=cfg.drop_remainder)
        ds = ds.repeat(cfg.n_epochs).shuffle(max(1, min(n_batches, 256)), seed=cfg.seed)

        ctrl_keys = ("x_control", "xe_control", "c_control")
        n_val = int(len(ctrl_dict["x_control"]) * cfg.val_frac)
        # The control points repeat for ever; zipping the two streams gives one
        # iterator that yields a batch of observations and a batch of control
        # points per step, which is what the graph loop consumes.
        self.data = tf.data.Dataset.zip(
            (
                ds,
                _make_ctrl_dataset(
                    [ctrl_dict[k][n_val:] for k in ctrl_keys], cfg.batch_size, cfg.seed
                ),
            )
        ).prefetch(2)

        # Validation: sweep the whole held-out set once per epoch, in batches as
        # close to the training size as the arithmetic allows.
        #
        # Validating every step at the training batch size would spend about as
        # much time on validation as on training; validating less often at a
        # smaller batch instead would make ``chi2_val`` a noisier and less
        # comparable number than ``chi2``. So validate once every ``val_every``
        # steps, where the default is the data-size ratio, and size the batch so
        # that the calls an epoch makes cover the held-out set exactly:
        #
        #     calls/epoch = batches/epoch // val_every
        #     val_batch   = ceil(n_val_obs / calls/epoch)
        #
        # With 52 batches and a ratio of 7 that is 7 calls of 69,213 observations,
        # which covers all 484,492 held-out observations once per epoch. Sizing it
        # "round" instead (65,536) would leave a fixed 5% tail that a cycling
        # stream never shows.
        self.val_data = None
        if len(val_rows) > 0 and n_val > 0:
            self.val_every, val_batch, n_val_batches = validation_schedule(
                n_batches, len(train_rows), len(val_rows), cfg.val_every
            )
            val_repeats = -(-self.n_steps // n_val_batches) + 1  # one to spare

            # ``batch`` drops the short tail, so a cycle with a partial last batch
            # would silently lose it and the coverage above would not hold. Wrap a
            # few observations from the front of the set instead: at most
            # ``val_batch - 1`` of them are then seen twice, which is nothing next
            # to a mean over 484,492.
            pad = (-len(val_rows)) % val_batch
            val_row_batches = (
                np.concatenate([val_rows, val_rows[:pad]]) if pad else val_rows
            )

            # The control points are a penalty term, not a per-star quantity, so
            # their count per call is free. Their stream is unbounded, so that
            # ``zip`` (which stops at the shorter stream) is ended by the
            # observations and the held-out set is covered in full.
            ctrl_batch = min(val_batch, n_val)
            self.val_data = tf.data.Dataset.zip(
                (
                    tf.data.Dataset.from_tensor_slices(val_row_batches.astype("i4"))
                    .batch(val_batch, drop_remainder=True)
                    .repeat(val_repeats),
                    _make_ctrl_dataset(
                        [ctrl_dict[k][:n_val] for k in ctrl_keys],
                        ctrl_batch,
                        cfg.seed,
                        None,  # for ever
                    ),
                )
            ).prefetch(2)
            log.info(
                "validation every %d steps: %d calls of %d observations per epoch, "
                "%d batches covering all %d held-out observations",
                self.val_every, n_batches // max(1, self.val_every), val_batch,
                n_val_batches, len(val_rows),
            )
        else:
            self.val_every = 0
            log.warning("no held-out stars or control points; validation is off")

        # GPU copies of the observation columns; a batch is gathered by row index
        self.x_obs = tf.constant(np.asarray(obs_dict["x"]))
        self.xs_obs = tf.constant(np.asarray(obs_dict["xs"]))
        self.c_obs = tf.constant(np.asarray(obs_dict["c"], dtype="int32"))
        self.star_obs = tf.constant(star)

        # The rows of ``y`` follow the sorted unique ids (the observations are
        # sorted by id, so the star index counts them in order).
        self.ids = np.unique(np.asarray(obs_dict["id"]))
        if len(self.ids) != n_stars:
            raise ValueError(
                f"the star index has {n_stars} entries but there are {len(self.ids)} ids"
            )

        # Held-out stars, for the agreement shown with each progress figure
        self.eval_pairs = observation_pairs(
            obs_dict,
            max_stars=cfg.agreement_n_stars,
            seed=cfg.seed,
            stars=np.flatnonzero(is_val_star),
        )
        log.info(
            "%d steps expected (%d batches of %d observations per epoch; each epoch "
            "is %d block permutations = %d passes over the data); "
            "%d held-out stars for the agreement.",
            self.n_steps, n_batches, cfg.batch_size,
            max(1, cfg.block_shuffles), max(1, cfg.block_shuffles), n_val_stars,
        )

    # -- loss ---------------------------------------------------------------
    def _normal_equations(self, x, xs, x_pred, jac, chi2_obs, star):
        """Per-star ``(A, b)`` of the Gauss-Newton step in ``y``.

        With ``C = diag(sigma^2)`` diagonal, the normal equations of one
        observation are ``J^T W J`` and ``J^T W (x - x_pred)``, ``W =
        w / sigma^2`` elementwise. The weight ``w`` is the curvature of the
        softened chi^2 the loss actually uses: the loss is
        ``S * arcsinh(chi2 / S)``, whose derivative with respect to ``chi2`` is
        ``1 / sqrt(1 + (chi2 / S)^2)``, so weighting by it makes the step the
        exact minimiser of that loss and not merely of a quadratic stand-in.

        Returns the summed matrices of shape ``(n_star, n_dim, n_dim)`` and
        vectors of shape ``(n_star, n_dim)``, and the ``tf.unique`` of the star
        index.
        """
        cfg = self.cfg
        weight = (1.0 / tf.sqrt(1.0 + (chi2_obs / cfg.chi2_turnover) ** 2))[:, None]
        weight = weight / xs**2
        dx = x - x_pred
        A = tf.einsum("npa,np,npb->nab", jac, weight, jac)
        b = tf.einsum("npa,np,np->na", jac, weight, dx)
        uniq, inv = tf.unique(star)
        n = tf.size(uniq)
        return (
            tf.math.unsorted_segment_sum(A, inv, n),
            tf.math.unsorted_segment_sum(b, inv, n),
            uniq,
        )

    def _solve_y(self, A, b, return_scale=False):
        """Solve the per-star normal equations, with a ridge for flat directions.

        A parameter that none of a star's catalogs measures has ``sigma = 1e9``,
        so its row and column of ``A`` are ~1e-18 and the system is singular; the
        ridge makes the step in that direction ~0 instead of undefined, which is
        the same outcome the ``den > 1e-6`` test gives the gradient solver. It is
        diagonal on purpose: an off-diagonal term would leak the other
        components' steps into a direction they say nothing about. It also damps
        the nearly-flat directions, where the full step is hundreds of sigma.

        The result is then passed through the trust region, and the fraction of
        the step that radii allowed is returned as well when ``return_scale``.
        """
        n_dim = tf.shape(b)[-1]
        scale = tf.reduce_mean(tf.linalg.diag_part(A), axis=1)[:, None, None]
        ridge = (self.cfg.y_ridge * scale + 1e-8) * tf.eye(n_dim, dtype=A.dtype)
        step = tf.linalg.solve(A + ridge, b[:, :, None])[:, :, 0]
        step, allowed = self._trust_region(step)
        return (step, allowed) if return_scale else step

    def _trust_region(self, step):
        """Scale whole steps down to the trust radius; returns ``(step, allowed)``.

        A Gauss-Newton step is a full Newton step, and where a star's normal
        matrix is nearly singular it can be arbitrarily large. A star flung like
        that is a star whose `y` is being fitted against a map that has just
        changed under it; the next sweep of the map then trains on the flung
        value, and the two can push each other away over a few steps.
        ``latent_gn_5h`` went from chi^2 4.1 to 105 in fifteen steps, which is too
        large for one star to cause (the loss is a mean over 65k observations and
        the arcsinh softening caps a single star's contribution), but not too
        large for a self-feeding ``y``-map loop.

        So no star may move by more than ``y_trust_abs`` in any one parameter. The
        step is *scaled*, never clipped component by component: every component
        keeps its share, so the scaled step is still the exact minimiser of the
        quadratic along that line, and the normal-equation solve is untouched in
        the (overwhelmingly common) case where nothing binds.
        """
        cfg = self.cfg
        if not cfg.y_trust_abs:
            return step, tf.ones_like(step[:, 0])
        biggest = tf.reduce_max(tf.abs(step), axis=1)
        allowed = tf.minimum(1.0, cfg.y_trust_abs / tf.maximum(biggest, 1e-12))
        return step * allowed[:, None], allowed

    def _fit_y_val(self, x, xs, c, star) -> tf.Tensor:
        """One Gauss-Newton sweep of the held-out stars' ``y``, kept between calls.

        The training ``y`` is re-solved once per epoch and so sits at its running
        fixed point. A validation ``y`` refitted from scratch on every call would
        instead be one sweep from the precision-weighted mean -- a cold start --
        and would report a chi^2 about 0.3 higher than the training one for that
        reason alone: on the finished ``latent_gn_dr1logg_1h`` the held-out stars
        reach 4.32 after one sweep but 4.03 at the fixed point, against 3.90 for
        the trained ``y``. So keep a second array and update it the same way: one
        sweep, once per epoch per star, on the held-out batch. The two numbers are
        then the same quantity.

        Fitted only on held-out data, per star, in sample: no leakage into the map,
        and no effect on it.
        """
        cfg = self.cfg
        x_pred, _, jac = self.bij.inverse_with_s_and_jac(
            tf.gather(self.y_val, star), c=c
        )
        chi2_obs = tf.reduce_sum((x - x_pred) ** 2 / xs**2, axis=1)
        A, b, uniq = self._normal_equations(x, xs, x_pred, jac, chi2_obs, star)
        step = self._solve_y(A, b)
        if cfg.y_learning_rate:
            self.y_val.scatter_nd_add(
                tf.expand_dims(uniq, 1), cfg.y_learning_rate * step
            )
        return tf.gather(self.y_val, star)

    def _gauss_newton_y(self, x, xs, c, star) -> tf.Tensor:
        """Move ``y`` to the Gauss-Newton optimum of the batch's chi^2.

        Called before the map's gradient step, so the map trains against the
        harmonized values it will actually be scored with. The batch holds the
        complete observation set of nearly every star in it (the rows are
        shuffled in star blocks), so one step is the full profile optimum for
        those stars rather than a partial one; the measurement on a finished run
        showed that a second step moves the mean softened chi^2 by only 2%,
        hence one step here.

        Returns the mean fraction of the step the trust region allowed, so that
        the progress figure shows whether the caps are binding.
        """
        cfg = self.cfg
        if not cfg.y_learning_rate:
            return tf.constant(1.0)
        x_pred, _, jac = self.bij.inverse_with_s_and_jac(tf.gather(self.y, star), c=c)
        chi2_obs = tf.reduce_sum((x - x_pred) ** 2 / xs**2, axis=1)
        A, b, uniq = self._normal_equations(x, xs, x_pred, jac, chi2_obs, star)
        step, allowed = self._solve_y(A, b, return_scale=True)
        self.y.scatter_nd_add(tf.expand_dims(uniq, 1), cfg.y_learning_rate * step)
        return tf.reduce_mean(allowed)

    def _update_y(self, grad_y, xs, star) -> None:
        """One step of ``y`` along the exact gradient of the loss.

        ``grad_y`` is the gradient of the batch loss with respect to each
        observation's harmonized vector. It comes out of the same tape as the
        map's gradients, so it contains the Jacobian of the inverse map and is
        the true direction of steepest descent in ``y`` even where ``f_c^-1``
        is far from the identity.

        The per-star step is divided by the star's total precision
        ``sum_obs 1/sigma^2``: that is the Gauss-Newton curvature of the chi^2
        for a map close to the identity, which the control-point penalties
        enforce (they hold ``f_c`` near the identity in any region the catalog
        does not measure). Any positive divisor keeps the step a descent
        direction, so an inaccurate curvature only costs speed. A dimension
        that none of the star's observations measures has ``sigma = 1e9``;
        its weight is ~1e-18, which fails the ``den > 1e-6`` test and is left
        alone instead of being divided by ~1e-18.
        """
        cfg = self.cfg
        if not cfg.y_learning_rate:
            return
        w = 1.0 / xs**2
        uniq, inv = tf.unique(star)
        n = tf.size(uniq)
        den = tf.math.unsorted_segment_sum(w, inv, n)
        num = tf.math.unsorted_segment_sum(grad_y, inv, n)
        delta = tf.where(den > 1e-6, num / (den + 1e-30), tf.zeros_like(num))
        self.y.scatter_nd_sub(tf.expand_dims(uniq, 1), cfg.y_learning_rate * delta)

    def _run_steps_graph(self, n_steps, freeze_list, fit_y):
        """``n_steps`` optimizer steps in one graph.

        AutoGraph turns the loop into a ``tf.while_loop`` around the dataset
        iterator and the optimizer, so each call advances the state by
        ``n_steps`` without returning to Python. ``freeze_list`` and ``fit_y`` are
        Python values: changing either retraces once, which is all a run ever
        needs (``fit_y`` flips once, at the end of the pre-training phase).
        """
        out = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
        for step in tf.range(n_steps):
            obs_idx, (x_ctrl, xe_ctrl, c_ctrl) = self._it.get_next()
            res = self._grad_step(
                obs_idx,
                x_ctrl,
                xe_ctrl,
                c_ctrl,
                apply_grads=True,
                freeze_list=freeze_list,
                fit_y=fit_y,
            )
            out = out.write(step, tf.stack([res[k] for k in METRIC_KEYS]))
        return out.stack()

    def _val_step_graph(self, fit_y):
        """One validation step: the loss on held-out stars, nothing updated."""
        obs_idx, (x_ctrl, xe_ctrl, c_ctrl) = self._val_it.get_next()
        return self._grad_step(
            obs_idx, x_ctrl, xe_ctrl, c_ctrl,
            apply_grads=False, freeze_list=(), fit_y=fit_y,
        )

    def _grad_step(
        self,
        obs_idx,
        x_ctrl,
        xe_ctrl,
        c_ctrl,
        apply_grads=True,
        freeze_list=(),
        fit_y=True,
    ):
        cfg = self.cfg
        bij = self.bij

        vars_filtered = [
            v for v in self.variables if not any(s in v.name for s in freeze_list)
        ]

        # Scatter the control points, and create an offset copy of each, which is
        # used to measure stretching/contraction of space
        x_ctrl = x_ctrl + cfg.sigma_ctrl * tf.random.normal(tf.shape(x_ctrl))
        x1_ctrl = x_ctrl + cfg.eps_div * tf.random.normal(tf.shape(x_ctrl))

        x = tf.gather(self.x_obs, obs_idx)
        xs = tf.gather(self.xs_obs, obs_idx)
        c = tf.gather(self.c_obs, obs_idx)
        star = tf.gather(self.star_obs, obs_idx)

        # The Gauss-Newton solver needs the Jacobian and no gradient with respect
        # to y; the gradient solver needs the opposite, so ``y`` is watched only
        # for the latter. Either way, nothing touches ``y`` while it is being
        # held at its initial guess.
        gauss_newton = apply_grads and fit_y and cfg.y_solver == "gauss_newton"
        y_gradient = apply_grads and fit_y and cfg.y_solver == "gradient"

        # ``y`` is fitted first, outside the tape: it is a separate block of the
        # optimisation, and its solve carries a Jacobian whose ops nothing needs
        # to differentiate, so keeping it out of the tape keeps it out of the
        # graph's gradient bookkeeping as well. The map then takes its step at
        # the fitted ``y`` rather than at the one about to be replaced.
        trust = tf.constant(1.0)
        if gauss_newton:
            trust = self._gauss_newton_y(x, xs, c, star)
        if apply_grads or not fit_y:
            # A pre-trained run holds every star at its initial guess, held-out or
            # not, so the validation loss measures the objective being trained.
            y = tf.gather(self.y, star)
        else:
            y = self._fit_y_val(x, xs, c, star)

        with tf.GradientTape(watch_accessed_variables=False) as g:
            g.watch(vars_filtered)
            if y_gradient:
                # The gradient with respect to y comes from the same tape: one
                # backward pass for the map and for the free parameters.
                g.watch(y)

            # Map every observation's harmonized value back into its catalog
            x_pred, s = bij.inverse_with_s(y, c=c)

            # chi^2 of each observation. An unmeasured parameter has sigma = 1e9
            # and contributes ~1e-18, so no mask is needed.
            chi2_obs = tf.reduce_sum((x - x_pred) ** 2 / xs**2, axis=1)
            chi2 = cfg.chi2_turnover * arcsinh_of_exp(
                tf.math.log(chi2_obs + 1e-30) - tf.math.log(cfg.chi2_turnover)
            )
            chi2 = tf.reduce_mean(chi2)

            # Path-length penalty of the observation transformations
            penalty_obs = cfg.l1_path * tf.reduce_mean(tf.abs(s)) + cfg.l2_path * tf.reduce_mean(
                s**2
            )

            # Map control points (and their offset copies) to the common space
            x_ctrl_pred, s_ctrl = bij.forward_with_s(x_ctrl, c=c_ctrl)
            x1_ctrl_pred, _ = bij.forward_with_s(x1_ctrl, c=c_ctrl)

            # Penalties on control-point transformations
            length1_ctrl = tf.reduce_mean(tf.abs((x_ctrl_pred - x_ctrl) / xe_ctrl))
            length2_ctrl = tf.reduce_mean(((x_ctrl_pred - x_ctrl) / xe_ctrl) ** 2)
            penalty_ctrl = (
                cfg.l1_total * length1_ctrl
                + cfg.l2_total * length2_ctrl
                + cfg.l1_path * tf.reduce_mean(tf.abs(s_ctrl))
                + cfg.l2_path * tf.reduce_mean(s_ctrl**2)
            )

            # Penalty on stretching/contraction of space
            length_div = tf.reduce_sum((x1_ctrl - x_ctrl) ** 2, axis=1)
            length_div_p = tf.reduce_sum((x1_ctrl_pred - x_ctrl_pred) ** 2, axis=1)
            eps = 1e-2 * cfg.eps_div
            penalty_div = cfg.weight_div * tf.reduce_mean(
                tf.math.log((length_div + eps) / (length_div_p + eps)) ** 2
            )

            # Weight decay on the dense-layer weights
            penalty_l2 = cfg.l2_weights * tf.add_n(
                [tf.nn.l2_loss(v) for v in vars_filtered if "w:" in v.name]
            )

            loss = chi2 + penalty_ctrl + penalty_div + penalty_l2
            if cfg.obs_path_in_loss:
                loss = loss + penalty_obs

        res = {
            "loss": loss,
            "chi2": chi2,
            "penalty_obs": penalty_obs,
            "penalty_ctrl": penalty_ctrl,
            "penalty_div": penalty_div,
            "penalty_l2": penalty_l2,
            "y_trust": trust,
        }

        if apply_grads:
            if y_gradient:
                *grads, grad_y = g.gradient(loss, vars_filtered + [y])
            else:
                grads = g.gradient(loss, vars_filtered)
            grads, norm = tf.clip_by_global_norm(grads, cfg.clip_norm)
            self.opt.apply_gradients(zip(grads, vars_filtered))
            if y_gradient and grad_y is not None:
                self._update_y(grad_y, xs, star)
            res["norm"] = norm

        return res

    # -- resumable state ----------------------------------------------------
    def _manager(self, run_dir: Path) -> tf.train.CheckpointManager:
        ckpt = tf.train.Checkpoint(
            bij=self.bij, latent_y=self.y, latent_y_val=self.y_val
        )
        return tf.train.CheckpointManager(ckpt, str(Path(run_dir) / "checkpoints"), max_to_keep=3)

    def save_state(
        self, run_dir: Path, step: int, metrics, n_figs: int, elapsed: float,
        agreement, complete: bool = False,
    ) -> None:
        """Write everything needed to resume: model, y, y_val, optimizer, metrics, progress.

        The model and the two free harmonized parameter arrays go into a
        ``tf.train.Checkpoint`` numbered by step; the rest into
        ``state-<step>.npz`` next to it, written first so that a checkpoint
        never lacks its state.
        """
        ckpt_dir = Path(run_dir) / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        meta = dict(
            step=step, n_figs=n_figs, elapsed=elapsed, complete=complete,
            metrics=metrics, agreement=agreement,
        )
        arrays = {f"opt_{i}": np.asarray(v) for i, v in enumerate(self.opt.variables)}
        tmp = ckpt_dir / "state.tmp"
        with open(tmp, "wb") as f:
            np.savez(f, meta=np.array(json.dumps(meta)), **arrays)
        os.replace(tmp, ckpt_dir / f"state-{step}.npz")

        manager = self._manager(run_dir)
        if step != self._last_saved_step:
            manager.save(checkpoint_number=step)
            self._last_saved_step = step
        keep = {int(p.rsplit("-", 1)[1]) for p in manager.checkpoints}
        for f in ckpt_dir.glob("state-*.npz"):
            if int(f.stem.split("-")[1]) not in keep:
                f.unlink()

    def load_state(self, run_dir: Path) -> Optional[dict]:
        """Restore the latest complete checkpoint; return its metadata (or None)."""
        manager = self._manager(run_dir)
        for path in reversed(manager.checkpoints):
            step = int(path.rsplit("-", 1)[1])
            state_path = Path(run_dir) / "checkpoints" / f"state-{step}.npz"
            if not state_path.exists():
                continue
            (tf.train.Checkpoint(bij=self.bij, latent_y=self.y, latent_y_val=self.y_val)
             .restore(path).assert_existing_objects_matched())
            data = np.load(state_path)
            meta = json.loads(str(data["meta"]))
            opt_vars = self.opt.variables
            n_saved = sum(k.startswith("opt_") for k in data.files)
            if n_saved != len(opt_vars):
                raise ValueError(
                    f"Checkpoint has {n_saved} optimizer variables, expected {len(opt_vars)}."
                )
            for i, v in enumerate(opt_vars):
                v.assign(data[f"opt_{i}"])
            self._last_saved_step = step
            log.info("Resumed from %s (step %d)", path, step)
            return meta
        return None

    # -- training loop ------------------------------------------------------
    def train(self, run_dir: Path, resume: bool = False) -> Dict[str, list]:
        cfg = self.cfg
        run_dir = Path(run_dir)
        fig_dir = run_dir / "progress"
        fig_dir.mkdir(parents=True, exist_ok=True)

        labels = registry.labels()[: self.n_cat]
        colors = catalog_colors(self.n_cat)
        self._last_saved_step = -1

        metrics: Dict[str, list] = {"step_val": []}
        agreement: list = []
        self.agreement = agreement
        n_figs = 0
        elapsed_prev = 0.0
        start = 0
        if resume:
            meta = self.load_state(run_dir)
            if meta is None:
                raise FileNotFoundError(f"No checkpoint to resume from in {run_dir}/checkpoints")
            metrics, agreement = meta["metrics"], meta["agreement"]
            self.agreement = agreement
            n_figs, elapsed_prev, start = meta["n_figs"], meta["elapsed"], meta["step"]
            tf.random.set_seed(cfg.seed + start)

        t_start = time.monotonic() - elapsed_prev
        t_last_plot = t_last_ckpt = time.monotonic()
        step_last_plot = start

        def checkpoint(step, complete=False):
            nonlocal t_last_ckpt
            self.save_state(
                run_dir, step, metrics, n_figs, time.monotonic() - t_start, agreement, complete
            )
            t_last_ckpt = time.monotonic()

        def make_figure(step):
            nonlocal n_figs, t_last_plot, step_last_plot
            elapsed = time.monotonic() - t_start
            path = fig_dir / f"progress_{n_figs + 1:04d}.png"
            summary = quick_agreement(self.to_harmonized, self.eval_pairs)
            agreement.append({"step": step, "summary": summary})
            log.info("Catalog agreement at step %d (held-out stars):\n%s",
                     step, format_summary(summary))
            ok = [s for s in summary if s["n_pairs"]]
            title = (
                f"step {step:,} / {self.n_steps:,}  ({elapsed/60:.1f} min)\n"
                "mean over catalog pairs, catalog -> harmonized\n"
                "|offset|: " + ", ".join(
                    f"{PARAM_SHORT[s['param']]} {s['abs_bias_orig']:.3f}->{s['abs_bias_harm']:.3f}"
                    for s in ok
                )
                + "\nscatter: " + ", ".join(
                    f"{PARAM_SHORT[s['param']]} {s['sigma_orig']:.3f}->{s['sigma_harm']:.3f}"
                    for s in ok
                )
            )
            save_progress_figure(
                path, metrics, self.bij, self.samples, labels, colors,
                title=title, valid=self.sample_valid, penalties=LATENT_PENALTIES,
            )
            n_figs += 1
            checkpoint(step)
            t_last_plot = time.monotonic()
            step_last_plot = step
            return path

        freeze_step = None
        if cfg.freeze_after_frac is not None:
            freeze_step = int(cfg.freeze_after_frac * self.n_steps)
        freeze_list: Tuple[str, ...] = ()
        if freeze_step is not None and start > freeze_step:
            freeze_list = cfg.freeze_vars

        # The iterators are resources captured by the graph functions; resumes
        # skip the batches (and the validation steps) already done.
        self._it = iter(self.data.skip(start))
        self._val_it = None
        if self.val_data is not None:
            self._val_it = iter(self.val_data.skip(len(metrics["step_val"])))
        step_last_val = metrics["step_val"][-1] if metrics["step_val"] else 0

        def _on_sigterm(signum, frame):
            raise KeyboardInterrupt

        old_handler = None
        if threading.current_thread() is threading.main_thread():
            old_handler = signal.signal(signal.SIGTERM, _on_sigterm)

        def report(step):
            avg = {k: np.mean(v[-16:]) for k, v in metrics.items() if k != "step_val"}
            pbar.set_postfix(
                {
                    "loss": avg["loss"],
                    "chi2": avg["chi2"],
                    "plty_o": avg["penalty_obs"],
                    "plty_c": avg["penalty_ctrl"],
                    "plty_d": avg["penalty_div"],
                    "plty_l2": avg["penalty_l2"],
                    "norm": avg["norm"],
                    "lr": float(self.learning_rate(step)),
                },
                refresh=False,
            )

        # Steps per graph call. A chunk costs one host round trip and is not
        # interruptible, so it is kept near a couple of seconds.
        chunk = 4
        done = start
        pbar = tqdm(total=self.n_steps, initial=start, desc="training")
        try:
            while done < self.n_steps:
                if freeze_step is not None and done >= freeze_step and not freeze_list:
                    freeze_list = cfg.freeze_vars
                    log.info("Step %d: freezing variables matching %s", done, freeze_list)

                n = min(chunk, self.n_steps - done)
                # A chunk cannot straddle the end of the pre-training phase: the
                # boundary is where ``fit_y`` flips, and a chunk is one graph
                # execution with a single value of it.
                if done < cfg.pretrain_steps:
                    n = min(n, cfg.pretrain_steps - done)
                fit_y = done >= cfg.pretrain_steps
                t0 = time.monotonic()
                block = self._run_steps(tf.constant(n, tf.int32), freeze_list, fit_y).numpy()
                dt = time.monotonic() - t0

                for key, column in zip(METRIC_KEYS, block.T):
                    metrics.setdefault(key, []).extend(np.asarray(column, "f8").tolist())
                done += n
                pbar.update(n)
                if done == cfg.pretrain_steps:
                    log.info(
                        "Step %d: pre-training done, y starts being fitted (%s)",
                        done, cfg.y_solver,
                    )

                if self._val_it is not None and done - step_last_val >= self.val_every:
                    try:
                        res = self._val_step(fit_y)
                    except tf.errors.OutOfRangeError:
                        # Never lose a long run to a diagnostic: drop validation
                        # and carry on with the training stream.
                        log.warning(
                            "Validation stream exhausted at step %d; continuing "
                            "without validation from here on.", done,
                        )
                        self._val_it, res = None, None
                    if res is not None:
                        for k, v in res.items():
                            metrics.setdefault(k + "_val", []).append(float(v.numpy()))
                        metrics["step_val"].append(done)
                    step_last_val = done

                report(done)
                if dt > 0:
                    chunk = int(np.clip(round(n * 2.0 / dt), 1, 1024))

                due = False
                if cfg.plot_every_seconds is not None:
                    due |= time.monotonic() > t_last_plot + cfg.plot_every_seconds
                if cfg.plot_every_steps is not None:
                    due |= done - step_last_plot >= cfg.plot_every_steps
                if due and len(metrics["loss"]) > 1:
                    pbar.write(f"Wrote {make_figure(done)}")
                elif (
                    cfg.checkpoint_every_seconds is not None
                    and time.monotonic() > t_last_ckpt + cfg.checkpoint_every_seconds
                ):
                    checkpoint(done)
        except KeyboardInterrupt:
            done = len(metrics["loss"])
            pbar.close()
            checkpoint(done)
            log.warning(
                "Interrupted at step %d; resume with --resume --run-dir %s", done, run_dir
            )
            raise
        finally:
            if old_handler is not None:
                signal.signal(signal.SIGTERM, old_handler)

        pbar.close()
        done = len(metrics["loss"])
        if done > 1 and step_last_plot != done:
            log.info("Wrote %s", make_figure(done))
        checkpoint(done, complete=True)
        return metrics

    def save(self, run_dir: Path, metrics) -> None:
        """Write the SavedModel (``to_harmonized``/``from_harmonized``), ``y`` and the metrics."""
        run_dir = Path(run_dir)
        bij, n_dim = self.bij, self.n_dim

        bij.to_harmonized = tf.function(
            func=bij._forward,
            input_signature=(
                tf.TensorSpec([None, n_dim], tf.float32),  # x
                tf.TensorSpec([None], tf.int32),  # c
            ),
        )
        bij.from_harmonized = tf.function(
            func=bij._inverse,
            input_signature=(
                tf.TensorSpec([None, n_dim], tf.float32),
                tf.TensorSpec([None], tf.int32),
            ),
        )
        tf.saved_model.save(bij, str(run_dir / "bijector"))

        # The free harmonized parameters are training state, not part of the map;
        # kept for the evaluation and for reference.
        np.savez_compressed(
            run_dir / "latent_y.npz",
            id=self.ids,
            y=self.y.numpy(),
            y_val=self.y_val.numpy(),
        )
        with open(run_dir / "metrics_history.json", "w") as f:
            json.dump(metrics, f)


# --- Driver ---------------------------------------------------------------------
def _config_from_json(d: dict) -> dict:
    d = dict(d)
    if "freeze_vars" in d:
        d["freeze_vars"] = tuple(d["freeze_vars"])
    return d


def _check_resumable(saved: dict, cfg: LatentTrainConfig) -> None:
    """Refuse to resume with settings that would change the run."""
    old = dataclasses.asdict(LatentTrainConfig(**_config_from_json(saved["config"])))
    new = dataclasses.asdict(cfg)
    changed = [k for k in new if k not in RESUME_MUTABLE and new[k] != old[k]]
    if changed:
        raise ValueError(
            "Cannot change these settings when resuming: "
            + ", ".join(f"{k} ({old[k]!r} -> {new[k]!r})" for k in changed)
        )


def run_training(
    cfg: LatentTrainConfig,
    run_dir: Path,
    observations_path: Optional[Path] = None,
    resume: bool = False,
) -> Dict[str, list]:
    """Train a bijector with free per-star parameters; write everything to ``run_dir``."""
    run_dir = Path(run_dir)
    cfg_path = run_dir / "config.json"
    if not resume and (run_dir / "checkpoints").exists():
        raise FileExistsError(
            f"{run_dir} already contains a run. Use --resume to continue it, "
            "or choose a different --run-name."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    enable_memory_growth()

    obs_dict, ctrl_dict = load_observations(observations_path)
    log.info(
        "%d observations of %d stars, %d control points",
        len(obs_dict["x"]), obs_dict["n_stars"], len(ctrl_dict["x_control"]),
    )

    sizes = dict(
        n_observations=len(obs_dict["x"]),
        n_stars=int(obs_dict["n_stars"]),
        n_control=len(ctrl_dict["x_control"]),
    )
    if resume:
        with open(cfg_path) as f:
            snapshot = json.load(f)
        for key, value in sizes.items():
            if snapshot.get(key) != value:
                raise ValueError(
                    "The observation file differs from the one the run was started "
                    f"with ({key}: {snapshot.get(key)} -> {value})."
                )
        _check_resumable(snapshot, cfg)
        snapshot["config"] = dataclasses.asdict(cfg)
        snapshot.setdefault("resumed", []).append(
            datetime.now().isoformat(timespec="seconds")
        )
    else:
        snapshot = {
            "config": dataclasses.asdict(cfg),
            "loss": "latent_y",
            "observations_path": str(observations_path or config.OBSERVATIONS_PATH),
            "observations_attrs": read_attrs(observations_path),
            **sizes,
            "catalogs": list(registry.keys()[: int(ctrl_dict["c_control"].max()) + 1]),
            "tensorflow": tf.__version__,
            "started": datetime.now().isoformat(timespec="seconds"),
        }
    with open(cfg_path, "w") as f:
        json.dump(snapshot, f, indent=2)

    trainer = LatentTrainer(obs_dict, ctrl_dict, cfg)
    metrics = trainer.train(run_dir, resume=resume)
    trainer.save(run_dir, metrics)
    log.info("Saved model to %s", run_dir)

    with open(run_dir / "agreement_history.json", "w") as f:
        json.dump(trainer.agreement, f)
    evaluate_bijector(
        trainer.to_harmonized, trainer.eval_pairs, ctrl_dict, run_dir / "evaluation",
        val_frac=1.0,
    )
    return metrics


def _add_config_args(parser: argparse.ArgumentParser) -> None:
    defaults = LatentTrainConfig()
    for f in dataclasses.fields(LatentTrainConfig):
        default = getattr(defaults, f.name)
        flag = "--" + f.name.replace("_", "-")
        if isinstance(default, bool):
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=None)
        elif isinstance(default, tuple):
            parser.add_argument(flag, nargs="*", default=None)
        elif f.name in ("max_steps", "plot_every_steps", "val_every"):
            parser.add_argument(flag, type=int, default=None)
        elif f.name in ("freeze_after_frac", "plot_every_seconds", "checkpoint_every_seconds"):
            parser.add_argument(flag, type=float, default=None)
        else:
            parser.add_argument(flag, type=type(default), default=None)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="astrotune-train-latent",
        description="Train the Astrotune bijector with one free harmonized "
        "parameter vector per star (the observation-based loss). Every "
        "LatentTrainConfig field is available as an option.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="output directory name under models/ (default: timestamp)",
    )
    parser.add_argument("--run-dir", type=Path, default=None, help="explicit output directory")
    parser.add_argument(
        "--observations-path", type=Path, default=None,
        help="observations_and_ctrl_points.h5 (default: config.OBSERVATIONS_PATH)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue the run in --run-dir/--run-name from its latest checkpoint. The "
        "saved configuration is reused; only plotting/checkpoint intervals may be changed.",
    )
    _add_config_args(parser)
    args = parser.parse_args(argv)

    overrides = {
        f.name: getattr(args, f.name)
        for f in dataclasses.fields(LatentTrainConfig)
        if getattr(args, f.name) is not None
    }
    if "freeze_vars" in overrides:
        overrides["freeze_vars"] = tuple(overrides["freeze_vars"])

    if args.run_dir is None and args.run_name is None:
        if args.resume:
            parser.error("--resume needs --run-dir or --run-name")
        args.run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = args.run_dir or config.MODELS_DIR / args.run_name

    if args.resume:
        with open(run_dir / "config.json") as f:
            saved = _config_from_json(json.load(f)["config"])
        cfg = LatentTrainConfig(**{**saved, **overrides})
    else:
        cfg = LatentTrainConfig(**overrides)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_training(cfg, run_dir, observations_path=args.observations_path, resume=args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
