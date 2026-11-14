"""The delta model: extra (diagonal) variance for every catalog entry.

``ln_delta_cov(x, ln_sigma_x, catalog) -> (4,)`` is the natural log of the
variance added, in the *harmonized* space, to the propagated covariance of an
entry. The inputs are the entry's own parameters (the catalog's coordinates,
e.g. ``5 ln(Teff/5000 K)``), the log of its reported uncertainties, and its
catalog (through FiLM, as in the bijector). The network is unchanged from the
pair-based version: only the likelihood it is fitted with has changed.

The likelihood is the one of the latent-``y`` paradigm. A star has entries
``k = 1..K`` in different catalogs, all measurements of one unknown value ``z``:

    y_k = z + e_k,   e_k ~ N(0, S_k),   S_k = C_k + diag(exp(ln_delta_k)),

with the ``e_k`` independent given ``z``. Marginalising ``z`` with a flat prior
gives, per star,

    -2 ln p = sum_k d_k^T S_k^-1 d_k + ln det A + sum_k ln det S_k + n_res ln 2 pi

with ``d_k = y_k - z``, ``A = sum_k S_k^-1`` and ``z = A^-1 sum_k S_k^-1 y_k``
(the precision-weighted mean, i.e. the three-cornered hat), and ``n_res`` the
dimension of the space of differences: the number of parameters the entries measure
less the number the star has at all. For ``K = 2`` this is *identical* to the old
pair likelihood, constant included, since
``det A = det(S_1 + S_2) / (det S_1 det S_2)`` and ``n_res`` is the dimension of
``y_1 - y_2``; it differs for ``K >= 3``, where the pair version counted
``K (K-1) / 2`` correlated terms instead of one.

A parameter that an entry does not measure carries a placeholder variance of
``MISSING_VAR`` in ``C`` and a harmonized value of 0, so its row and column of
``S_k`` are ``1e9``: its precision ``1e-9`` contributes essentially nothing to
``A``, its residual is 0, and its ``ln det`` contribution is removed by the
mask. A parameter that *no* entry of the star measures leaves ``A`` with a
``1e-8`` diagonal and ``z = 0`` there, which is the same "no information"
answer the trainer's ``y``-ridge gives; the masked ``ln det A`` drops it too.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np
import sonnet as snt
import tensorflow as tf

from ..model.vector_network import VectorNetwork
from .stars import N_PARAMS

log = logging.getLogger(__name__)

#: Smooth lower bound of ln_delta: sigma_delta = 0.003 (far below any quoted error)
LN_MIN = float(2 * np.log(3e-3))

#: Weak Gaussian prior on ln_delta: center (sigma = 0.03, the floor the merge used) and width
PRIOR_MEAN = float(2 * np.log(0.03))
PRIOR_SD = 3.0

#: Per-entry batch arrays, plus the star each entry belongs to
BATCH_KEYS = ("x", "ls", "c", "y", "C", "m")

DeltaFn = Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor]


def floor_ln_delta(raw):
    """``LN_MIN + softplus(raw - LN_MIN)``: equals ``raw`` well above ``LN_MIN``,
    tends smoothly to ``LN_MIN`` below it (so gradients never vanish abruptly)."""
    return LN_MIN + tf.math.softplus(raw - LN_MIN)


class DeltaNetwork(snt.Module):
    """FiLM network ``(x, ln_sigma_x, catalog) -> ln_delta_cov`` (4 values)."""

    def __init__(self, n_cat: int, hidden_size: int = 64, n_hidden: int = 3, name=None):
        super().__init__(name=name)
        self.n_cat = n_cat
        self.net = VectorNetwork(
            n_cat, N_PARAMS, n_hidden, hidden_size,
            output_scale=1e-3, output_bias=PRIOR_MEAN,
            use_residual=True, use_film=True, use_layer_norm=True,
        )
        self(tf.zeros([1, N_PARAMS]), tf.zeros([1, N_PARAMS]), tf.zeros([1], dtype=tf.int32))  # build

    def __call__(self, x, ln_sigma, c):
        return floor_ln_delta(self.net(tf.concat([x, ln_sigma], axis=1), c))


class ConstDelta(snt.Module):
    """One ln_delta per catalog and parameter (the simplest baseline)."""

    def __init__(self, n_cat: int, name=None):
        super().__init__(name=name)
        self.raw = tf.Variable(tf.fill([n_cat, N_PARAMS], PRIOR_MEAN), name="raw")

    def __call__(self, x, ln_sigma, c):
        return floor_ln_delta(tf.gather(self.raw, c))


def fixed_delta(sigma: float) -> DeltaFn:
    """The same added sigma everywhere (``0`` for no added variance)."""
    value = -60.0 if sigma <= 0 else float(2 * np.log(sigma))

    def fn(x, ln_sigma, c):
        return tf.fill(tf.shape(x), tf.constant(value, dtype=x.dtype))

    return fn


def make_batch(stars, idx, seg) -> Dict[str, object]:
    """One block of whole stars as tensors: entry arrays plus the star segment."""
    return {
        **{k: tf.constant(stars[k][idx]) for k in BATCH_KEYS},
        "seg": tf.constant(np.asarray(seg, dtype="i4")),
    }


def star_nll(batch: Dict[str, object], delta_fn: DeltaFn) -> Tuple[tf.Tensor, tf.Tensor]:
    """Per-star negative log likelihood, in float64; also returns the log variances.

    The sum over a star's entries is a segmented reduction, so a batch is a flat
    array of entries plus the index of the star each belongs to: no loop over
    observations and no ragged tensor. Every star of the block must have at least
    one entry, which is what :func:`astrotune.delta.stars.block_indices` yields.
    """
    seg = batch["seg"]
    n_star = tf.reduce_max(seg) + 1  # a dynamic segment count: no retracing per block
    m = tf.cast(batch["m"], tf.float64)  # (N, 4): measured by this entry
    la = delta_fn(batch["x"], batch["ls"], batch["c"])  # (N, 4) log added variance

    y = tf.cast(batch["y"], tf.float64)
    S = tf.cast(batch["C"], tf.float64) + tf.linalg.diag(tf.exp(tf.cast(la, tf.float64)))
    L = tf.linalg.cholesky(S)
    Sinv = tf.linalg.inv(S)  # symmetric positive definite; the Cholesky is for ln det

    # Star-level quantities: the precision sum A and the precision-weighted mean z.
    A = tf.math.unsorted_segment_sum(Sinv, seg, n_star)  # (S, 4, 4)
    b = tf.math.unsorted_segment_sum(tf.linalg.matvec(Sinv, y), seg, n_star)  # (S, 4)
    LA = tf.linalg.cholesky(A)
    z = tf.linalg.cholesky_solve(LA, b[..., None])[..., 0]  # (S, 4)

    d = y - tf.gather(z, seg)  # (N, 4)
    quad = tf.reduce_sum(d * tf.linalg.matvec(Sinv, d), axis=1)  # (N,)
    # ln det S_k over the parameters this entry measures: the placeholder axes of an
    # unmeasured parameter are 1e9 and must not contribute ln(1e9).
    logdet_S = 2.0 * tf.reduce_sum(m * tf.math.log(tf.linalg.diag_part(L)), axis=1)
    # ln det A over the parameters *some* entry of the star measures.
    any_m = tf.cast(tf.math.unsorted_segment_max(m, seg, n_star), tf.float64)  # (S, 4)
    logdet_A = 2.0 * tf.reduce_sum(any_m * tf.math.log(tf.linalg.diag_part(LA)), axis=1)

    quad_s = tf.math.unsorted_segment_sum(quad, seg, n_star)
    logdetS_s = tf.math.unsorted_segment_sum(logdet_S, seg, n_star)
    n_meas_s = tf.math.unsorted_segment_sum(tf.reduce_sum(m, axis=1), seg, n_star)
    # The density lives in the space of the differences, of dimension (measured
    # parameters over the entries) - (parameters the star has at all). With K = 2
    # this constant is what makes the expression identical to the pair likelihood,
    # normalisation included; with K = 1 it is 0, i.e. no information.
    n_res = n_meas_s - tf.reduce_sum(any_m, axis=1)
    neg2lnp = quad_s + logdet_A + logdetS_s + n_res * tf.cast(np.log(2 * np.pi), tf.float64)
    return 0.5 * neg2lnp, la


def prior_penalty(batch: Dict[str, object], la: tf.Tensor) -> tf.Tensor:
    """Mean over measured parameters of the Gaussian prior on ln_delta.

    One term per *entry*, which the star framing gives for free: the pair version
    applied it once per pair, i.e. ``K - 1`` times to the same entry.
    """
    m = tf.cast(batch["m"], tf.float32)
    z2 = ((la - PRIOR_MEAN) / PRIOR_SD) ** 2
    return 0.5 * tf.reduce_sum(z2 * m) / tf.maximum(tf.reduce_sum(m), 1.0)


def l2_penalty(module: snt.Module):
    ws = [v for v in module.trainable_variables if v.shape.rank == 2]
    return tf.add_n([tf.nn.l2_loss(v) for v in ws]) if ws else tf.constant(0.0)


# --- Numpy interface (diagnostics) --------------------------------------------
def numpy_delta_fn(model, batch_size: int = 65536):
    """``f(x, ln_sigma, catalog) -> ln_delta`` on NumPy arrays, in batches."""

    def fn(x, ln_sigma, c):
        out = []
        for i in range(0, len(x), batch_size):
            sl = slice(i, i + batch_size)
            out.append(
                model(tf.constant(x[sl], tf.float32), tf.constant(ln_sigma[sl], tf.float32),
                      tf.constant(c[sl], tf.int32)).numpy()
            )
        return np.concatenate(out) if out else np.zeros((0, N_PARAMS), dtype="f4")

    return fn


def save_delta_model(out_dir: Path, model: DeltaNetwork, config: dict) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = tf.train.Checkpoint(model=model).write(str(out_dir / "delta_model"))
    with open(out_dir / "delta_model_config.json", "w") as f:
        json.dump(config, f, indent=1)
    return Path(path)


def load_delta_model(out_dir: Path, keys=None) -> DeltaNetwork:
    """Restore a model saved by :func:`save_delta_model`.

    ``n_cat`` alone would not catch a model trained on a different catalog list of
    the same length, whose category indices would silently mean other catalogs, so
    the saved keys are compared whenever ``keys`` is given.
    """
    out_dir = Path(out_dir)
    with open(out_dir / "delta_model_config.json") as f:
        cfg = json.load(f)
    if keys is not None and tuple(cfg.get("keys", ())) != tuple(keys):
        raise ValueError(
            f"{out_dir}/delta_model was trained on {list(cfg.get('keys', ()))}, "
            f"not {list(keys)}; retrain it."
        )
    model = DeltaNetwork(cfg["n_cat"], cfg["hidden_size"], cfg["n_hidden"])
    tf.train.Checkpoint(model=model).read(str(out_dir / "delta_model")).assert_existing_objects_matched()
    return model
