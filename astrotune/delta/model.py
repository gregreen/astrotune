"""The delta model: extra (diagonal) variance for every catalog entry.

``ln_delta_cov(x, ln_sigma_x, catalog) -> (4,)`` is the natural log of the
variance added, in the *harmonized* space, to the propagated covariance of an
entry. The inputs are the entry's own parameters (the catalog's coordinates,
e.g. ``5 ln(Teff/5000 K)``), the log of its reported uncertainties, and its
catalog (through FiLM, as in the bijector).

For a pair of entries of one star the likelihood of the harmonized difference is

    d = y_a - y_b ~ N(0, S),   S = C_a + C_b + diag(exp(ln_delta_a) + exp(ln_delta_b)),

evaluated on the parameters measured by both catalogs (the others are dropped
from the quadratic form *and* the log determinant). The negative log likelihood
includes ``ln det S``, so inflating the variance without limit is penalized.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Dict

import numpy as np
import sonnet as snt
import tensorflow as tf

from ..model.vector_network import VectorNetwork
from .pairs import N_PARAMS

log = logging.getLogger(__name__)

#: Smooth lower bound of ln_delta: sigma_delta = 0.003 (far below any quoted error)
LN_MIN = float(2 * np.log(3e-3))

#: Weak Gaussian prior on ln_delta: center (sigma = 0.03, the floor the merge used) and width
PRIOR_MEAN = float(2 * np.log(0.03))
PRIOR_SD = 3.0

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


def pair_nll(batch: Dict[str, tf.Tensor], delta_fn: DeltaFn):
    """Negative log likelihood of every pair of a batch, in float64.

    Returns ``(nll, ln_delta_a, ln_delta_b)``; ``nll`` is over the parameters
    measured by both catalogs, including the ``0.5 ln(2 pi)`` terms.
    """
    f8 = tf.float64
    la = delta_fn(batch["xa"], batch["lsa"], batch["ca"])
    lb = delta_fn(batch["xb"], batch["lsb"], batch["cb"])
    var = tf.exp(tf.cast(la, f8)) + tf.exp(tf.cast(lb, f8))
    m = tf.cast(batch["m"], f8)
    S = tf.cast(batch["Ca"], f8) + tf.cast(batch["Cb"], f8) + tf.linalg.diag(var)
    S = S * m[:, :, None] * m[:, None, :] + tf.linalg.diag(1.0 - m)  # drop unmeasured parameters
    S = 0.5 * (S + tf.linalg.matrix_transpose(S))
    d = (tf.cast(batch["ya"], f8) - tf.cast(batch["yb"], f8)) * m
    L = tf.linalg.cholesky(S)
    w = tf.linalg.triangular_solve(L, d[..., None], lower=True)
    quad = tf.reduce_sum(w**2, axis=[1, 2])
    logdet = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L)), axis=1)
    nll = 0.5 * (quad + logdet + tf.reduce_sum(m, axis=1) * tf.cast(np.log(2 * np.pi), f8))
    return nll, la, lb


def prior_penalty(batch, la, lb):
    """Mean over measured parameters of the Gaussian prior on ln_delta (both sides)."""
    m = tf.cast(batch["m"], tf.float32)
    z2 = ((la - PRIOR_MEAN) / PRIOR_SD) ** 2 + ((lb - PRIOR_MEAN) / PRIOR_SD) ** 2
    return 0.5 * tf.reduce_sum(z2 * m) / tf.maximum(tf.reduce_sum(m), 1.0)


def l2_penalty(module: snt.Module):
    ws = [v for v in module.trainable_variables if v.shape.rank == 2]
    return tf.add_n([tf.nn.l2_loss(v) for v in ws]) if ws else tf.constant(0.0)


# --- Numpy interface (diagnostics, apply) -------------------------------------
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


def load_delta_model(out_dir: Path) -> DeltaNetwork:
    """Restore a model saved by :func:`save_delta_model`."""
    out_dir = Path(out_dir)
    with open(out_dir / "delta_model_config.json") as f:
        cfg = json.load(f)
    model = DeltaNetwork(cfg["n_cat"], cfg["hidden_size"], cfg["n_hidden"])
    tf.train.Checkpoint(model=model).read(str(out_dir / "delta_model")).assert_existing_objects_matched()
    return model
