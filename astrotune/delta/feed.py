"""Materialize a trained delta model's calibrated sigma, one file per catalog.

The delta model answers "how wrong is this catalog's quoted error here", and its
inputs are the entry's own parameters and its catalog -- *not* the bijector. The
bijector only enters the likelihood the delta model was fitted with (through
``J``). So one fitted delta model gives a table of calibrated sigmas that any
number of bijector retrainings can read:

.. code-block:: text

    bijector (sigma_x = scaled + floored)  ->  delta model
        -> sigma_cal = sqrt(exp(ln_scale) * sigma_raw^2 + exp(ln_add))
        -> bijector (sigma_x = sigma_cal)  ->  delta model  ->  ...

which is the iteration described in ``porting_project.md``. ``sigma_cal`` is
written to ``<run>/delta/calibrated_sigma/<key>.npz`` and is keyed by a checksum
of the delta model's weights, so it is rebuilt when the model changes and reused
when it does not. :func:`astrotune.training_data.load_tables` reads it with
``sigma_source="delta"``.
"""

from __future__ import annotations

import logging
import zlib
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

from .. import registry

log = logging.getLogger(__name__)


def model_checksum(model, keys: Sequence[str]) -> int:
    """CRC32 over the model's trainable variables and its catalog list."""
    crc = zlib.crc32(",".join(keys).encode())
    for var in sorted(model.trainable_variables, key=lambda v: v.name):
        crc = zlib.crc32(np.ascontiguousarray(var.numpy(), dtype="f4").tobytes(), crc)
    return crc


def calibrated_sigma(run_dir, keys: Optional[Sequence[str]] = None,
                     out_dir: Optional[Path] = None,
                     batch_size: int = 262_144, model=None) -> Dict[str, np.ndarray]:
    """``{catalog key: sigma_cal}``, ``(n, d)`` in the catalog's own units.

    Rows line up with the prepared table of that catalog (``load_catalog(key)``),
    before any deduplication, so the caller may dedupe afterwards. A parameter a
    catalog does not measure keeps its placeholder value, so it stays unmeasured.

    The result is cached per catalog in ``<out-dir>/calibrated_sigma/``; the cache
    is regenerated when the delta model or the catalog list changes.
    """
    from ..apply import padded_inputs
    from ..prep.io import load_catalog
    from ..prep.transforms import unmask
    from .covariance import entry_cov_np  # noqa: F401  (documents the provenance)
    from .model import load_delta_model, numpy_delta_fn
    from .stars import raw_sigma

    run_dir = Path(run_dir)
    delta_dir = Path(out_dir) if out_dir else run_dir / "delta"
    keys = tuple(keys) if keys is not None else registry.run_keys(run_dir)
    cache_dir = delta_dir / "calibrated_sigma"

    model = model if model is not None else load_delta_model(delta_dir, keys=keys)
    fn = numpy_delta_fn(model, batch_size=batch_size)
    group_of = registry.group_index(keys)
    checksum = model_checksum(model, keys)

    out: Dict[str, np.ndarray] = {}
    for k, key in enumerate(keys):
        path = cache_dir / f"{key}.npz"
        if path.exists():
            with np.load(path) as f:
                if int(f["checksum"]) == checksum:
                    out[key] = f["sigma"]
                    continue
        t = load_catalog(key)
        x, sigma_prep, measured = padded_inputs(t)
        raw = raw_sigma(t, key)
        d = np.asarray(t["sigma_x"]).shape[1]
        ls = np.log(np.where(measured, np.pad(raw, ((0, 0), (0, 4 - raw.shape[1]))), 1.0))
        ln_scale, ln_add = fn(x, ls.astype("f4"),
                              np.full(len(x), group_of[k], dtype="i4"))
        sig = np.sqrt(np.exp(ln_scale.astype("f8")) * raw.astype("f8") ** 2
                      + np.exp(ln_add.astype("f8")))
        sig = np.where(measured[:, :d], sig[:, :d], sigma_prep[:, :d])
        out[key] = sig.astype("f4")
        cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, sigma=out[key], checksum=np.int64(checksum), keys=np.asarray(keys))
        tmp.replace(path)
        log.info("%s: calibrated sigma written (%d rows)", key, len(sig))
    return out
