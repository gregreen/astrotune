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
from ..config import N_PARAMS

log = logging.getLogger(__name__)


def model_checksum(model, keys: Sequence[str]) -> int:
    """CRC32 over the model's trainable variables and its catalog list."""
    crc = zlib.crc32(",".join(keys).encode())
    for var in sorted(model.trainable_variables, key=lambda v: v.name):
        crc = zlib.crc32(np.ascontiguousarray(var.numpy(), dtype="f4").tobytes(), crc)
    return crc


def installed_models(path) -> Dict[str, Path]:
    """``{model type: directory}`` for the delta models rooted at ``path``.

    ``path`` may be a run directory (whose ``delta/`` holds one subdirectory per
    type), that ``delta/`` directory itself, or a single model directory. The older
    flat layout -- ``delta_model.*`` sitting directly in the root -- is read too, so
    runs trained before the type directories existed still resolve.
    """
    from .model import delta_model_type

    root = Path(path)
    found: Dict[str, Path] = {}

    def add(directory: Path) -> None:
        if (directory / "delta_model_config.json").exists():
            found.setdefault(delta_model_type(directory), directory)

    for base in (root, root / "delta"):
        if (base / "delta_model_config.json").exists():
            add(base)  # the legacy flat layout
        if base.is_dir():
            for sub in sorted(p for p in base.iterdir() if p.is_dir()):
                add(sub)

    return found


def resolve_delta_model_dir(path, model_type: Optional[str] = None) -> Path:
    """The directory holding ``delta_model.*``, from any of the accepted layouts.

    With ``model_type`` unset the choice is made for the caller only when it is
    unambiguous: a directory holding one model resolves to it, and one holding
    several raises with the list, because silently picking the alphabetically first
    would apply the wrong recalibration.
    """
    from .model import canonical_model_type

    found = installed_models(path)
    root = Path(path)
    if not found:
        raise ValueError(
            f"no delta model under {root}: expected <type>/delta_model_config.json, "
            f"delta/<type>/delta_model_config.json, or the older flat "
            f"delta_model_config.json"
        )
    if model_type is None:
        if len(found) == 1:
            return next(iter(found.values()))
        raise ValueError(
            f"{root} holds several delta models ({sorted(found)}); say which one to "
            f"use with --delta-model-type"
        )
    kind = canonical_model_type(model_type)
    if kind not in found:
        raise ValueError(
            f"no {kind} delta model under {root}; found {sorted(found)}. Train it with "
            f"astrotune-delta-train --model-type {kind}"
        )
    return found[kind]


def calibrated_sigma(run_dir, keys: Optional[Sequence[str]] = None,
                     out_dir: Optional[Path] = None,
                     batch_size: int = 262_144, model=None,
                     model_type: Optional[str] = None) -> Dict[str, np.ndarray]:
    """``{catalog key: sigma_cal}``, ``(n, d)`` in the catalog's own units.

    Rows line up with the prepared table of that catalog (``load_catalog(key)``),
    before any deduplication, so the caller may dedupe afterwards. A parameter a
    catalog does not measure keeps its placeholder value, so it stays unmeasured.

    Which model is used follows :func:`resolve_delta_model_dir`: pass ``model_type``
    to choose among the models saved under one run's ``delta/``. The result is
    cached per catalog in ``<model dir>/calibrated_sigma/``; the cache is
    regenerated when the model or the catalog list changes.
    """
    from ..apply import padded_inputs
    from ..prep.io import load_catalog
    from ..prep.transforms import unmask
    from .covariance import entry_cov_np  # noqa: F401  (documents the provenance)
    from .model import load_delta_model, numpy_delta_fn
    from .stars import raw_sigma

    run_dir = Path(run_dir)
    root = Path(out_dir) if out_dir else run_dir / "delta"
    keys = tuple(keys) if keys is not None else registry.run_keys(run_dir)
    if model is None:
        model_dir = resolve_delta_model_dir(root, model_type)
        model = load_delta_model(model_dir, keys=keys)
    else:
        try:
            model_dir = resolve_delta_model_dir(root, model_type)
        except ValueError:
            # An explicitly given model with nothing on disk: cache beside it.
            model_dir = root
    cache_dir = model_dir / "calibrated_sigma"
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
        # A three-parameter catalog (no [alpha/Fe], or no logg) has a ``(n, 3)``
        # quoted-sigma table, while the model always emits four columns: pad with
        # a neutral 1.0 so the squares line up. The padded axes are unmeasured,
        # so the ``where`` below drops them again either way, but the product
        # cannot be formed without them.
        d = raw.shape[1]
        raw = np.pad(raw, ((0, 0), (0, N_PARAMS - d)), constant_values=1.0)
        ls = np.log(np.where(measured, raw, 1.0))
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
