"""Project-wide configuration: paths, parameter ordering, error floors.

Paths can be overridden with environment variables, which is useful on
clusters where large data live on a scratch filesystem:

    ASTROTUNE_DATA_DIR        (default: <project>/data)
    ASTROTUNE_DIAGNOSTIC_DIR  (default: <project>/diagnostic_plots)
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _env_path(var: str, default: Path) -> Path:
    value = os.environ.get(var)
    path = Path(value).expanduser() if value else default
    return path.resolve()


DATA_DIR = _env_path("ASTROTUNE_DATA_DIR", PROJECT_ROOT / "data")
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
TRAINING_DIR = DATA_DIR / "training"

DIAGNOSTIC_DIR = _env_path("ASTROTUNE_DIAGNOSTIC_DIR", PROJECT_ROOT / "diagnostic_plots")

# Training pairs + control points, built from the per-catalog tables
PAIRS_PATH = DATA_DIR / "pairs_and_ctrl_points.h5"

# Trained models: models/<run-name>/
MODELS_DIR = PROJECT_ROOT / "models"

# Read-only location of the outputs produced by the original monolithic
# notebook. Used by the parity checks, never written to.
LEGACY_TRAINING_DIR = PROJECT_ROOT / "training_data"

# --- Stellar parameters -----------------------------------------------------
# Fixed order, used by every catalog and by the model. Do not reorder: the
# index of a parameter is part of the on-disk format.
PARAM_NAMES = (
    "5 ln(Teff/5000 K)",
    "[Fe/H]",
    "logg",
    "[alpha/Fe]",
)
N_PARAMS = len(PARAM_NAMES)

# Default error floors, added in quadrature to the catalog uncertainties.
# (Teff, [Fe/H], logg, [alpha/Fe]); catalogs with fewer parameters use a
# prefix of this array. The [alpha/Fe] floor was 0.05 (legacy notebook, the first
# trained models) and was lowered to 0.02 on 2026-11-09: it was 77-100 % of the
# [alpha/Fe] variance of most catalogs, and the pair differences of the harmonized
# estimates are consistent with errors of about 0.02 (porting_project.md, section 9.10).
ERROR_FLOORS = np.array([0.01, 0.01, 0.04, 0.02], dtype="f8")

# The [alpha/Fe] floor of the legacy tables, to compare prepared tables with them.
LEGACY_ALPHA_FLOOR = 0.05

# Default regularization error used when a catalog does not override it.
DEFAULT_X_ERR = 0.1


# --- Path helpers -----------------------------------------------------------
def raw_dir(key: str) -> Path:
    return RAW_DIR / key


def interim_dir(key: str) -> Path:
    return INTERIM_DIR / key


def training_dir(key: str) -> Path:
    return TRAINING_DIR / key


def diagnostic_dir(key: str) -> Path:
    return DIAGNOSTIC_DIR / key


def catalog_path(key: str) -> Path:
    """Final per-catalog table: data/training/<key>/<key>.h5"""
    return training_dir(key) / f"{key}.h5"


def meta_path(key: str) -> Path:
    """Sidecar metadata: data/training/<key>/<key>.meta.json"""
    return training_dir(key) / f"{key}.meta.json"
