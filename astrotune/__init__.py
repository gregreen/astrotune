"""Astrotune: bringing disparate stellar parameter catalogs into harmony."""

from .config import N_PARAMS, PARAM_NAMES, PROJECT_ROOT  # noqa: F401
from .registry import CATALOGS, get_spec, keys, labels  # noqa: F401

__version__ = "0.1.0"
