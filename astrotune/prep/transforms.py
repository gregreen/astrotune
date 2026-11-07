"""Parameter transformations, error floors, and table assembly.

Every catalog ends up in the same on-disk format:

    id       int64             star identifier (Gaia DR3 source_id)
    x        float (n, d)      transformed stellar parameters
    sigma_x  float (n, d)      transformed statistical uncertainties
    x_err    float (n, d)      regularization errors (see the paper, App. A)
    <extra>  float (n,)        e.g. ra, dec

``d`` is 4 (Teff, [Fe/H], logg, [alpha/Fe]) or 3 when [alpha/Fe] is absent.
"""

from __future__ import annotations

from typing import Callable, Iterable, Optional, Sequence

import astropy.table
import numpy as np
from astropy.table import Table

from ..config import DEFAULT_X_ERR, ERROR_FLOORS, N_PARAMS

TEFF_REF_K = 5000.0
TEFF_SCALE = 5.0


# --- Parameter transformations ---------------------------------------------
def teff_to_x(teff):
    """x_Teff = 5 ln(Teff / 5000 K)."""
    return TEFF_SCALE * np.log(np.asarray(teff, dtype="f8") / TEFF_REF_K)


def teff_err_to_x(teff, teff_err):
    """sigma_x = 5 sigma_Teff / Teff."""
    return TEFF_SCALE * np.asarray(teff_err, dtype="f8") / np.asarray(teff, dtype="f8")


def _identity(x):
    return x


def _identity_err(x, x_err):
    return x_err


PARAM_TRANSF: tuple[Callable, ...] = (teff_to_x, _identity, _identity, _identity)
PARAM_UNC_TRANSF: tuple[Callable, ...] = (
    teff_err_to_x,
    _identity_err,
    _identity_err,
    _identity_err,
)


# --- Array helpers ----------------------------------------------------------
def unmask(values, fill=np.nan) -> np.ndarray:
    """Return a plain ndarray, filling masked entries.

    Integer columns are promoted to float when a NaN fill is requested, since
    NaN is not representable as an integer.
    """
    if isinstance(values, (astropy.table.MaskedColumn, np.ma.MaskedArray)):
        try:
            values = values.filled(fill)
        except (TypeError, ValueError):
            values = values.astype("f8").filled(fill)
    return np.asarray(values)


def as_id_array(values) -> np.ndarray:
    """Return integer identifiers, with invalid (masked) entries set to 0."""
    return unmask(values, fill=0).astype("i8")


# --- Error floors -----------------------------------------------------------
def with_err_floor(func: Callable, floor: float) -> Callable:
    """Add ``floor`` in quadrature to the uncertainty returned by ``func``."""

    def wrapped(values, errs):
        return np.sqrt(np.asarray(func(values, errs), dtype="f8") ** 2 + floor**2)

    wrapped.__name__ = f"{getattr(func, '__name__', 'func')}_floor{floor:g}"
    return wrapped


def apply_err_floors(funcs: Iterable[Callable], floors: Iterable[float]) -> tuple:
    return tuple(with_err_floor(f, e) for f, e in zip(funcs, floors))


# --- Table assembly ---------------------------------------------------------
def extract_table(
    t: Table,
    id_key: str,
    param_keys: Sequence[str],
    param_unc_keys: Sequence[str],
    extra_keys: Sequence[str] = (),
    *,
    param_transf: Optional[Sequence[Callable]] = None,
    param_unc_transf: Optional[Sequence[Callable]] = None,
    param_errs=None,
    mask=None,
) -> Table:
    """Build the standardized table described in the module docstring.

    Parameters
    ----------
    t
        Input catalog.
    id_key
        Column holding the (Gaia DR3) source identifier. Rows with ``id <= 0``
        are dropped.
    param_keys, param_unc_keys
        Columns holding the parameters and their statistical uncertainties,
        in the canonical order (Teff, [Fe/H], logg, [alpha/Fe]).
    extra_keys
        Additional columns to carry along (e.g. ``ra``, ``dec``).
    param_transf, param_unc_transf
        Callables applied to the parameters / uncertainties. Default to the
        identity.
    param_errs
        Regularization errors. Either ``(n, d)``-shaped, already aligned with
        the full input table, or a per-parameter sequence of constants. When
        omitted, every axis gets :data:`~astrotune.config.DEFAULT_X_ERR`.
    mask
        Additional boolean mask over the input table (e.g. quality cuts).
    """
    if param_transf is None:
        param_transf = tuple(_identity for _ in param_keys)
    if param_unc_transf is None:
        param_unc_transf = tuple(_identity_err for _ in param_keys)

    valid = as_id_array(t[id_key]) > 0  # id <= 0 is treated as invalid
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)

    n = int(np.count_nonzero(valid))
    d = len(param_keys)

    if param_errs is None:
        param_errs = DEFAULT_X_ERR * np.ones((n, d), dtype="f4")
    elif isinstance(param_errs, (tuple, list)):
        errs = np.empty((n, d), dtype="f4")
        for i, value in enumerate(param_errs):
            errs[:, i] = value
        param_errs = errs
    else:
        param_errs = np.asarray(param_errs, dtype="f4")
        if param_errs.shape[0] == len(t):
            param_errs = param_errs[valid]
        elif param_errs.shape[0] != n:
            raise ValueError(
                f"param_errs has {param_errs.shape[0]} rows; expected {len(t)} (full "
                f"table) or {n} (selected rows)."
            )

    x = np.stack(
        [f(unmask(t[k][valid])) for f, k in zip(param_transf, param_keys)], axis=1
    )
    sigma_x = np.stack(
        [
            f(unmask(t[k0][valid]), unmask(t[k1][valid]))
            for f, k0, k1 in zip(param_unc_transf, param_keys, param_unc_keys)
        ],
        axis=1,
    )

    out = Table(
        {
            "id": as_id_array(t[id_key])[valid],
            "x": np.asarray(x, dtype="f4"),
            "sigma_x": np.asarray(sigma_x, dtype="f4"),
            "x_err": param_errs,
        }
    )
    for k in extra_keys:
        out[k] = unmask(t[k][valid])
    return out


def default_floors(n_params: int) -> np.ndarray:
    """Error floors for a catalog with ``n_params`` parameters."""
    if n_params > N_PARAMS:
        raise ValueError(f"At most {N_PARAMS} parameters are supported.")
    return ERROR_FLOORS[:n_params]
