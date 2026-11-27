"""Parameter transformations, error floors, and table assembly.

Every catalog ends up in the same on-disk format:

    id       int64             star identifier (Gaia DR3 source_id)
    x        float (n, d)      transformed stellar parameters
    sigma_x  float (n, d)      transformed statistical uncertainties
    cov_x    float (n, d, d)   transformed covariance (only for catalogs that have one)
    x_err    float (n, d)      regularization errors (see the paper, App. A)
    <extra>  float (n,)        e.g. ra, dec

``d`` is 4 (Teff, [Fe/H], logg, [alpha/Fe]) or 3 when [alpha/Fe] is absent.
``sigma_x`` is always present and always the marginal of ``cov_x`` (where there
is one): it is what every consumer that wants one number per parameter uses.
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


def teff_cov_to_x(cov, teff):
    """Propagate a covariance through x_Teff = 5 ln(Teff/5000 K).

    The transform is nonlinear, so this is the usual first-order propagation
    ``J C J^T`` with ``J = diag(5 / Teff, 1, 1, ...)`` -- the same linearization
    that ``teff_err_to_x`` uses for a single uncertainty, applied to the whole
    block. Everything except the first row and column passes through untouched.
    """
    cov = np.array(cov, dtype="f8", copy=True)
    jac = TEFF_SCALE / np.asarray(teff, dtype="f8")
    cov[:, 0, :] *= jac[:, None]
    cov[:, :, 0] *= jac[:, None]
    return cov


PARAM_COV_TRANSF: tuple[Callable, ...] = (
    teff_cov_to_x,
    None,
    None,
    None,
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


def scale_covariance(cov, scales=None, floors=None, max_condition=1e6):
    """``a_i C_ij a_j + b_i delta_ij``, the covariance counterpart of the floors.

    ``a = scales`` (the catalog's ``UNC_SCALES``) is applied as a diagonal
    congruence, which keeps the matrix positive semi-definite and leaves every
    correlation coefficient unchanged -- ``a_i a_j`` cancels between the numerator
    and the two variances. ``b = floors**2`` is added on the diagonal only; that
    is the part that dilutes correlations, as an uncorrelated floor must.

    The result is symmetrized and its smallest eigenvalue is raised to
    ``largest / max_condition`` when a block is (numerically) singular. Rows that
    are already invalid (a non-positive variance in the source) stay NaN and are
    removed by the catalog's cuts.
    """
    out = np.array(cov, dtype="f8", copy=True)
    if scales is not None:
        a = np.asarray(scales, dtype="f8")
        out = out * a[:, None] * a[None, :]
    if floors is not None:
        b = np.asarray(floors, dtype="f8") ** 2
        idx = np.arange(out.shape[-1])
        out[..., idx, idx] += b
    # symmetrize (the source files are not always exactly symmetric)
    out = 0.5 * (out + np.swapaxes(out, 1, 2))
    if max_condition is not None:
        finite = np.all(np.isfinite(out), axis=(1, 2))
        if finite.any():
            block = out[finite]
            sigma = np.sqrt(np.einsum("nii->ni", block))
            rho = block / (sigma[:, :, None] * sigma[:, None, :])
            evals, evecs = np.linalg.eigh(rho)
            bad = evals[:, 0] < 1.0 / max_condition
            if bad.any():
                evals[bad, 0] = 1.0 / max_condition
                rho = np.einsum("nij,nj,nkj->nik", evecs, evals, evecs)
                # clipping changes the diagonal; restore it, so that the stored
                # sigma_x remains exactly the marginal of the stored block
                d = np.sqrt(np.einsum("nii->ni", rho))
                rho = rho / (d[:, :, None] * d[:, None, :])
            out[finite] = rho * sigma[:, :, None] * sigma[:, None, :]
    return out


def covariance_to_sigma(cov) -> np.ndarray:
    """The marginal uncertainties, ``sqrt(diag(C))``."""
    return np.sqrt(np.einsum("nii->ni", np.asarray(cov, dtype="f8")))


def covariance_to_correlation(cov, max_condition=None) -> np.ndarray:
    """Correlation matrices, for diagnostics (invalid rows come out as NaN)."""
    cov = np.asarray(cov, dtype="f8")
    sigma = covariance_to_sigma(cov)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = cov / (sigma[:, :, None] * sigma[:, None, :])
    return corr


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
    param_unc_transf_raw: Optional[Sequence[Callable]] = None,
    param_errs=None,
    cov_key: Optional[str] = None,
    param_cov_transf: Optional[Callable] = None,
    cov_scales=None,
    cov_floors=None,
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
    param_unc_transf_raw
        The same, but without the catalog's own rescale and without the error
        floors. Used for the ``sigma_x_raw`` column: what the catalog reported,
        in our parameter units. The bijector is trained on ``sigma_x`` (scaled
        and floored), the delta model on ``sigma_x_raw``.
    param_errs
        Regularization errors. Either ``(n, d)``-shaped, already aligned with
        the full input table, or a per-parameter sequence of constants. When
        omitted, every axis gets :data:`~astrotune.config.DEFAULT_X_ERR`.
    cov_key
        Optional column holding the catalog's ``(n, d, d)`` covariance in the
        canonical order. When given, ``cov_x`` is written: propagated through
        ``param_cov_transf``, then scaled and floored by :func:`scale_covariance`
        with ``cov_scales``/``cov_floors``. ``sigma_x`` is written either way and
        stays consistent with it (same scales, same floors).
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
    if param_unc_transf_raw is not None:
        # What the catalog reported, in our units: no catalog rescale, no floor.
        sigma_raw = np.stack(
            [
                f(unmask(t[k0][valid]), unmask(t[k1][valid]))
                for f, k0, k1 in zip(param_unc_transf_raw, param_keys, param_unc_keys)
            ],
            axis=1,
        )
        out["sigma_x_raw"] = np.asarray(sigma_raw, dtype="f4")
    for k in extra_keys:
        out[k] = unmask(t[k][valid])
    if cov_key is not None:
        cov = np.asarray(unmask(t[cov_key][valid]), dtype="f8")
        if cov.shape[1:] != (d, d):
            raise ValueError(
                f"{cov_key} holds covariance blocks of shape {cov.shape[1:]}, "
                f"expected {(d, d)}"
            )
        if param_cov_transf is not None:
            cov = param_cov_transf(cov, unmask(t[param_keys[0]][valid]))
        block = scale_covariance(cov, cov_scales, cov_floors)
        # Store float32, so clamp once more on the rounded values: the storage
        # precision is what the likelihood will see.
        out["cov_x"] = scale_covariance(block.astype("f4").astype("f8")).astype("f4")
    return out


def default_floors(n_params: int) -> np.ndarray:
    """Error floors for a catalog with ``n_params`` parameters."""
    if n_params > N_PARAMS:
        raise ValueError(f"At most {N_PARAMS} parameters are supported.")
    return ERROR_FLOORS[:n_params]
