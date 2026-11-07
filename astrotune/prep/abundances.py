"""Compute [alpha/Fe] from individual alpha-element abundances."""

from __future__ import annotations

import numpy as np
from astropy.table import Table

from .transforms import unmask

#: Elements averaged over, in decreasing order of availability.
ALPHA_ELEMENTS = ("O", "Ne", "Mg", "Si", "S", "Ar", "Ca", "Ti")


def _render_fmt(fmt: str, element: str) -> str:
    """Render a column-name format string such as ``'[{x}/Fe]'``.

    Uses f-string semantics so that both ``'{x}Fe'`` and ``'{x.upper()}_FE'``
    work, matching the column-naming conventions of the various catalogs.
    """
    return eval(f"f'''{fmt}'''", {}, {"x": element, "X": element})


def calc_alpha_Fe_from_abundances(
    t: Table,
    X_Fe_fmt: str,
    X_Fe_err_fmt: str,
    flag_fmt: str | None = None,
    err_floor: float = 1e-2,
):
    """Inverse-variance weighted mean of the available [X/Fe] abundances.

    Parameters
    ----------
    t
        Catalog containing per-element abundances.
    X_Fe_fmt, X_Fe_err_fmt
        Format strings for the abundance and uncertainty column names, e.g.
        ``'[{x}/Fe]'`` or ``'{x}Fe'``.
    flag_fmt
        Optional format string for per-element flags; ``0`` means good.
    err_floor
        Floor added in quadrature to the abundance uncertainties.

    Returns
    -------
    alpha_Fe, alpha_Fe_err : np.ndarray
        Weighted mean abundance and its uncertainty. Stars with no valid
        abundance get ``nan``.
    abundances : astropy.table.Table
        The individual abundances, for diagnostic plots.
    """
    X_Fe, X_Fe_weight, X_Fe_bad, names = [], [], [], []

    for element in ALPHA_ELEMENTS:
        key = _render_fmt(X_Fe_fmt, element)
        if key not in t.colnames:
            continue
        names.append(element)

        err_key = _render_fmt(X_Fe_err_fmt, element)
        values = unmask(t[key]).astype("f8").copy()
        weight = 1.0 / (unmask(t[err_key]).astype("f8") ** 2 + err_floor**2)

        if flag_fmt is not None:
            flag_key = _render_fmt(flag_fmt, element)
            bad = ~(unmask(t[flag_key]) == 0)
            if np.any(bad):
                print(f"{np.mean(bad):.3%} of [{element}/Fe] measurements flagged.")
            weight[bad] = 0.0
            values[bad] = 0.0
            X_Fe_bad.append(bad)

        X_Fe.append(values)
        X_Fe_weight.append(weight)

    if not X_Fe:
        raise ValueError("No alpha-element abundance columns were found.")

    weights = np.stack(X_Fe_weight, axis=1)
    no_data = np.all(weights == 0, axis=1)

    values = np.stack(X_Fe, axis=1)
    weights = np.maximum(weights, 1e-10)

    alpha_Fe, weight_sum = np.average(values, weights=weights, axis=1, returned=True)
    alpha_Fe_err = 1.0 / np.sqrt(weight_sum)

    alpha_Fe[no_data] = np.nan
    alpha_Fe_err[no_data] = np.nan

    abundances = Table({name: values[:, i] for i, name in enumerate(names)})
    for name, bad in zip(names, X_Fe_bad):
        abundances[name][bad] = np.nan

    return alpha_Fe, alpha_Fe_err, abundances
