"""Tests for the alpha-element abundance helpers."""

import numpy as np
import pytest
from astropy.table import Table

from astrotune.prep.abundances import as_x_fe, calc_alpha_Fe_from_abundances


def _catalog(fe_h, x_fe, names=("Mg", "Si", "Ca"), err=0.02):
    """A minimal catalogue storing ``[X/H]`` columns, as SDSS-V MDC does."""
    t = Table({"FE_H": np.asarray(fe_h, dtype="f8")})
    t["FE_H_ERR"] = np.full(len(t), 0.05)
    for i, name in enumerate(names):
        t[f"{name}_H"] = np.asarray(fe_h, dtype="f8") + np.asarray(x_fe[i], dtype="f8")
        t[f"{name}_H_ERR"] = np.full(len(t), err)
    return t


def test_weighted_mean_of_equal_uncertainties_is_the_plain_mean():
    rng = np.random.default_rng(3)
    fe_h = rng.normal(-0.1, 0.2, 500)
    x_fe = [np.full(500, 0.05), np.full(500, 0.10), np.full(500, 0.15)]
    t = _catalog(fe_h, x_fe)

    alpha_h, alpha_h_err, abundances = calc_alpha_Fe_from_abundances(
        t, "{x}_H", "{x}_H_ERR"
    )

    # the helper averages the columns it is given: [alpha/H] here, so the
    # catalogue's prepare_columns subtracts [Fe/H] to get [alpha/Fe]
    assert np.allclose(alpha_h - fe_h, 0.10)
    # three independent 0.02 measurements, each floored at 0.01 in quadrature
    sigma = np.hypot(0.02, 0.01) / np.sqrt(3)
    assert np.allclose(alpha_h_err, sigma, rtol=1e-6)
    # the per-element table is in the same units as the input: [X/H]
    assert np.allclose(abundances["Mg"], fe_h + 0.05)


def test_as_x_fe_recovers_the_per_element_abundance_ratios():
    rng = np.random.default_rng(4)
    fe_h = rng.normal(0.0, 0.2, 500)
    x_fe = [rng.normal(0.02, 0.03, 500) for _ in range(3)]
    t = _catalog(fe_h, x_fe)

    alpha_h, _, abundances = calc_alpha_Fe_from_abundances(t, "{x}_H", "{x}_H_ERR")
    alpha_fe = alpha_h - fe_h
    ratios = as_x_fe(abundances, fe_h)

    for name, expected in zip(("Mg", "Si", "Ca"), x_fe):
        assert np.allclose(ratios[name], expected)
        assert np.corrcoef(ratios[name], alpha_fe)[0, 1] > 0.5

    # dropping the metallicity trend is the whole point: [X/H] itself carries
    # almost none of the alpha signal, because [X/Fe] is a small residual on top
    # of a much larger common [Fe/H] scatter
    assert abs(np.corrcoef(abundances["Mg"], alpha_fe)[0, 1]) < 0.2


def test_as_x_fe_keeps_missing_values_missing():
    t = Table({"Mg": np.array([0.1, np.nan, -0.2]), "Si": np.array([0.0, 0.1, np.nan])})
    fe_h = np.array([0.0, 0.2, -0.1])

    ratios = as_x_fe(t, fe_h)

    assert ratios["Mg"][0] == pytest.approx(0.1)
    assert np.isnan(ratios["Mg"][1])
    assert ratios["Si"][2] is np.nan or np.isnan(ratios["Si"][2])
