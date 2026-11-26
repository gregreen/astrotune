"""Tests for the MaStar candidate catalog.

The catalog cannot be prepared without the 7.9 MB VAC file and a Gaia
crossmatch, so these tests cover the module contract and the cut logic directly
on a small synthetic table, plus the parts of it that are easy to get wrong and
were wrong once already (the parameter order).
"""

import numpy as np
import pytest
from astropy.table import Table

from astrotune import registry
from astrotune.catalogs import mastar
from astrotune.prep import pipeline


def _good_row(i=0):
    return dict(
        RA=10.0 + i, DEC=20.0,
        gdr3_source_id=1000 + i,
        ra_gdr3=10.0 + i, dec_gdr3=20.0,
        NGROUPS=4, NALPHAGROUPS=3,
        TEFF_MED=5500.0, TEFF_MED_ERR=80.0,
        LOGG_MED=4.3, LOGG_MED_ERR=0.15,
        FEH_CAL_MED=-0.4, FEH_CAL_MED_ERR=0.1,
        ALPHA_MED=0.15, ALPHA_MED_ERR=0.05,
    )


def _table(overrides=()):
    """A good row plus one row per ``(column, value)`` override, which must fail."""
    rows = [_good_row(0)]
    for i, (column, value) in enumerate(overrides, start=1):
        row = _good_row(i)
        row[column] = value
        rows.append(row)
    return Table({k: [r[k] for r in rows] for k in rows[0]})


def _mask(t):
    mask, counts = pipeline.evaluate_cuts(
        t, mastar, pipeline.make_context(mastar.KEY, make_plots=False)
    )
    return np.asarray(mask), counts


def test_catalog_is_a_registered_candidate():
    assert mastar.KEY in registry.candidate_keys()
    assert mastar.KEY not in registry.INDEX  # not part of the model (yet)
    spec = registry.get_spec(mastar.KEY)
    assert spec.label == mastar.LABEL
    assert spec.n_params == mastar.N_PARAMS == 4


def test_parameter_columns_are_in_the_canonical_order():
    # Teff, [Fe/H], log g, [alpha/Fe]: the order the whole project assumes. It
    # is easy to write the file's own order here instead, and nothing complains.
    assert mastar.PARAM_COLS == ("TEFF_MED", "FEH_CAL_MED", "LOGG_MED", "ALPHA_MED")
    assert mastar.UNC_COLS == (
        "TEFF_MED_ERR", "FEH_CAL_MED_ERR", "LOGG_MED_ERR", "ALPHA_MED_ERR"
    )
    # and each uncertainty must belong to the parameter in the same slot
    for param, unc in zip(mastar.PARAM_COLS, mastar.UNC_COLS):
        assert unc == f"{param}_ERR" or unc.startswith(param.split("_")[0])


def test_prepared_x_columns_hold_the_right_parameters():
    # The transformed columns must be (5 ln(Teff/5000), [Fe/H], log g, [a/Fe]).
    t = _table()
    x = np.asarray(
        pipeline.build_table(t, np.ones(len(t), dtype=bool), mastar,
                             pipeline.make_context(mastar.KEY, make_plots=False))["x"],
        dtype="f8",
    )
    assert x[0, 0] == pytest.approx(5.0 * np.log(5500.0 / 5000.0))
    assert x[0, 1] == pytest.approx(-0.4)  # [Fe/H]
    assert x[0, 2] == pytest.approx(4.3)  # log g
    assert x[0, 3] == pytest.approx(0.15)  # [alpha/Fe]


def test_cuts_remove_each_kind_of_bad_star():
    bad = {
        "gdr3_source_id": -1,  # no Gaia match
        "NGROUPS": 1,  # one method only, no cross-check
        "NALPHAGROUPS": 1,  # ditto for [alpha/Fe]
        "TEFF_MED": np.nan,  # -999 sentinel
        "TEFF_MED_ERR": 900.0,  # failed fit
        "LOGG_MED_ERR": -999.0,
        "FEH_CAL_MED_ERR": 0.8,
        "ALPHA_MED_ERR": 0.9,
    }
    t = _table(bad.items())
    mask, counts = _mask(t)
    assert mask.tolist() == [True] + [False] * len(bad)
    assert list(counts)[-1] == "uncertainties finite and below limits"


def test_cuts_remove_grid_edges_but_keep_the_hot_sequence():
    # Both ends of the Teff range are the fitting grid's boundaries; a hot star
    # well inside them must survive, since LAMOST HotPayne gives the model data
    # up there.
    t = _table([
        ("TEFF_MED", 3500.0),  # exactly on the cool edge of the grid
        ("TEFF_MED", 3400.0),  # below it
        ("TEFF_MED", 30000.0),  # exactly on the hot edge
        ("TEFF_MED", 31000.0),  # above it
        ("LOGG_MED", 5.0),  # exactly at the top of the grid
        ("LOGG_MED", -0.6),
        ("FEH_CAL_MED", -3.6),
        ("ALPHA_MED", 0.9),
    ])
    mask, _ = _mask(t)
    assert mask.tolist() == [True] + [False] * 8

    hot = _table([("TEFF_MED", 12000.0), ("TEFF_MED", 25000.0)])
    assert _mask(hot)[0].tolist() == [True, True, True]


def test_missing_parameters_are_turned_into_nan_on_load(monkeypatch, tmp_path):
    # The sentinel is -999 in every column, errors included; the crossmatch
    # wrapper is the only thing load() does around it, so stub it out.
    from astropy.io import fits

    path = tmp_path / mastar.FILENAME
    columns = {"RA": [1.0, 2.0], "DEC": [3.0, 4.0]}
    for param, unc in zip(mastar.PARAM_COLS, mastar.UNC_COLS):
        columns[param] = [5000.0, -999.0]
        columns[unc] = [50.0, -999.0]
    fits.BinTableHDU(Table(columns)).writeto(path, overwrite=True)

    captured = {}

    def fake_crossmatch(make_catalog, **kwargs):
        captured["catalog"] = make_catalog()
        return captured["catalog"]

    monkeypatch.setattr(mastar, "crossmatch_catalog", fake_crossmatch)
    ctx = pipeline.make_context(mastar.KEY, make_plots=False)
    ctx.raw_dir = tmp_path

    loaded = mastar.load(ctx)
    for name in mastar.PARAM_COLS + mastar.UNC_COLS:
        assert np.isnan(loaded[name][1]), name
        assert loaded[name][0] == columns[name][0], name
