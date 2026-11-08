"""Tests for the SDSS-V BOSS-CLAM candidate catalog."""

import numpy as np
import pytest
from astropy.table import Table

from astrotune import registry
from astrotune.catalogs import sdssv_clam as clam
from astrotune.prep import pipeline
from astrotune.prep.transforms import teff_err_to_x


def _row(i, **changes):
    row = dict(
        id=1000 + i, ra=10.0, dec=20.0, snr=50.0,
        teff=5000.0, logg=4.3, fe_h=-0.3, alpha_m=0.1,
        e_teff=20.0, e_logg=0.03, e_fe_h=0.02, e_alpha_m=0.01,
        flag_bad=False, flag_warn=False,
    )
    row.update(changes)
    return row


def _table(rows):
    return Table({k: [r[k] for r in rows] for k in rows[0]})


def _run(rows):
    """Cuts + table assembly, as ``prepare()`` does."""
    t = _table(rows)
    ctx = pipeline.make_context(clam.KEY, make_plots=False)
    mask, counts = pipeline.evaluate_cuts(t, clam, ctx)
    table = clam.finalize(pipeline.build_table(t, mask, clam, ctx), ctx)
    return mask, counts, table


def test_catalog_is_registered():
    assert clam.KEY in registry.keys()
    assert registry.get_spec(clam.KEY).label == clam.LABEL


def test_each_cut_removes_its_star():
    violations = [
        dict(id=-1),
        dict(flag_bad=True),
        dict(snr=clam.SNR_MIN - 1),
        dict(teff=clam.TEFF_MIN - 100),
        dict(logg=clam.LOGG_RANGE[1] + 0.2),
        dict(fe_h=clam.FEH_RANGE[0] - 0.5),
        dict(alpha_m=clam.ALPHA_RANGE[1] + 0.3),
        dict(e_teff=np.nan),  # negative covariance variance
        dict(e_teff=clam.MAX_ERR["e_teff"] + 1),
        dict(e_fe_h=clam.MAX_ERR["e_fe_h"] + 0.1),
    ]
    rows = [_row(0)] + [_row(i, **v) for i, v in enumerate(violations, start=1)]
    mask, counts, table = _run(rows)
    assert mask.tolist() == [True] + [False] * len(violations)
    assert len(table) == 1


def test_no_upper_teff_cut_by_default():
    assert clam.TEFF_MAX is None
    mask, _, _ = _run([_row(0, teff=15000.0, e_teff=100.0)])
    assert mask.tolist() == [True]


def test_warned_star_is_kept_but_its_alpha_is_missing():
    # A warned star with an unusable [alpha/M] must not be rejected for it
    warned = _row(1, flag_warn=True, teff=7000.0, alpha_m=2.0, e_alpha_m=np.nan)
    mask, _, table = _run([_row(0), warned])
    assert mask.tolist() == [True, True]

    good, warn = table[0], table[1]
    assert not good["flag_warn"] and warn["flag_warn"]
    # Measured: finite value, small uncertainty, regularization error
    assert good["x"][3] == pytest.approx(0.1)
    assert good["sigma_x"][3] < 1.0 and good["x_err"][3] < 1.0
    # Not measured: the same placeholders as a catalog without [alpha/Fe]
    assert warn["x"][3] == 0.0
    assert warn["sigma_x"][3] == 1e9 and warn["x_err"][3] == 1e9
    # Everything else about the warned star is kept
    assert np.all(warn["sigma_x"][:3] < 1.0) and np.all(warn["x_err"][:3] < 1e6)
    assert warn["x"][1] == pytest.approx(-0.3)


def test_warn_can_also_drop_feh(monkeypatch):
    monkeypatch.setattr(clam, "WARN_IGNORES_FEH", True)
    warned = _row(1, flag_warn=True, fe_h=-3.0)  # out of range, but ignored
    mask, _, table = _run([_row(0), warned])
    assert mask.tolist() == [True, True]
    assert table["sigma_x"][1][clam.FEH_INDEX] == 1e9 and table["x"][1][clam.FEH_INDEX] == 0.0
    assert table["sigma_x"][0][clam.FEH_INDEX] < 1.0  # unwarned stars keep it


def test_uncertainties_are_scaled_before_the_floors():
    _, _, table = _run([_row(0)])
    floor = 0.01  # default [Fe/H] floor, in quadrature
    expected_feh = np.hypot(clam.UNC_SCALES[1] * 0.02, floor)
    assert table["sigma_x"][0][1] == pytest.approx(expected_feh, rel=1e-4)
    expected_teff = np.hypot(clam.UNC_SCALES[0] * teff_err_to_x(5000.0, 20.0), floor)
    assert table["sigma_x"][0][0] == pytest.approx(expected_teff, rel=1e-4)


def test_hot_stars_get_larger_regularization_errors():
    _, _, table = _run([_row(0), _row(1, teff=7000.0)])
    cool, hot = table["x_err"][0], table["x_err"][1]
    assert cool[0] == pytest.approx(clam.X_ERR[0]) and cool[1] == pytest.approx(clam.X_ERR[1])
    assert hot[0] > cool[0] and hot[1] > cool[1]
    assert hot[2] == cool[2] and hot[3] == cool[3]
