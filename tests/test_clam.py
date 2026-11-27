"""Tests for the SDSS-V BOSS-CLAM candidate catalog."""

import numpy as np
import pytest
from astropy.table import Table

from astrotune import config, registry
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
    return mask, counts, pipeline.build_table(t, mask, clam, ctx)


def test_catalog_is_registered():
    assert clam.KEY in registry.keys()
    assert registry.get_spec(clam.KEY).label == clam.LABEL


def test_each_cut_removes_its_star():
    violations = [
        dict(id=-1),
        dict(flag_bad=True),
        dict(flag_warn=True),  # no usable [alpha/M] (or [Fe/H]) -> dropped
        dict(snr=clam.SNR_MIN - 1),
        dict(teff=clam.TEFF_MIN - 100),
        dict(logg=clam.LOGG_RANGE[1] + 0.2),
        dict(fe_h=clam.FEH_RANGE[0] - 0.5),
        dict(alpha_m=clam.ALPHA_RANGE[1] + 0.3),
        dict(alpha_m=np.nan),  # [alpha/M] not measured
        dict(fe_h=np.nan),  # [Fe/H] not measured
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


def test_warned_star_is_removed_even_though_it_has_a_value():
    """A warned row has no usable [alpha/M], so the catalog cannot keep it.

    The catalog is four-parameter for every star it contributes: a partially
    measured catalog makes the map [alpha/Fe]-dependent and then not invertible
    for the rows without it (see the module notes).
    """
    warned = _row(1, flag_warn=True, teff=7000.0, alpha_m=0.1, e_alpha_m=0.02)
    mask, _, table = _run([_row(0), warned])
    assert mask.tolist() == [True, False]
    assert len(table) == 1 and not table["flag_warn"][0]


def test_every_kept_row_has_all_four_parameters():
    warned = [_row(i, flag_warn=True) for i in range(1, 4)]
    _, _, table = _run([_row(0)] + warned)
    assert np.all(np.asarray(table["sigma_x"]) < 1e6)
    assert np.all(np.asarray(table["x_err"]) < 1e6)
    assert np.all(np.isfinite(np.asarray(table["x"])))


def test_uncertainties_are_scaled_before_the_floors():
    _, _, table = _run([_row(0)])
    floors = clam.ERROR_FLOORS  # the catalog's own floors, in quadrature
    assert floors != tuple(config.ERROR_FLOORS)  # and not the defaults
    expected_feh = np.hypot(clam.UNC_SCALES[1] * 0.02, floors[1])
    assert table["sigma_x"][0][1] == pytest.approx(expected_feh, rel=1e-4)
    expected_teff = np.hypot(clam.UNC_SCALES[0] * teff_err_to_x(5000.0, 20.0), floors[0])
    assert table["sigma_x"][0][0] == pytest.approx(expected_teff, rel=1e-4)


def test_the_pipeline_reads_the_floors_under_the_name_it_expects():
    # ``ERROR_FLOORS`` is what ``prep.pipeline.build_table`` looks up; a module
    # that spells it differently (``UNC_FLOOR`` was the mistake) silently falls
    # back to the defaults, which is how this catalog ended up as 44% of the
    # training loss.
    assert hasattr(clam, "ERROR_FLOORS")
    assert not hasattr(clam, "UNC_FLOOR")
    assert len(clam.ERROR_FLOORS) == len(clam.PARAM_COLS) == 4
    # the [Fe/H] floor has to cover the real scatter, ~0.15 dex
    assert clam.ERROR_FLOORS[1] >= 0.1



def test_hot_stars_get_larger_regularization_errors():
    _, _, table = _run([_row(0), _row(1, teff=7000.0)])
    cool, hot = table["x_err"][0], table["x_err"][1]
    assert cool[0] == pytest.approx(clam.X_ERR[0]) and cool[1] == pytest.approx(clam.X_ERR[1])
    assert hot[0] > cool[0] and hot[1] > cool[1]
    assert hot[2] == cool[2] and hot[3] == cool[3]
