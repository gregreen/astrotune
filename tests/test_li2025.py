"""Tests for the LAMOST M-star (Li et al. 2025) candidate catalog."""

import numpy as np
import pytest
from astropy.table import Table

from astrotune import registry
from astrotune.catalogs import lamost_mstar_li2025 as li
from astrotune.prep import pipeline
from astrotune.prep.transforms import teff_err_to_x, teff_to_x


def _row(i, **changes):
    row = dict(
        source_id=1000 + i, ra=10.0, dec=20.0, parameter_flag=1,
        snr_u=5.0, snr_g=10.0, snr_r=30.0, snr_i=40.0, snr_z=35.0,  # SNR_tot ~ 61
        luminosity_class="dM", CMD_pos="dM", subclass="dM2",
        teff=3800.0, m_h=-0.2, logg=4.8, alpha_m=0.05,
        teff_err=20.0, m_h_err=0.05, logg_err=0.05, alpha_m_err=0.015,
    )
    row.update(changes)
    return row


def _run(rows):
    t = Table({k: [r[k] for r in rows] for k in rows[0]})
    ctx = pipeline.make_context(li.KEY, make_plots=False)
    mask, counts = pipeline.evaluate_cuts(t, li, ctx)
    return t, mask, counts, pipeline.build_table(t, mask, li, ctx)


def test_catalog_is_registered():
    assert li.KEY in registry.keys()
    assert "dr12" not in li.KEY  # the spectra are DR10, not the official DR12 catalog


def test_total_snr_ignores_the_negative_sentinel():
    t = Table(
        {
            "snr_u": [5.0, -9999.0, -9999.0],
            "snr_g": [5.0, -9999.0, -9999.0],
            "snr_r": [5.0, 5.0, -9999.0],
            "snr_i": [5.0, 5.0, -9999.0],
            "snr_z": [5.0, 5.0, -9999.0],
        }
    )
    # Negative values count as zero, so the sentinel never inflates the sum (~1e4)
    assert li.total_snr(t) == pytest.approx([np.sqrt(5 * 25), np.sqrt(3 * 25), 0.0])


def test_each_cut_removes_its_star():
    violations = [
        dict(source_id=-1),
        dict(parameter_flag=0),
        dict(snr_u=3.0, snr_g=3.0, snr_r=10.0, snr_i=12.0, snr_z=11.0),  # SNR_tot ~ 19
        # Two bands with the -9999 sentinel: summing squares naively would give ~1e4
        dict(snr_u=-9999.0, snr_g=-9999.0, snr_r=5.0, snr_i=5.0, snr_z=5.0),
        dict(CMD_pos="other"),
        dict(luminosity_class="gM_checked", CMD_pos="dM"),  # LAMOST giant, CMD dwarf
        dict(luminosity_class="dM_checked", CMD_pos="gM"),  # LAMOST dwarf, CMD giant
        dict(teff_err=np.nan),
        dict(logg_err=li.MAX_ERR["logg_err"] + 0.1),
    ]
    rows = [_row(0)] + [_row(i, **v) for i, v in enumerate(violations, start=1)]
    _, mask, _, table = _run(rows)
    assert mask.tolist() == [True] + [False] * len(violations)
    assert len(table) == 1


def test_stars_without_cmd_information_and_giants_are_kept():
    rows = [
        _row(0, CMD_pos=""),  # no Gaia astrometry
        _row(1, luminosity_class="gM_checked", CMD_pos="gM", logg=0.8, teff=3700.0),
    ]
    _, mask, _, _ = _run(rows)
    assert mask.tolist() == [True, True]


def test_parameter_mapping_and_scaled_uncertainties():
    _, _, _, table = _run([_row(0)])
    x, sigma = table["x"][0], table["sigma_x"][0]
    assert x == pytest.approx([teff_to_x(3800.0), -0.2, 4.8, 0.05], abs=1e-5)  # [M/H] -> slot 1
    floors = (0.01, 0.01, 0.04, 0.05)
    quoted = (teff_err_to_x(3800.0, 20.0), 0.05, 0.05, 0.015)
    expected = [np.hypot(li.UNC_SCALES[i] * quoted[i], floors[i]) for i in range(4)]
    assert sigma == pytest.approx(expected, rel=1e-4)
    # All four parameters are measured for every star
    assert np.all(sigma < 1e6) and np.all(table["x_err"][0] == pytest.approx(li.X_ERR))
