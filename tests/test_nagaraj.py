"""Tests for the DESI (Nagaraj) candidate catalog and the overlap tool."""

import numpy as np
import pytest
from astropy.table import Table

from astrotune import registry
from astrotune.catalogs import desi_nagaraj2026 as nag
from astrotune.prep import overlaps, pipeline
from astrotune.prep.transforms import teff_to_x


def _good_row(i):
    return dict(
        GAIA_SOURCE_ID=1000 + i, SNR_MED=50.0, CHI2_FLAG=0.0,
        TEFF=5500, TEFF_ERR=30, LOGG=4.2, LOGG_ERR=0.1,
        FEH_RAW=-0.8, FEH=-0.4, FEH_ERR=0.04, VMIC=1.3,
        ALPHA_FE=0.2, ALPHA_FE_ERR=0.03,
    )


def test_catalog_is_registered():
    assert nag.KEY in registry.keys()
    assert registry.get_spec(nag.KEY).label == nag.LABEL
    assert nag.KEY in registry.INDEX


def test_cuts_remove_each_kind_of_bad_star():
    # Row 0 is good; every other row violates exactly one cut
    bad = {
        "GAIA_SOURCE_ID": -1,
        "SNR_MED": 10.0,
        "CHI2_FLAG": 5.0,
        "TEFF_ERR": -9999,  # failed uncertainty estimate
        "LOGG_ERR": np.nan,
        "VMIC": nag.VMIC_STUCK,  # fit never left its starting point
        "TEFF": 4010,  # at the lower Teff edge of the training grid
        "LOGG": 5.0,  # at the upper logg edge
        "FEH_RAW": -2.5,  # at the lower [Fe/H] edge
        "ALPHA_FE": 2.0,
        "FEH_ERR": 0.5,
    }
    rows = [_good_row(0)]
    for i, (col, value) in enumerate(bad.items(), start=1):
        row = _good_row(i)
        row[col] = value
        rows.append(row)
    t = Table({k: [r[k] for r in rows] for k in rows[0]})

    mask, counts = pipeline.evaluate_cuts(t, nag, pipeline.make_context(nag.KEY, make_plots=False))
    assert mask.tolist() == [True] + [False] * len(bad)
    assert list(counts)[-1] == "uncertainties below limits"


def test_x_to_teff_inverts_the_transformation():
    teff = np.array([4050.0, 5000.0, 7900.0])
    assert np.allclose(overlaps.x_to_teff(teff_to_x(teff)), teff)


def test_overlap_counts_pairs_and_stars(monkeypatch, tmp_path):
    def table(ids, teff, alpha_measured):
        """``alpha_measured`` per row; unmeasured parameters have sigma_x = 1e9."""
        n = len(ids)
        sigma = np.full((n, 4), 0.05)
        sigma[:, 3] = np.where(alpha_measured, 0.05, 1e9)
        x = np.c_[teff_to_x(np.asarray(teff, "f8")), np.zeros((n, 3))]
        return Table({"id": np.asarray(ids), "x": x, "sigma_x": sigma})

    tables = {
        # candidate: id 2 twice; [alpha/Fe] missing for id 3
        "a": table([1, 2, 2, 3], [4200, 5000, 5000, 6000], [True, True, True, False]),
        # id 2 twice (second row without [alpha/Fe]), id 3 once
        "b": table([2, 2, 3, 9], [5000, 5000, 5000, 5000], [True, False, True, True]),
        # a 3-parameter catalog: no [alpha/Fe] column at all
        "c": Table({"id": np.array([2, 3]), "x": np.zeros((2, 3)), "sigma_x": np.full((2, 3), 0.05)}),
    }
    monkeypatch.setattr(overlaps, "load_catalog", lambda key: tables[key])
    monkeypatch.setattr(overlaps.config, "catalog_path", lambda key: tmp_path)  # "exists"
    monkeypatch.setattr(overlaps.registry, "get_spec", lambda key: type("S", (), {"label": key})())

    res = overlaps.overlap_counts("a", partner_keys=["b", "c"], teff_edges=(4000, 4600, 5500, 8000))
    b, c = res["partners"]
    assert b["n_stars"] == 2  # ids 2 and 3
    assert b["n_pairs"] == 2 * 2 + 1 * 1  # many-to-many join: id 2 -> 4 pairs, id 3 -> 1
    assert b["n_stars_by_teff"] == [0, 1, 1]  # Teff 5000 K and 6000 K
    # [alpha/Fe] in both: id 2 (a has it, b has it in one row); not id 3 (missing in a)
    assert b["n_stars_alpha"] == 1
    assert b["n_stars_alpha_by_teff"] == [0, 1, 0]
    # A catalog without [alpha/Fe] never constrains it
    assert c["n_stars"] == 2 and c["n_stars_alpha"] == 0
