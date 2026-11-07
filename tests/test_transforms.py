"""Unit tests for the parameter transformations (no data required)."""

import numpy as np

from astrotune.prep import transforms


def test_teff_transform_reference_point():
    assert np.isclose(transforms.teff_to_x(5000.0), 0.0)


def test_teff_transform_is_monotonic():
    teff = np.geomspace(3000.0, 10000.0, 50)
    x = transforms.teff_to_x(teff)
    assert np.all(np.diff(x) > 0)


def test_teff_uncertainty_transform():
    teff, err = 5000.0, 50.0
    expected = 5.0 * err / teff
    assert np.isclose(transforms.teff_err_to_x(teff, err), expected)


def test_error_floor_added_in_quadrature():
    floored = transforms.with_err_floor(lambda v, e: 0.3 * np.ones_like(np.asarray(e, float)), 0.4)
    assert np.isclose(floored(None, np.zeros(3))[0], 0.5)


def test_extract_table_shapes_and_id_filter():
    from astropy.table import Table

    t = Table(
        {
            "gdr3": np.array([10, 20, 0, -5], dtype="i8"),
            "teff": np.array([5000.0, 6000.0, 4000.0, 3000.0]),
            "feh": np.array([0.0, -1.0, 0.5, 0.1]),
            "e_teff": np.array([50.0, 60.0, 40.0, 30.0]),
            "e_feh": np.array([0.05, 0.06, 0.04, 0.03]),
        }
    )
    out = transforms.extract_table(
        t,
        "gdr3",
        ("teff", "feh"),
        ("e_teff", "e_feh"),
        mask=np.array([True, True, True, False]),
    )

    assert len(out) == 2
    assert list(out["id"]) == [10, 20]
    assert out["x"].shape == (2, 2)
    assert out["sigma_x"].shape == (2, 2)
    assert out["x_err"].shape == (2, 2)
    assert np.allclose(out["x_err"], 0.1)


def test_extract_table_uses_full_length_param_errs():
    from astropy.table import Table

    t = Table(
        {
            "id": np.array([1, 2, 3], dtype="i8"),
            "teff": np.array([5000.0, 6000.0, 7000.0]),
            "feh": np.array([0.0, -1.0, 0.5]),
            "e_teff": np.ones(3),
            "e_feh": np.ones(3),
        }
    )
    errs = np.array([[0.11, 0.21], [0.12, 0.22], [0.13, 0.23]])
    out = transforms.extract_table(
        t,
        "id",
        ("teff", "feh"),
        ("e_teff", "e_feh"),
        param_errs=errs,
        mask=np.array([True, False, True]),
    )
    assert np.allclose(out["x_err"], [[0.11, 0.21], [0.13, 0.23]])
