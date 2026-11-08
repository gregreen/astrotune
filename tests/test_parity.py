"""Reproducibility tests: prepared tables must match the legacy notebook output.

These run only for catalogs that have already been built with
``astrotune-prepare``; everything else is skipped.
"""

import numpy as np
import pytest
from astropy.table import Table

from astrotune import config, registry
from astrotune.prep import io

BUILT = [spec.key for spec in registry.CATALOGS if spec.legacy_file and config.catalog_path(spec.key).exists()]


def test_at_least_one_catalog_built():
    if not BUILT:
        pytest.skip("no catalogs built yet; run `astrotune-prepare --all`")


@pytest.mark.parametrize("key", BUILT)
def test_table_satisfies_contract(key):
    spec = registry.get_spec(key)
    table = Table.read(config.catalog_path(key))
    problems = io.validate_table(table, key=key, n_params=spec.n_params)
    assert problems == []


@pytest.mark.parametrize("key", BUILT)
def test_metadata_matches_table(key):
    table = Table.read(config.catalog_path(key))
    meta = io.load_meta(key)
    assert meta["n_rows"] == len(table)
    assert meta["n_selected_rows"] == len(table)
    assert meta["n_params"] == table["x"].shape[1]


@pytest.mark.parametrize("key", BUILT)
def test_parity_with_legacy(key):
    legacy = config.LEGACY_TRAINING_DIR / f"{key}.h5"
    if not legacy.exists():
        pytest.skip(f"no legacy table for {key}")

    new = Table.read(config.catalog_path(key))
    reference = Table.read(legacy)
    report = io.parity_report(new, reference)

    assert report["n_new"] == report["n_ref"]
    assert report["ids_match"], io.format_parity(report)
    assert report["same_order"], io.format_parity(report)

    module = registry.get_module(key)
    exempt = getattr(module, "PARITY_EXEMPT", None)
    if exempt:
        # A deliberate deviation must say exactly which parameters of which
        # columns differ; everything else must still match, and the declared
        # differences must be real.
        differs = getattr(module, "PARITY_DIFFERS", None)
        assert differs, f"{key} declares PARITY_EXEMPT but not PARITY_DIFFERS"
        for column in ("x", "x_err", "sigma_x"):
            declared = list(differs.get(column, ()))
            kept = [i for i in range(new[column].shape[1]) if i not in declared]
            assert np.allclose(new[column][:, kept], reference[column][:, kept], atol=1e-5), column
            for i in declared:
                assert not np.allclose(new[column][:, i], reference[column][:, i], atol=1e-5), (column, i)
        pytest.skip(f"intentional deviation: {exempt}")

    for column, stats in report["columns"].items():
        # float32 round-off only
        assert stats["max_abs_diff"] < 1e-5, f"{key}:{column} differs by {stats['max_abs_diff']}"
        assert np.allclose(stats["pct_new"], stats["pct_ref"], rtol=1e-5, atol=1e-5)
