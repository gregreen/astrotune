"""The covariance on its way from the prepared table into the observations.

Two things have to hold for the whitening factor to be worth anything. It must
survive every step between the prepared table and the trainer -- the column
selection, the dedupe, the sigma replacement a delta model does -- and it must
still mean the same thing after a delta model has rewritten ``sigma_x``, which
it can only do if the block is re-marginalized onto the new sigma.
"""

import h5py
import numpy as np
from astropy.table import Table

from astrotune import training_data
from astrotune.prep import transforms


def _table(n=8, seed=1):
    """A prepared-looking table: parameters, sigma, and a correlated block."""
    rng = np.random.default_rng(seed)
    sigma = np.exp(rng.normal(scale=0.3, size=(n, 4))) * 0.1
    a = rng.normal(scale=0.5, size=(n, 4, 4))
    cov = np.einsum("nij,nkj->nik", a, a)
    cov += np.eye(4)[None] * np.exp(rng.normal(-4.0, 0.2, size=(n, 4, 1)))
    cov = transforms.covariance_with_sigma(cov, sigma)
    return Table(
        dict(
            id=np.arange(100, 100 + n, dtype="i8"),
            x=rng.normal(size=(n, 4)).astype("f4"),
            sigma_x=sigma.astype("f4"),
            sigma_x_raw=sigma.astype("f4"),
            x_err=np.full((n, 4), 0.1, dtype="f4"),
            cov_x=cov.astype("f4"),
        )
    )


def test_the_column_selection_keeps_the_covariance():
    """``load_tables`` keeps only the columns the observations need. Losing this
    one would be silent: every catalog would simply come out diagonal."""
    assert "cov_x" in training_data._COLUMNS


def test_extract_observations_whitens_from_the_block():
    t = _table()
    res = training_data.extract_observations(t, 0, np.asarray(t["id"]))
    assert res["u"].shape == (len(t), transforms.WHITENING_SIZE)
    np.testing.assert_allclose(
        res["u"],
        transforms.covariance_to_whitening(np.asarray(t["cov_x"])),
        rtol=1e-6,
    )
    # A table whose block happened to be diagonal would pass the line above
    # while testing nothing, so pin that this one is not.
    assert not np.allclose(res["u"], training_data.identity_whitening(len(t)))


def test_extract_observations_without_a_block_is_the_identity():
    t = _table()
    del t["cov_x"]
    res = training_data.extract_observations(t, 0, np.asarray(t["id"]))
    np.testing.assert_array_equal(
        res["u"], training_data.identity_whitening(len(t))
    )


def test_replacing_sigma_keeps_the_block_consistent():
    """A delta model's calibrated sigma is a new marginal; the block has to
    follow it, or the likelihood would weight the parameters by one sigma and
    whiten them by another."""
    t = _table()
    new = np.full((len(t), 4), 0.3)
    out = training_data.replace_sigma(t, new, "test")

    np.testing.assert_allclose(np.asarray(out["sigma_x"]), new, rtol=1e-6)
    cov = np.asarray(out["cov_x"], dtype="f8")
    np.testing.assert_allclose(transforms.covariance_to_sigma(cov), new, rtol=1e-6)
    # The correlations are the catalog's own, and survive the change of scale.
    np.testing.assert_allclose(
        transforms.covariance_to_correlation(cov),
        transforms.covariance_to_correlation(np.asarray(t["cov_x"], dtype="f8")),
        rtol=1e-6,
        atol=1e-7,
    )


def test_saving_an_obs_dict_without_u_writes_the_identity(tmp_path):
    """A hand-built or legacy ``obs_dict`` means "all diagonal". The writer makes
    the same substitution the reader does, so neither side has to know which kind
    of file it is looking at."""
    obs = {
        "id": np.array([1, 2], dtype="i8"),
        "c": np.zeros(2, "i4"),
        "x": np.zeros((2, 4), "f4"),
        "xe": np.full((2, 4), 0.1, "f4"),
        "xs": np.full((2, 4), 0.1, "f4"),
    }
    ctrl = {
        "x_control": np.zeros((1, 4), "f4"),
        "xe_control": np.full((1, 4), 0.1, "f4"),
        "c_control": np.zeros(1, "i4"),
        "from_catalog": np.zeros(1, bool),
    }
    path = training_data.save_observations(obs, ctrl, tmp_path / "obs.h5")
    loaded, _ = training_data.load_observations(path)
    np.testing.assert_array_equal(loaded["u"], training_data.identity_whitening(2))


def test_an_old_observations_file_loads_with_the_identity(tmp_path):
    """Files built before covariances carry no ``u``. Every catalog in them is
    diagonal, so the identity is the exact answer rather than a fallback."""
    path = tmp_path / "old.h5"
    n = 4
    with h5py.File(path, "w") as f:
        f.create_dataset("obs_dict/id", data=np.arange(n, dtype="i8"))
        f.create_dataset("obs_dict/x", data=np.zeros((n, 4), "f4"))
        f.create_dataset("obs_dict/xe", data=np.full((n, 4), 0.1, "f4"))
        f.create_dataset("obs_dict/xs", data=np.full((n, 4), 0.1, "f4"))
        f.create_dataset("obs_dict/c", data=np.zeros(n, "i4"))
        f.create_dataset("ctrl_dict/x_control", data=np.zeros((1, 4), "f4"))
        f.create_dataset("ctrl_dict/xe_control", data=np.full((1, 4), 1e9, "f4"))
        f.create_dataset("ctrl_dict/c_control", data=np.zeros(1, "i4"))
        f.create_dataset("ctrl_dict/from_catalog", data=np.zeros(1, bool))

    obs, _ = training_data.load_observations(path)
    np.testing.assert_array_equal(obs["u"], training_data.identity_whitening(n))
