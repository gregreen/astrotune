"""Tests for the two-way street between the bijector and the delta model.

Phase 0 -- the prepared tables carry ``sigma_x_raw`` (quoted, un-floored, un-scaled)
next to ``sigma_x`` (scaled and floored), because the bijector is trained on the
latter and the delta model must learn the former.

Phase 1 -- the delta model corrects the *catalog's own* sigma and pushes it through
the Jacobian, ``S = J diag((s sig)^2 + a^2) J^T``.

Phase 2 -- a fitted delta model can hand a calibrated sigma back to the bijector
(``delta/feed.py``), which is what makes the iteration
bijector -> delta -> bijector converge.
"""

import numpy as np
import pytest
from astropy.table import Table

from astrotune.delta.covariance import entry_cov_np, sigma_out_np

tf = pytest.importorskip("tensorflow")


def test_sigma_x_raw_is_the_quoted_sigma_before_rescale_and_floor():
    """``sigma_x`` is what the bijector sees; ``sigma_x_raw`` what the catalog said."""
    from astrotune.prep.transforms import apply_err_floors, extract_table

    n = 500
    t = Table({"id": np.arange(1, n + 1, dtype="i8"),
               "teff": np.full(n, 5000.0), "feh": np.zeros(n),
               "e_teff": np.full(n, 50.0), "e_feh": np.full(n, 0.2)})
    scales, floor = 3.0, 0.01

    out = extract_table(
        t, "id", ("teff", "feh"), ("e_teff", "e_feh"),
        param_transf=(lambda v: 5 * np.log(v / 5000), lambda v: v),
        param_unc_transf=apply_err_floors(
            (lambda v, e: 5 * e / v, lambda v, e: scales * e), (floor, floor)),
        param_unc_transf_raw=(lambda v, e: 5 * e / v, lambda v, e: e),
    )
    # teff: 5 x 50 / 5000 = 0.05 ; feh: 0.2
    assert np.allclose(out["sigma_x_raw"][:, 0], 0.05, rtol=1e-6)
    assert np.allclose(out["sigma_x_raw"][:, 1], 0.2, rtol=1e-6)
    # sigma_x has the rescale and the floor on top, in quadrature
    assert np.allclose(out["sigma_x"][:, 0], np.hypot(0.05, floor), rtol=1e-6)
    assert np.allclose(out["sigma_x"][:, 1], np.hypot(scales * 0.2, floor), rtol=1e-6)


def test_entry_cov_is_the_jacobian_sandwich():
    """``S = J diag(v) J^T``, with the scale and the additive term in catalog units."""
    rng = np.random.default_rng(0)
    n = 4
    sig = np.full((n, 4), 0.1)
    jac = np.broadcast_to(np.eye(4), (n, 4, 4)).copy()
    measured = np.ones((n, 4), bool)

    plain = entry_cov_np(sig, jac, measured=measured)
    assert np.allclose(plain, np.eye(4) * 0.01)

    # a pure scale of 2 in variance doubles sigma
    scaled = entry_cov_np(sig, jac, np.full((n, 4), np.log(4.0)), measured=measured)
    assert np.allclose(scaled, np.eye(4) * 0.04)
    assert np.allclose(sigma_out_np(scaled)[:, 0], 0.2)

    # the additive term is a variance, added before the Jacobian
    added = entry_cov_np(sig, jac, None, np.log(np.full((n, 4), 0.03**2)),
                         measured=measured)
    assert np.allclose(added, np.eye(4) * (0.01 + 0.0009))

    # a non-trivial Jacobian mixes the axes: diag(J diag(v) J^T) != v
    J = rng.normal(size=(n, 4, 4)) * 0.5 + np.eye(4)
    mixed = entry_cov_np(sig, J, measured=measured)
    assert np.allclose(mixed, J * 0.01 @ np.transpose(J, (0, 2, 1)))
    assert not np.allclose(np.diagonal(mixed, axis1=1, axis2=2), 0.01)

    # an unmeasured axis is cropped and gets the placeholder, not J's structure
    m = measured.copy()
    m[:, 3] = False
    cropped = entry_cov_np(sig, J, measured=m)
    assert np.all(cropped[:, 3, :3] == 0.0) and np.all(cropped[:, :3, 3] == 0.0)
    assert np.all(cropped[:, 3, 3] == 1e9)
    assert np.isnan(sigma_out_np(cropped)[0, 3])


def test_calibrated_sigma_reproduces_the_prepared_one_when_the_model_says_nothing():
    """Round trip: a delta model that encodes today's floors must give today's sigma."""
    import tempfile
    from pathlib import Path

    import astrotune.delta.feed as feed
    from astrotune.delta.model import ADD_MIN_SIGMA, ConstScaleAdd

    rng = np.random.default_rng(1)
    n = 300
    sig_raw = np.exp(rng.normal(np.log(0.08), 0.3, size=(n, 4))).astype("f4")
    floor = np.array([0.01, 0.01, 0.04, 0.02])
    table = Table({"id": np.arange(1, n + 1, dtype="i8"),
                   "x": np.zeros((n, 4), "f4"),
                   "sigma_x": np.hypot(sig_raw, floor).astype("f4"),
                   "sigma_x_raw": sig_raw})

    model = ConstScaleAdd(1)  # scale 1, additive term on its floor
    with tempfile.TemporaryDirectory() as tmp, pytest.MonkeyPatch.context() as mp:
        mp.setattr("astrotune.prep.io.load_catalog", lambda key: table)
        got = feed.calibrated_sigma(".", keys=("fake",), out_dir=Path(tmp),
                                    model=model)["fake"]

    want = np.hypot(sig_raw, ADD_MIN_SIGMA)  # what the model adds, nothing else
    assert got.shape == sig_raw.shape
    assert np.allclose(got, want, rtol=2e-3)
    assert np.all(got >= ADD_MIN_SIGMA * 0.99)
