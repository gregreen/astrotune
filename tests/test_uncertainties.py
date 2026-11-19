"""Tests for the quoted-vs-calibrated uncertainty figures."""

import numpy as np
import pytest

from astrotune.delta import uncertainties as un

tf = pytest.importorskip("tensorflow")


def _table(n_star=400, n_cat=2, sigma=0.03, alpha=True):
    """Stars in every catalog, with the same quoted sigma for all four parameters."""
    rng = np.random.default_rng(0)
    n, K = n_star, n_cat
    x = rng.normal(size=(n, K, 4))
    ls = np.full((n, K, 4), np.log(sigma), dtype="f4")
    m = np.ones((n, K, 4), dtype=bool)
    if not alpha:
        m[..., 3] = False  # nobody measures [alpha/Fe]
    C = np.broadcast_to(np.eye(4) * sigma**2, (n, K, 4, 4))
    sig = np.where(m, sigma, 0.0)
    out = {k: np.ascontiguousarray(v.reshape(n * K, *v.shape[2:]).astype("f4"))
           for k, v in dict(x=x, ls=ls, sig=sig, sig0=sig, y=n / 2 + 0 * x, C=C).items()}
    out["J"] = np.ascontiguousarray(
        np.broadcast_to(np.eye(4, dtype="f4"), (n * K, 4, 4)).copy())
    out["m"] = np.ascontiguousarray(m.reshape(n * K, 4))
    out["c"] = np.tile(np.arange(K, dtype="i4"), n)
    out["g"] = out["c"].copy()
    out["id"] = np.arange(n, dtype="i8") + 1
    out["n_entry"] = np.full(n, K, dtype="i4")
    out["off"] = np.r_[0, np.cumsum(out["n_entry"])].astype("i8")
    out["seg"] = np.repeat(np.arange(n, dtype="i4"), K)
    return out


def _stub(scale, add_sigma):
    """A model with one variance scale and one additive sigma, the same everywhere."""
    def model(x, ls, c):
        return (tf.fill(tf.shape(x), tf.constant(np.log(scale), tf.float32)),
                tf.fill(tf.shape(x), tf.constant(2 * np.log(add_sigma), tf.float32)))

    return model


def test_the_calibrated_uncertainty_is_the_congruence_on_the_diagonal(monkeypatch):
    sigma, scale, add = 0.03, 4.0, 0.01
    stars = _table(sigma=sigma)
    monkeypatch.setattr(un, "get_star_table", lambda run_dir, out_dir=None: (stars, ("a", "b")))
    dists = un.uncertainty_distributions("run", model=_stub(scale, add), sample="train",
                                         keys=("a", "b"))
    raw, cal, measured = dists[0]
    assert np.allclose(raw[:, 0], sigma, rtol=1e-6)
    # S_ii = scale * C_ii + add^2, so sigma_out = sqrt(scale sigma^2 + add^2)
    assert np.allclose(cal[:, 0], np.sqrt(scale * sigma**2 + add**2), rtol=1e-6)
    assert measured.all()


def test_one_axis_per_parameter_the_catalog_measures(tmp_path, monkeypatch):
    from astrotune import registry

    keys = tuple(registry.keys()[:2])  # the titles come from the catalog specs
    stars = _table(n_star=300, alpha=False)  # no [alpha/Fe] anywhere
    monkeypatch.setattr(un, "get_star_table", lambda run_dir, out_dir=None: (stars, keys))
    paths = un.make_uncertainty_plots("run", tmp_path, model=_stub(2.0, 0.005),
                                      sample="train", keys=keys, min_entries=10)
    assert len(paths) == 2 and all(p.exists() for p in paths)

    import matplotlib.pyplot as plt

    seen = {}
    real = plt.Figure.savefig

    def spy(self, *a, **k):
        seen["n_axes"] = len(self.axes)
        return real(self, *a, **k)

    monkeypatch.setattr(plt.Figure, "savefig", spy)
    raw, cal, measured = un.uncertainty_distributions(
        "run", model=_stub(2.0, 0.005), sample="train", keys=keys)[0]
    un.plot_catalog_uncertainty(0, "Catalog A", raw, cal, measured, tmp_path / "x.png")
    assert seen["n_axes"] == 3  # Teff, [Fe/H], logg -- no empty [alpha/Fe] panel


def test_a_parameter_with_too_few_entries_is_dropped(tmp_path, monkeypatch):
    stars = _table(n_star=400, alpha=True)
    stars["m"][stars["c"] == 1, 3] = False
    monkeypatch.setattr(un, "get_star_table", lambda run_dir, out_dir=None: (stars, ("a", "b")))
    raw, cal, measured = un.uncertainty_distributions(
        "run", model=_stub(1.0, 0.003), sample="train", keys=("a", "b"))[1]
    assert measured[:, 3].sum() == 0
    assert un.plot_catalog_uncertainty(1, "Catalog B", raw, cal, measured,
                                       tmp_path / "b.png", min_entries=10) is not None
    assert un.plot_catalog_uncertainty(1, "Catalog B", raw, cal, measured,
                                       tmp_path / "c.png", min_entries=10**9) is None
