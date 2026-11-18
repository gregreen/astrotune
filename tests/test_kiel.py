"""Tests for the per-catalog Kiel maps of the learned extra variance."""

import numpy as np
import pytest

from astrotune.delta import kiel

tf = pytest.importorskip("tensorflow")


def _table(n_star=400, n_cat=3, seed=0):
    """Stars in every catalog, with a spread of Teff, logg and [Fe/H]."""
    rng = np.random.default_rng(seed)
    n, K = n_star, n_cat
    x = np.empty((n, K, 4), dtype="f4")
    x[..., 0] = rng.uniform(-1.5, 4.0, (n, K))   # 5 ln(Teff/5000 K)
    x[..., 1] = rng.uniform(-2.2, 0.3, (n, K))   # [Fe/H]
    x[..., 2] = rng.uniform(0.5, 5.0, (n, K))    # logg
    x[..., 3] = 0.0
    ls = np.full((n, K, 4), np.log(0.1), dtype="f4")
    m = np.ones((n, K, 4), dtype=bool)
    m[..., 3] = False  # nobody measures [alpha/Fe]
    out = {k: np.ascontiguousarray(v.reshape(n * K, *v.shape[2:])) for k, v in
           dict(x=x, ls=ls, m=m).items()}
    out["c"] = np.tile(np.arange(K, dtype="i4"), n)
    out["g"] = out["c"].copy()
    out["id"] = np.arange(n, dtype="i8") + 1
    out["n_entry"] = np.full(n, K, dtype="i4")
    out["off"] = np.r_[0, np.cumsum(out["n_entry"])].astype("i8")
    out["seg"] = np.repeat(np.arange(n, dtype="i4"), K)
    return out


def _stub(scales):
    """A delta model that returns one variance scale per catalog and no additive term."""
    ln = tf.constant(np.log(scales).astype("f4"))

    def model(x, ls, c):
        log_scale = tf.tile(tf.gather(ln, tf.cast(c, tf.int32))[:, None], [1, 4])
        return log_scale, tf.fill(tf.shape(x), tf.constant(-60.0, tf.float32))

    return model


@pytest.fixture
def cube(monkeypatch):
    stars = _table()
    keys = ("a", "b", "c")
    monkeypatch.setattr(kiel, "get_star_table", lambda run_dir, out_dir=None: (stars, keys))
    scales = [4.0, 1.0, 0.25]  # sigma x2, x1, x0.5
    return kiel.build_kiel_cube("run", model=_stub(scales), sample="model",
                                keys=keys, n_bins=(20, 20)), scales


def test_cube_recovers_the_known_scale(cube):
    c, scales = cube
    assert c.log_ratio.shape == (3, 4, 4, 20, 20)
    assert c.sigma_add.shape == c.log_ratio.shape
    for k, scale in enumerate(scales):
        got = c.log_ratio[k, 0][np.isfinite(c.log_ratio[k, 0])]
        assert got.size > 100
        assert np.allclose(got, 0.5 * np.log(scale), atol=1e-4)  # ln(sigma_out/sigma_in)
        add = c.sigma_add[k, 0][np.isfinite(c.sigma_add[k, 0])]
        assert np.all(add < 1e-6)  # no additive term in the stub
    # the [alpha/Fe] column is empty everywhere: nothing measures it
    assert not c.measures[:, 3].any()
    assert np.all(~np.isfinite(c.log_ratio[:, 3]))


def test_empty_bins_are_masked_and_counts_agree(cube):
    c, _ = cube
    empty = ~np.isfinite(c.log_ratio[0, 0])
    assert empty.any() and (c.count[0, 0][empty] == 0).all()
    filled = np.isfinite(c.log_ratio[0, 0])
    assert (c.count[0, 0][filled] > 0).all()


def test_all_parameters_are_carried(cube):
    c, _ = cube
    for p in range(3):  # Teff, [Fe/H], logg are measured by every entry
        assert c.measures[:, p].all()
        assert np.isfinite(c.log_ratio[:, p]).any()


def test_the_figure_has_one_layout_and_one_colourbar(cube, tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    c, _ = cube
    seen = {}
    real = plt.Figure.savefig

    def spy(self, *a, **k):
        seen["axes"] = len(self.axes)
        seen["xlim"] = [ax.get_xlim() for ax in self.axes[:16]]
        seen["ylim"] = [ax.get_ylim() for ax in self.axes[:16]]
        seen["titles"] = [ax.get_title() for ax in self.axes[:16]]
        seen["ylabels"] = [self.axes[4 * r].get_ylabel() for r in range(4)]
        return real(self, *a, **k)

    monkeypatch.setattr(plt.Figure, "savefig", spy)
    lim = kiel._ratio_limit(c)
    path = kiel.plot_catalog_cube(c, 0, "Catalog A", tmp_path / "a.png", -lim, lim)
    assert path.exists() and path.stat().st_size > 0
    assert seen["axes"] == 17  # 16 panels + the colourbar
    assert len(set(map(tuple, seen["xlim"]))) == 1  # identical frame everywhere
    assert len(set(map(tuple, seen["ylim"]))) == 1
    assert seen["titles"][:4] == list(kiel.PARAM_NAMES)  # a column title per parameter
    # [Fe/H] descends from the top row to the bottom row
    lo = [float(label.split("\n")[-1].split("…")[0]) for label in seen["ylabels"]]
    assert lo == sorted(lo, reverse=True), seen["ylabels"]
    assert "-0.25" in seen["ylabels"][0] and "-2.50" in seen["ylabels"][-1]


def test_the_additive_page_uses_the_same_layout(cube, tmp_path):
    c, _ = cube
    vmin, vmax = kiel._add_scale(c, 0)
    path = kiel.plot_catalog_cube(c, 0, "Catalog A", tmp_path / "a_add.png",
                                  vmin, vmax, kind="add")
    assert path.exists() and path.stat().st_size > 0


def test_all_sample_reads_the_prepared_catalogs(monkeypatch):
    """`--all` streams the prepared tables through the same encoding as training."""
    from astropy.table import Table

    stars = _table(n_star=50)
    keys = ("a", "b", "c")
    monkeypatch.setattr(kiel, "get_star_table", lambda run_dir, out_dir=None: (stars, keys))

    seen = {}

    def fake_catalog(key):
        n = 300
        rng = np.random.default_rng(hash(key) % 2**32)
        x = np.empty((n, 4), "f4")
        x[:, 0] = rng.uniform(-1.5, 4.0, n)
        x[:, 1] = rng.uniform(-2.2, 0.3, n)
        x[:, 2] = rng.uniform(0.5, 5.0, n)
        x[:, 3] = 0.0
        sigma = np.full((n, 4), 0.1, "f4")
        sigma[:, 3] = 1e9  # [alpha/Fe] not measured
        seen[key] = sigma
        return Table({"id": np.arange(n, dtype="i8") + 1, "x": x, "sigma_x": sigma})

    monkeypatch.setattr("astrotune.training_data.load_catalog", fake_catalog)
    scales = [4.0, 1.0, 0.25]
    c = kiel.build_kiel_cube("run", model=_stub(scales), sample="all",
                             keys=keys, n_bins=(10, 10))
    for k, scale in enumerate(scales):
        got = c.log_ratio[k, 0][np.isfinite(c.log_ratio[k, 0])]
        assert got.size > 20 and np.allclose(got, 0.5 * np.log(scale), atol=1e-4)
    assert not c.measures[:, 3].any()
    assert np.all(seen[keys[0]][:, 3] == 1e9)  # the mask came from sigma_x, as in training
