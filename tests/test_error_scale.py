"""Tests for the uncertainty-scale check (astrotune.prep.error_scale)."""

import numpy as np
import pytest

from astrotune import config, registry
from astrotune.prep import error_scale as es


def test_quoted_sigma_undoes_the_scale_and_the_floor():
    from astrotune.catalogs import lamost_mstar_li2025 as li, sdssv_clam as clam

    quoted = np.array([[0.02, 0.03, 0.05, 0.01]])
    for key, mod in (("sdssv_clam", clam), ("lamost_mstar_li2025", li)):
        scales, floors = es.module_scales_and_floors(key, 4)
        assert np.allclose(scales, mod.UNC_SCALES)  # what the module declares is what the tool undoes
        assert np.allclose(floors, getattr(mod, "ERROR_FLOORS", config.ERROR_FLOORS))
        prepared = np.sqrt((scales * quoted) ** 2 + floors**2)
        assert np.allclose(es.quoted_sigma(key, prepared), quoted)


def test_fit_recovers_the_scale_and_the_plateau():
    rng = np.random.default_rng(0)
    n = 40000
    q = np.exp(rng.uniform(np.log(0.01), np.log(0.1), n))  # quoted sigma
    sp = np.full(n, 0.02)  # partner sigma
    s_true, sys_true = 3.0, 0.05
    d = rng.normal(size=n) * np.sqrt((s_true * q) ** 2 + sys_true**2 + sp**2)
    edges = np.quantile(q, np.linspace(0, 1, 9))
    edges[-1] += 1e-9
    s, sys = es.fit_scale(es.bin_stats(d, q, sp, edges))
    assert s == pytest.approx(s_true, rel=0.1) and sys == pytest.approx(sys_true, rel=0.15)


def test_robust_scale_ignores_outliers():
    v = np.r_[np.random.default_rng(1).normal(size=10000), np.full(200, 100.0)]
    assert es.robust_scale(v) == pytest.approx(1.0, rel=0.05)


def test_scaled_catalogs_have_default_partners():
    for key in ("sdssv_clam", "lamost_mstar_li2025"):
        assert key in es.DEFAULT_PARTNERS and all(k in registry.keys() for k in es.DEFAULT_PARTNERS[key])
