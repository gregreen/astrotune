"""SDSS-V Milky Way Mapper (DR19): ASPCAP parameters from APOGEE spectra.

References
----------
Meszaros et al. (2025), AJ 170, 96
https://sdss.org/dr19/mwm/astra/pipelines-in-astra/aspcap/

Notes
-----
The ``astra`` summary files keep the stellar parameters in HDU 2.

The reduced chi^2 of the spectral fit grows with SNR: the fit uncertainties
keep shrinking while the mismatch between model and data does not, so
``chi2_nu ~ 1 + (Delta/x)^2 SNR^2``. We therefore cut on
``rchi2 / (1 + 1e-4 SNR^2)`` rather than on ``rchi2``; the coefficient
``1e-4`` implies ~1% flux systematics.

The regularization error grows for stars hotter than 6000 K.
"""

from __future__ import annotations

import numpy as np
from astropy.io import fits
from astropy.table import Table

from ..prep import plots
from ..prep.cuts import finite_mask
from ..prep.download import download_sdss
from ..prep.transforms import unmask

KEY = "sdssv_aspcap"
LABEL = "SDSS-V (ASPCAP)"
N_PARAMS = 4
REFERENCE = "Meszaros et al. 2025, AJ 170, 96"
SOURCES = (
    "https://data.sdss5.org/sas/dr19/spectro/astra/0.6.0/summary/"
    "astraAllStarASPCAP-0.6.0.fits.gz",
)
FILENAME = "astraAllStarASPCAP-0.6.0.fits.gz"
HDU = 2

ID_COL = "gaia_dr3_source_id"
PARAM_COLS = ("teff", "fe_h", "logg", "alpha_m_atm")
UNC_COLS = ("e_teff", "e_fe_h", "e_logg", "e_alpha_m_atm")
EXTRA_COLS = ("ra", "dec")

#: Bit mask of the spectrum flags that disqualify a visit.
SPECTRUM_FLAGS_REJECT = 2**0 | 2**2 | 2**19
#: Fractional flux systematic, see the module docstring.
RCHI2_SNR_COEFF = 1e-4
#: Maximum chi^2_nu / (1 + c SNR^2).
RCHI2_MAX = 1e2


def download(ctx) -> None:
    download_sdss(SOURCES[0], ctx.raw_dir / FILENAME)


def load(ctx) -> Table:
    with fits.open(ctx.raw_dir / FILENAME, mode="readonly") as f:
        return Table(f[HDU].data)


def cuts(t: Table, ctx) -> dict:
    teff = unmask(t["teff"])
    fe_h = unmask(t["fe_h"])
    snr = unmask(t["snr"])
    rchi2 = unmask(t["rchi2"])
    spectrum_flags = unmask(t["spectrum_flags"]).astype("i8")

    return {
        "no warning/bad flags": ~(
            unmask(t["flag_warn"]).astype(bool) | unmask(t["flag_bad"]).astype(bool)
        ),
        "parameters finite": finite_mask(t, PARAM_COLS),
        "[Fe/H] in range": (fe_h > -2.3) & (fe_h < 0.9),
        "SNR > 20": snr > 20.0,
        "reduced chi2 below SNR-dependent limit": (
            rchi2 / (1 + RCHI2_SNR_COEFF * snr**2) < RCHI2_MAX
        ),
        "spectrum flags clear": ~(
            (spectrum_flags & SPECTRUM_FLAGS_REJECT).astype(bool)
        ),
        "uncertainties below limits": (
            (unmask(t["e_teff"]) / teff < 0.1)
            & (unmask(t["e_fe_h"]) < 0.3)
            & (unmask(t["e_logg"]) < 0.3)
            & (unmask(t["e_alpha_m_atm"]) < 0.3)
        ),
    }


def param_errs(t: Table, ctx) -> np.ndarray:
    teff = unmask(t["teff"], fill=np.nan).astype("f8")

    hot = np.clip(0.1 + 0.4 * (teff - 6000.0) / 1000.0, 0.1, 1.0)
    hot = np.where(np.isfinite(hot), hot, 0.1)

    n = len(t)
    return np.stack(
        [
            hot,               # Teff
            hot,               # [Fe/H]: ASPCAP metallicities are less certain
            np.full(n, 0.1),   # logg
            np.full(n, 0.05),  # [alpha/Fe]
        ],
        axis=1,
    ).astype("f4")


def diagnostics(t: Table, mask, table: Table, ctx):
    """SNR/chi2 quality plane and uncertainty scaling with SNR."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    keep = np.asarray(mask, dtype=bool)
    snr = unmask(t["snr"], fill=np.nan)
    rchi2 = unmask(t["rchi2"], fill=np.nan)
    reduced = rchi2 / (1 + RCHI2_SNR_COEFF * snr**2)

    good = keep & np.isfinite(snr) & np.isfinite(reduced) & (snr > 0) & (reduced > 0)

    fig, ax = plt.subplots(figsize=(5, 4), layout="constrained")
    ax.hist2d(
        np.log10(snr[good]),
        np.log10(reduced[good]),
        bins=100,
        range=[(0, 4), (-2, 4)],
        norm=LogNorm(),
        rasterized=True,
    )
    ax.axhline(np.log10(RCHI2_MAX), c="w", lw=1, alpha=0.8)
    ax.set_xlabel(r"$\log_{10}\,\mathrm{SNR}$")
    ax.set_ylabel(
        r"$\log_{10}\left[\chi^2_\nu / \left(1+10^{-4}\mathrm{SNR}^2\right)\right]$"
    )
    paths = [plots.save(fig, ctx.diagnostic_dir, "snr_reduced_chi2")]

    # Median uncertainty as a function of SNR.
    log_snr = np.linspace(0.0, 3.5, 30)
    series = {
        "Teff": unmask(t["e_teff"], fill=np.nan) / unmask(t["teff"], fill=np.nan),
        "[Fe/H]": unmask(t["e_fe_h"], fill=np.nan),
        "logg": unmask(t["e_logg"], fill=np.nan),
    }
    log10_snr = np.log10(np.where(snr > 0, snr, np.nan))

    fig, ax = plt.subplots(figsize=(5, 4), layout="constrained")
    for name, values in series.items():
        medians = [
            np.nanmedian(values[np.abs(log10_snr - ls) < 0.1]) for ls in log_snr
        ]
        ax.loglog(10**log_snr, medians, label=name)
    ax.axvline(10**1.75, c="k", ls=":", alpha=0.4)
    ax.set_xlabel("SNR")
    ax.set_ylabel(r"$\sigma$")
    ax.legend()
    paths.append(plots.save(fig, ctx.diagnostic_dir, "uncertainty_vs_snr"))

    return paths
