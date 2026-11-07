"""APOGEE DR17 (SDSS-IV), ASPCAP stellar parameters.

References
----------
ASPCAP: Garcia Perez et al. (2016), AJ 151, 144
SDSS DR17: Abdurro'uf et al. (2022), ApJS 259, 35

Notes
-----
``[alpha/M]`` from ASPCAP is used as the alpha abundance (rather than an
inverse-variance mean of the individual ``*_FE`` abundances, which is only
computed for a diagnostic comparison).

The regularization error grows for stars hotter than 6000 K, where the
parameter determination is less reliable.
"""

from __future__ import annotations

import numpy as np
from astropy.table import Table

from ..prep import plots
from ..prep.abundances import calc_alpha_Fe_from_abundances
from ..prep.cuts import finite_mask
from ..prep.download import download as download_file
from ..prep.transforms import unmask

KEY = "apogee_dr17"
LABEL = "APOGEE (DR17)"
N_PARAMS = 4
REFERENCE = "ASPCAP: Garcia Perez et al. 2016; SDSS DR17: Abdurro'uf et al. 2022"
SOURCES = (
    "https://data.sdss.org/sas/dr17/apogee/spectro/aspcap/dr17/synspec_rev1/"
    "allStarLite-dr17-synspec_rev1.fits",
)
FILENAME = "allStarLite-dr17-synspec_rev1.fits"

ID_COL = "GAIAEDR3_SOURCE_ID"
PARAM_COLS = ("TEFF", "M_H", "LOGG", "ALPHA_M")
UNC_COLS = ("TEFF_ERR", "M_H_ERR", "LOGG_ERR", "ALPHA_M_ERR")
EXTRA_COLS = ("RA", "DEC")

# STAR_WARN bit in ASPCAPFLAG
STAR_WARN_BIT = 2**7


def download(ctx) -> None:
    download_file(SOURCES[0], ctx.raw_dir / FILENAME)


def load(ctx) -> Table:
    return Table.read(ctx.raw_dir / FILENAME)


def cuts(t: Table, ctx) -> dict:
    flag = unmask(t["ASPCAPFLAG"]).astype("i8")
    alpha = unmask(t["ALPHA_M"])

    teff_err = unmask(t["TEFF_ERR"])
    logg_err = unmask(t["LOGG_ERR"])
    m_h_err = unmask(t["M_H_ERR"])
    alpha_err = unmask(t["ALPHA_M_ERR"])

    return {
        "STAR_WARN flag clear": (flag & STAR_WARN_BIT) == 0,
        "parameters finite": finite_mask(t, PARAM_COLS),
        "uncertainties finite": finite_mask(t, UNC_COLS),
        "uncertainties below limits": (
            (teff_err < 500.0)
            & (logg_err < 0.5)
            & (m_h_err < 0.5)
            & (alpha_err < 0.5)
        ),
        "alpha/M in range": (alpha > -0.4) & (alpha < 0.7),
    }


def param_errs(t: Table, ctx) -> np.ndarray:
    teff = unmask(t["TEFF"], fill=np.nan).astype("f8")

    teff_err = np.clip(0.1 + 0.4 * (teff - 6000.0) / 1000.0, 0.1, np.inf)
    teff_err = np.where(np.isfinite(teff_err), teff_err, 0.1)

    n = len(t)
    return np.stack(
        [
            teff_err,
            np.full(n, 0.1),
            np.full(n, 0.1),
            np.full(n, 0.05),
        ],
        axis=1,
    ).astype("f4")


def diagnostics(t: Table, mask, table: Table, ctx):
    """Compare ASPCAP's [alpha/M] with a mean of the individual abundances."""
    paths = []
    try:
        alpha_fe, alpha_fe_err, abundances = calc_alpha_Fe_from_abundances(
            t,
            "{x.upper()}_FE",
            "{x.upper()}_FE_ERR",
            flag_fmt="{x.upper()}_FE_FLAG",
        )
    except Exception as exc:  # pragma: no cover - depends on columns present
        print(f"[{KEY}] skipping abundance diagnostics: {exc}")
        return paths

    import matplotlib.pyplot as plt

    alpha_m = unmask(t["ALPHA_M"], fill=np.nan)
    good = np.isfinite(alpha_m) & np.isfinite(alpha_fe) & np.asarray(mask, dtype=bool)

    fig, ax = plt.subplots(figsize=(4.5, 4.5), layout="constrained")
    ax.hist2d(
        alpha_m[good],
        alpha_fe[good],
        range=[[-0.7, 0.7], [-0.7, 0.7]],
        bins=100,
        norm=plt.matplotlib.colors.LogNorm(),
        rasterized=True,
    )
    ax.axline((0, 0), slope=1, c="gray", alpha=0.5)
    ax.set_xlabel("ALPHA_M")
    ax.set_ylabel(r"$\left[\alpha/\mathrm{Fe}\right]$")
    ax.set_aspect("equal")
    paths.append(plots.save(fig, ctx.diagnostic_dir, "alpha_M_vs_alpha_Fe"))

    paths.append(
        plots.plot_alpha_abundances(
            alpha_fe[good], abundances[good], outdir=ctx.diagnostic_dir
        )
    )
    return paths
