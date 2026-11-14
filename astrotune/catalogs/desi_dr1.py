"""DESI DR1 Milky Way Survey, radial-velocity stellar parameters.

References
----------
Milky Way Survey catalog of stellar radial velocities and stellar parameters
(DESI DR1 VAC): https://data.desi.lbl.gov/doc/releases/dr1/vac/mws/
Koposov et al. (2024), MNRAS 533, 1012

Notes
-----
The three per-program tables (bright, dark, backup) are stacked and Gaia DR3
identifiers are read from the ``GAIA`` HDU of each file.

The regularization error grows both for hot stars and toward the
cool-dwarf edge of the parameter space, where the DESI parameters are less
reliable (see the ``teff_edge`` metric below).
"""

from __future__ import annotations

import astropy.table
import numpy as np
from astropy.io import fits
from astropy.table import Table

from ..prep import plots
from ..prep.cuts import snr_chisq_relation
from ..prep.download import download as download_file
from ..prep.transforms import unmask

KEY = "desi_dr1"
LABEL = "DESI (DR1)"
N_PARAMS = 4
REFERENCE = "DESI DR1 MWS VAC; Koposov et al. 2024"
URL_BASE = "https://data.desi.lbl.gov/public/dr1/vac/dr1/mws/iron/v1.0/rv_output/240520"
# Stacking order matters only for row ordering; it is kept as in the legacy
# analysis so that row order is comparable.
FNAMES = ("rvpix-main-backup.fits", "rvpix-main-dark.fits", "rvpix-main-bright.fits")
SOURCES = tuple(f"{URL_BASE}/{fn}" for fn in FNAMES)

ID_COL = "gdr3_source_id"
PARAM_COLS = ("TEFF", "FEH", "LOGG", "ALPHAFE")
UNC_COLS = ("TEFF_ERR", "FEH_ERR", "LOGG_ERR", "ALPHAFE_ERR")
EXTRA_COLS = ("TARGET_RA", "TARGET_DEC")

#: The logg regularization error was raised from ``0.1 + edge_penalty`` to
#: ``0.2 + 2 * edge_penalty`` after the legacy table was made, so the parity check
#: is allowed to fail on ``x_err[:, 2]`` only.
PARITY_EXEMPT = (
    "x_err[:, 2] is 0.2 + 2*edge_penalty; the legacy table used 0.1 + edge_penalty"
)
#: Columns of the table that differ from the legacy one, and which parameters.
PARITY_DIFFERS = {"x_err": (2,)}

# Points defining the cool-dwarf boundary in the (Teff, logg) plane.
TEFF_EDGE = (6000.0, 5.0)
TEFF_EDGE_2 = (5000.0, 3.0)


def download(ctx) -> None:
    for fn in FNAMES:
        download_file(f"{URL_BASE}/{fn}", ctx.raw_dir / fn)


def load(ctx) -> Table:
    tables = []
    for fn in FNAMES:
        with fits.open(ctx.raw_dir / fn, mode="readonly") as f:
            d = Table(f["RVTAB"].data)
            d["gdr3_source_id"] = f["GAIA"].data["SOURCE_ID"][:]
        tables.append(d)
    return astropy.table.vstack(tables)


def total_snr(t: Table) -> np.ndarray:
    return np.sqrt(
        unmask(t["SN_B"]) ** 2 + unmask(t["SN_R"]) ** 2 + unmask(t["SN_Z"]) ** 2
    )


def teff_edge(logg):
    """Teff of the cool-dwarf boundary at a given logg."""
    (t0, g0), (t1, g1) = TEFF_EDGE, TEFF_EDGE_2
    return t0 + (np.asarray(logg, dtype="f8") - g0) * (t1 - t0) / (g1 - g0)


def _distance_to_edge(t: Table) -> np.ndarray:
    """How far a star sits beyond the cool-dwarf edge, clipped to [0, 1000] K."""
    return np.clip(teff_edge(unmask(t["LOGG"])) - unmask(t["TEFF"]), 0.0, 1000.0)


def cuts(t: Table, ctx) -> dict:
    snr = total_snr(t)
    chisq = unmask(t["CHISQ_TOT"])

    # SNR-dependent chi^2 threshold; cached on the context for diagnostics.
    chisq_of_snr = snr_chisq_relation(snr, chisq)
    ctx.extra["chisq_of_snr"] = chisq_of_snr

    logg = unmask(t["LOGG"])
    alpha = unmask(t["ALPHAFE"])
    spectype = np.asarray(t["RR_SPECTYPE"])

    return {
        "RVS_WARN clear": unmask(t["RVS_WARN"]) == 0,
        "spectral type STAR": spectype == "STAR",
        "SNR > 20": snr > 20.0,
        "logg in range": (logg < 6.0) & (logg > -0.2),
        "alpha in range": (alpha > -0.35) & (alpha < 1.15),
        "chi2 below SNR-dependent limit": chisq < 2 * chisq_of_snr(2 * snr),
        "uncertainties below limits": (
            (unmask(t["TEFF_ERR"]) < 500.0)
            & (unmask(t["LOGG_ERR"]) < 0.5)
            & (unmask(t["FEH_ERR"]) < 0.5)
            & (unmask(t["ALPHAFE_ERR"]) < 0.5)
        ),
    }


def param_errs(t: Table, ctx) -> np.ndarray:
    teff = unmask(t["TEFF"]).astype("f8")

    hot_penalty = np.clip(0.1 * (teff - 6000.0) / 1000.0, 0.0, np.inf)
    edge_penalty = 0.0004 * _distance_to_edge(t)

    teff_err = 0.1 + hot_penalty + edge_penalty
    n = len(t)
    return np.stack(
        [
            teff_err,
            np.full(n, 0.1),
            2*edge_penalty + 0.2,
            np.full(n, 0.4),
        ],
        axis=1,
    ).astype("f4")


def diagnostics(t: Table, mask, table: Table, ctx):
    """SNR vs chi^2 with the fitted quality threshold."""
    import matplotlib.pyplot as plt

    snr = total_snr(t)
    chisq = unmask(t["CHISQ_TOT"])
    f = ctx.extra.get("chisq_of_snr")
    keep = np.asarray(mask, dtype=bool)

    fig, ax = plt.subplots(figsize=(5, 4), layout="constrained")
    ax.hist2d(
        np.log10(snr[keep & np.isfinite(snr)]),
        np.log10(chisq[keep & np.isfinite(chisq) & (chisq > 0)]),
        bins=100,
        norm=plt.matplotlib.colors.LogNorm(),
        rasterized=True,
    )
    if f is not None:
        snr_plot = np.logspace(0.2, 3.0, 128)
        ax.plot(np.log10(snr_plot), np.log10(2 * f(2 * snr_plot)), c="w", lw=1, alpha=0.8)
    ax.set_xlabel(r"$\log_{10}\,\mathrm{SNR}$")
    ax.set_ylabel(r"$\log_{10}\,\chi^2$")
    ax.set_title("accepted stars and fitted threshold")

    path = plots.save(fig, ctx.diagnostic_dir, "snr_chisq")

    fig, ax = plt.subplots(figsize=(5, 4), layout="constrained")
    sc = ax.scatter(
        unmask(t["TEFF"])[keep][::10],
        unmask(t["LOGG"])[keep][::10],
        c=_distance_to_edge(t)[keep][::10],
        s=2,
        edgecolors="none",
        alpha=0.3,
        rasterized=True,
    )
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xlabel(r"$T_{\mathrm{eff}}$")
    ax.set_ylabel(r"$\log g$")
    fig.colorbar(sc, ax=ax, label=r"distance beyond cool edge (K)")
    return [path, plots.save(fig, ctx.diagnostic_dir, "teff_logg_edge_penalty")]
