"""DESI DR1 stellar parameters and abundances from a neural-network SED fit.

References
----------
Nagaraj, V., Hayden, M., et al. 2026, "Stellar parameters and abundances for
3.2 million DESI DR1 spectra using machine learning", arXiv:2610.04294
(submitted to ApJS).
Partial release (170,111 stars) on Zenodo: https://doi.org/10.5281/zenodo.21816591

Notes
-----
* The catalog is a Payne-style neural-network fit to the flux-calibrated,
  *un-normalised* DESI spectra, so it is independent of the official DESI
  pipelines (:mod:`~astrotune.catalogs.desi_dr1`, :mod:`~astrotune.catalogs.desi_ddp`)
  although it uses the same spectra, i.e. many of the same stars.
* **Only a partial release (~5% of the 3.2 million stars) is available at the
  time of writing.** Overlaps with other surveys will grow about twentyfold
  with the full release; the cuts below were tuned on the partial release.
* The file has Gaia DR3 source ids (``GAIA_SOURCE_ID``; -1 where unknown, which
  the pipeline drops) so no crossmatch is needed.
* There is no [alpha/Fe] column. It is built as the inverse-variance weighted
  mean of [Mg/Fe], [Si/Fe], [Ca/Fe] and [Ti/Fe] (the paper's validation
  against APOGEE/GALAH: offsets of about -0.07, +0.03...+0.06, -0.05...-0.07 and
  +0.01...+0.11 dex respectively).
* ``FEH`` is the calibrated [Fe/H]; ``FEH_RAW`` has a ~0.3 dex offset that
  depends on Teff and logg and is only used for the grid-edge cut.
* ~47% of the stars have ``HAS_GAIA_GUESS = False``: they lack a Gaia XP
  spectrum and were fit with a slower "global" grid search instead of the
  standard fit. The paper reports comparable quality, so they are kept, but
  they are a different population (fainter, more dwarfs, higher [alpha/Fe]);
  set :data:`REQUIRE_GAIA_GUESS` to drop them.

Cuts (see the module's :func:`cuts`; all chosen by looking at the Kiel and
[Fe/H]-[alpha/Fe] diagrams, and checked against the stars shared with APOGEE,
GALAH and SDSS-V ASPCAP):

* SNR_MED >= 20 and CHI2_FLAG <= 3 (the paper's recommendation).
* Finite uncertainties (3,140 fits have NaN errors, all from the global fit).
  These turn out to be fits that never left the starting node of the global
  grid search: their micro-turbulence is exactly 2.595 and their Teff, logg and
  [Fe/H] are the grid-node values (e.g. logg = 2.505, 3.752, 5.0), which show up
  as narrow spikes in the parameter histograms. A few more such stars are
  removed explicitly, and loose upper limits are put on the uncertainties.
* Stars piled up against the edges of the neural network's training grid are
  removed by cutting slightly inside the edges: Teff in (4050, 7950) K (grid
  4000-8000 K; 14% of the stars sit at 4010 K or below), logg in (0.1, 4.97)
  (grid 0-5), raw [Fe/H] in (-2.45, 0.95) (grid -2.5...1.0). The micro-turbulence
  also has a floor (0.75 km/s) that many stars sit on, but those stars follow
  the main sequence and the [Fe/H]-[alpha/Fe] relation normally, so no cut is
  applied.
"""

from __future__ import annotations

import numpy as np
from astropy.table import Table

from ..prep import plots
from ..prep.abundances import calc_alpha_Fe_from_abundances
from ..prep.download import download as download_file
from ..prep.transforms import unmask

KEY = "desi_nagaraj2026"
LABEL = "DESI (Nagaraj)"
N_PARAMS = 4
REFERENCE = "Nagaraj et al. 2026, arXiv:2610.04294"
SOURCES = ("https://zenodo.org/records/21816591/files/DESI_DR1_catalog.fits?download=1",)
FILENAME = "desi_dr1_nagaraj2026.fits"

ID_COL = "GAIA_SOURCE_ID"
PARAM_COLS = ("TEFF", "FEH", "LOGG", "ALPHA_FE")
UNC_COLS = ("TEFF_ERR", "FEH_ERR", "LOGG_ERR", "ALPHA_FE_ERR")
EXTRA_COLS = ("RA", "DEC")

#: Regularization errors: how far the transformation may move each parameter.
#: A first guess, in transformed units (cf. the offsets of ~0.05-0.07 found
#: against APOGEE); tune once the catalog is in the model.
X_ERR = (0.1, 0.1, 0.1, 0.2)

SNR_MIN = 20.0
CHI2_FLAG_MAX = 3.0  # recommended by the paper

#: Cuts just inside the edges of the neural network's training grid
TEFF_RANGE = (4050.0, 7950.0)  # grid: 4000-8000 K
LOGG_RANGE = (0.1, 4.97)  # grid: 0-5
FEH_RAW_RANGE = (-2.45, 0.95)  # grid: -2.5-1.0 (uncalibrated [Fe/H])

ALPHA_RANGE = (-0.5, 1.1)
MAX_ERR = {"TEFF_ERR": 150.0, "LOGG_ERR": 0.4, "FEH_ERR": 0.2, "ALPHA_FE_ERR": 0.2}

#: Drop the stars fit without a Gaia XP initial guess (the "global" fit)
REQUIRE_GAIA_GUESS = False

#: Micro-turbulence is not optimised away from its starting value (the centre of
#: its range) when the "global" fit gets stuck at a node of the starting grid.
#: Those fits have NaN uncertainties (removed by the finite-uncertainty cut);
#: this removes the few that do not.
VMIC_STUCK = 2.595


def download(ctx) -> None:
    download_file(SOURCES[0], ctx.raw_dir / FILENAME)


def load(ctx) -> Table:
    return Table.read(ctx.raw_dir / FILENAME)


def prepare_columns(t: Table, ctx) -> Table:
    """Add the inverse-variance weighted mean [alpha/Fe] (Mg, Si, Ca, Ti)."""
    alpha_fe, alpha_fe_err, abundances = calc_alpha_Fe_from_abundances(
        t, "{x.upper()}_FE", "{x.upper()}_FE_ERR"
    )
    t["ALPHA_FE"] = alpha_fe
    t["ALPHA_FE_ERR"] = alpha_fe_err
    ctx.extra["abundances"] = abundances
    return t


def cuts(t: Table, ctx) -> dict:
    teff = unmask(t["TEFF"]).astype("f8")
    logg = unmask(t["LOGG"])
    feh_raw = unmask(t["FEH_RAW"])
    alpha = unmask(t["ALPHA_FE"])

    errs = {k: unmask(t[k]).astype("f8") for k in MAX_ERR}
    finite = np.isfinite(alpha) & np.isfinite(unmask(t["ALPHA_FE_ERR"]))
    for k, v in errs.items():
        finite &= np.isfinite(v) & (v > 0)  # TEFF_ERR is -9999 where the fit failed

    named = {
        "Gaia source id": np.asarray(t[ID_COL]) > 0,
        f"SNR >= {SNR_MIN:g}": unmask(t["SNR_MED"]) >= SNR_MIN,
        f"CHI2_FLAG <= {CHI2_FLAG_MAX:g}": unmask(t["CHI2_FLAG"]) <= CHI2_FLAG_MAX,
        "finite uncertainties": finite,
        "fit not stuck at its starting point": np.abs(unmask(t["VMIC"]) - VMIC_STUCK) > 2e-3,
        "Teff inside grid edges": (teff > TEFF_RANGE[0]) & (teff < TEFF_RANGE[1]),
        "logg inside grid edges": (logg > LOGG_RANGE[0]) & (logg < LOGG_RANGE[1]),
        "raw [Fe/H] inside grid edges": (feh_raw > FEH_RAW_RANGE[0]) & (feh_raw < FEH_RAW_RANGE[1]),
        "alpha in range": (alpha > ALPHA_RANGE[0]) & (alpha < ALPHA_RANGE[1]),
        "uncertainties below limits": np.all(
            [(errs[k] < limit) for k, limit in MAX_ERR.items()], axis=0
        ),
    }
    if REQUIRE_GAIA_GUESS:
        named["has Gaia XP initial guess"] = np.asarray(t["HAS_GAIA_GUESS"], dtype=bool)
    return named


def param_errs(t: Table, ctx) -> tuple:
    return X_ERR


def diagnostics(t: Table, mask, table: Table, ctx):
    """Grid-edge histograms with the cuts, SNR vs CHI2_FLAG, alpha elements."""
    import matplotlib.pyplot as plt

    keep = np.asarray(mask, dtype=bool)
    written = []

    # Distribution of each quantity with a cut just inside a grid edge
    panels = (
        ("TEFF", "Teff (K)", TEFF_RANGE, np.arange(4000, 8001, 25.0)),
        ("LOGG", "logg", LOGG_RANGE, np.linspace(0, 5, 201)),
        ("FEH_RAW", "raw [Fe/H]", FEH_RAW_RANGE, np.linspace(-2.5, 1.0, 141)),
        ("VMIC", "micro-turbulence (km/s; not cut)", (), np.linspace(0.75, 3.5, 111)),
    )
    fig, axes = plt.subplots(1, 4, figsize=(14, 3), layout="constrained")
    base = np.asarray(t[ID_COL]) > 0
    for ax, (col, label, limits, bins) in zip(axes, panels):
        x = unmask(t[col]).astype("f8")
        ax.hist(x[base], bins=bins, histtype="step", label="all with Gaia id")
        ax.hist(x[keep], bins=bins, histtype="stepfilled", alpha=0.5, label="selected")
        for v in limits:
            ax.axvline(v, c="k", ls="--", lw=0.8)
        ax.set_yscale("log")
        ax.set_xlabel(label)
    axes[0].legend(fontsize=7)
    written.append(plots.save(fig, ctx.diagnostic_dir, "grid_edges"))

    fig, ax = plt.subplots(figsize=(5, 4), layout="constrained")
    snr = unmask(t["SNR_MED"])
    cf = unmask(t["CHI2_FLAG"])
    ax.hist2d(
        np.log10(snr[base]),
        np.clip(cf[base], -3, 12),
        bins=[np.linspace(1, 2.6, 100), np.linspace(-3, 12, 100)],
        norm=plt.matplotlib.colors.LogNorm(),
        rasterized=True,
    )
    ax.axhline(CHI2_FLAG_MAX, c="w", ls="--", lw=0.8)
    ax.axvline(np.log10(SNR_MIN), c="w", ls="--", lw=0.8)
    ax.set_xlabel(r"$\log_{10}\,\mathrm{SNR}$")
    ax.set_ylabel("CHI2_FLAG (clipped to 12)")
    written.append(plots.save(fig, ctx.diagnostic_dir, "snr_chi2_flag"))

    abundances = ctx.extra.get("abundances")
    if abundances is not None:
        alpha = unmask(t["ALPHA_FE"])
        good = keep & np.isfinite(alpha)
        written.append(
            plots.plot_alpha_abundances(alpha[good], abundances[good], outdir=ctx.diagnostic_dir)
        )
    return written
