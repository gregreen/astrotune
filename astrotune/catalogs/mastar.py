"""MaStar stellar parameters (SDSS DR17, VAC v2).

References
----------
Yan et al. (in preparation): the MaStar stellar-parameter value-added catalog
Hill et al. (2022), MNRAS 509, 4308 (LH method)
Imig et al. (2022), AJ 163, 56 (JI method)
Lazarz et al. (in press) and Chen et al. (in preparation): the DL and YC methods
of v2
https://www.sdss4.org/dr17/mastar/mastar-stellar-parameters/

Notes
-----
Four independent fits were made to every MaStar spectrum (labelled JI, DL, LH,
YC), and the master set is the *median over the methods that are valid for that
star*: ``TEFF_MED``, ``LOGG_MED``, ``FEH_MED``, ``ALPHA_MED`` with a ``_MED_ERR``
each. ``NGROUPS`` is how many methods contributed to the Teff/log g/[Fe/H]
median and ``NALPHAGROUPS`` to the [alpha/Fe] one, which is filtered separately
for alpha and is usually the smaller of the two -- it is the number to look at
when judging whether an alpha value means anything.

[Fe/H] comes from ``FEH_CAL_MED``: three of the four methods' [Fe/H] were
calibrated against ASPCAP on the overlap sample and the SDSS page recommends the
calibrated median (the per-method columns carry only the uncalibrated values, and
``FEH_MED`` is the uncalibrated median). [alpha/Fe] has no calibrated version, so
``ALPHA_MED`` is what is used.

Missing values are the sentinel ``-999`` in *every* column, errors included, so
they are turned into NaN as soon as the table is read.

The catalog has no Gaia identifiers, only positions (decimal degrees, from the
MaStar targeting), so it is crossmatched against Gaia DR3 once and cached in
``data/interim/mastar/``. The Gaia Archive was down when this was added; the
``--gaia-archive gavo`` mirror was used.

Cuts (chosen from the Kiel and [Fe/H]-[alpha/Fe] diagrams and from the pile-ups
at the edges of the fitting grid, see ``grid_edges.png``):

* at least two methods in both medians: a single method's value has had no
  cross-check, and NGROUPS == 0 never has valid parameters anyway;
* every parameter finite and inside a loose range;
* Teff > 3500 K and < 30,000 K: 110 stars sit exactly on 3500 K and a further
  458 below it, 12 sit exactly on 30,000 K and 7 above it. Both ends are the
  edges of the fitting grid, where the fit runs out of grid and the value is
  pinned (the same signature shows up on 157 stars at exactly log g = 5.000).
  The bounds are deliberately loose: the hot tail is real for our purposes --
  LAMOST HotPayne already supplies the model with plenty of stars above 8000 K,
  and the hot rows that survive the cuts below number only a few hundred;
* log g < 5.0: 157 stars sit exactly at 5.000, the top of the grid;
* loose upper limits on the uncertainties (Teff < 500 K, log g < 0.5,
  [Fe/H] < 0.5, [alpha/Fe] < 0.4): the tails reach 23,480 K and 11.6 dex, which
  are failed fits rather than measurements.
"""

from __future__ import annotations

import astropy.units as units
import numpy as np
from astropy.io import fits
from astropy.table import Table

from ..prep import plots
from ..prep.download import download as download_file
from ..prep.gaia import crossmatch_catalog
from ..prep.transforms import unmask

KEY = "mastar"
LABEL = "MaStar"
N_PARAMS = 4
REFERENCE = "Yan et al. (in prep.); Hill et al. 2022, MNRAS 509, 4308; Imig et al. 2022, AJ 163, 56"
SOURCES = (
    "https://data.sdss.org/sas/dr17/manga/spectro/mastar/v3_1_1/v1_7_7/vac/"
    "parameters/v2/mastar-goodstars-v3_1_1-v1_7_7-params-v2.fits",
)
FILENAME = "mastar-goodstars-v3_1_1-v1_7_7-params-v2.fits"
HDU = 1

CROSSMATCH_FILE = "crossmatched.h5"
#: The positions are from the MaStar targeting, so they are good to ~0.2".
MAX_SEPARATION = 0.5 * units.arcsec

ID_COL = "gdr3_source_id"
#: Canonical order: Teff, [Fe/H], log g, [alpha/Fe]
PARAM_COLS = ("TEFF_MED", "FEH_CAL_MED", "LOGG_MED", "ALPHA_MED")
UNC_COLS = ("TEFF_MED_ERR", "FEH_CAL_MED_ERR", "LOGG_MED_ERR", "ALPHA_MED_ERR")
EXTRA_COLS = ("ra_gdr3", "dec_gdr3")

#: Everything missing in this catalog is this sentinel, errors included.
SENTINEL = -999.0

#: Methods that must contribute to the median of Teff/log g/[Fe/H] and of
#: [alpha/Fe] (``NGROUPS``, ``NALPHAGROUPS``).
MIN_GROUPS, MIN_ALPHA_GROUPS = 2, 2

# The T_eff cut to avoid grid edges, where stars pile up
TEFF_RANGE = (3_500.0, 30_000.0)
LOGG_RANGE = (-0.5, 5.0)
FEH_RANGE = (-3.5, 0.7)
ALPHA_RANGE = (-0.5, 0.8)

#: Loose upper limits on the catalog's own uncertainties (Teff in K)
MAX_ERR = {"TEFF_MED_ERR": 500.0, "LOGG_MED_ERR": 0.5,
           "FEH_CAL_MED_ERR": 0.5, "ALPHA_MED_ERR": 0.4}

#: Regularization errors. Nothing is known about this catalog's systematics yet
#: (it is a library of stars, not a survey with an established scale), so the
#: usual 0.1 everywhere.
X_ERR = (0.1, 0.1, 0.1, 0.1)


def download(ctx) -> None:
    download_file(SOURCES[0], ctx.raw_dir / FILENAME)


def load(ctx) -> Table:
    def read_catalog() -> Table:
        with fits.open(ctx.raw_dir / FILENAME, mode="readonly") as f:
            t = Table(f[HDU].data)
        # Sentinels first: the crossmatch only needs RA/DEC, but leaving -999 in
        # the parameter columns would poison everything downstream.
        for name in set(PARAM_COLS) | set(UNC_COLS) | {"RA", "DEC"}:
            values = np.asarray(t[name], dtype="f8")
            t[name] = np.where(values <= SENTINEL, np.nan, values)
        return t

    return crossmatch_catalog(
        read_catalog,
        ra_col="RA",
        dec_col="DEC",
        cache_path=ctx.interim_dir / CROSSMATCH_FILE,
        max_separation=MAX_SEPARATION,
        archive=ctx.gaia_archive,
        force=ctx.force,
        allow_regression=ctx.allow_crossmatch_regression,
        key=KEY,
        verbose=True,
        serialize_metadata=True,
        compression="gzip",
        compression_opts=3,
        chunks=True,
    )


def cuts(t: Table, ctx) -> dict:
    teff = unmask(t["TEFF_MED"]).astype("f8")
    logg = unmask(t["LOGG_MED"]).astype("f8")
    feh = unmask(t["FEH_CAL_MED"]).astype("f8")
    alpha = unmask(t["ALPHA_MED"]).astype("f8")
    err = {k: unmask(t[k]).astype("f8") for k in MAX_ERR}

    def inside(values, limits):
        return (values > limits[0]) & (values < limits[1])

    def small(k):
        return np.isfinite(err[k]) & (err[k] > 0) & (err[k] < MAX_ERR[k])

    return {
        "Gaia crossmatch": np.asarray(t[ID_COL]) > 0,
        f"NGROUPS >= {MIN_GROUPS}": np.asarray(t["NGROUPS"]) >= MIN_GROUPS,
        f"NALPHAGROUPS >= {MIN_ALPHA_GROUPS}": (
            np.asarray(t["NALPHAGROUPS"]) >= MIN_ALPHA_GROUPS
        ),
        "all four parameters finite": (
            np.isfinite(teff) & np.isfinite(logg) & np.isfinite(feh) & np.isfinite(alpha)
        ),
        "Teff inside the grid": inside(teff, TEFF_RANGE),
        "log g inside the grid": inside(logg, LOGG_RANGE),
        "[Fe/H] in range": inside(feh, FEH_RANGE),
        "[alpha/Fe] in range": inside(alpha, ALPHA_RANGE),
        "uncertainties finite and below limits": (
            small("TEFF_MED_ERR") & small("LOGG_MED_ERR")
            & small("FEH_CAL_MED_ERR") & small("ALPHA_MED_ERR")
        ),
    }


def param_errs(t: Table, ctx):
    """Constant regularization errors; a tuple is broadcast by extract_table."""
    return X_ERR


def diagnostics(t: Table, mask, table: Table, ctx):
    """Where the grid edges and the method counts bite, and what the sample is."""
    import matplotlib.pyplot as plt

    keep = np.asarray(mask, dtype=bool)
    teff = unmask(t["TEFF_MED"]).astype("f8")
    logg = unmask(t["LOGG_MED"]).astype("f8")
    feh = unmask(t["FEH_CAL_MED"]).astype("f8")
    alpha = unmask(t["ALPHA_MED"]).astype("f8")
    ngroups = np.asarray(t["NGROUPS"])
    nalphagroups = np.asarray(t["NALPHAGROUPS"])
    written = []

    # Pile-ups at exact values: the signature of a fit that hit the grid edge.
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4), layout="constrained")
    for ax, (values, name, limits) in zip(
        axes,
        ((teff, "Teff (K)", TEFF_RANGE), (logg, "log g", LOGG_RANGE),
         (alpha, r"[$\alpha$/Fe]", ALPHA_RANGE)),
    ):
        lo, hi = np.nanpercentile(values[keep], [0.5, 99.5]) if keep.any() else (0, 1)
        ax.hist(values[keep], bins=np.linspace(lo, hi, 120), histtype="step", label="kept")
        ax.hist(values[~keep & np.isfinite(values)], bins=np.linspace(lo, hi, 120),
                histtype="step", label="cut", color="C1")
        for limit in limits:
            ax.axvline(limit, c="k", ls="--", lw=0.8)
        ax.set_yscale("log")
        ax.set_xlabel(name)
        ax.legend(fontsize=7)
    # the exact-value spikes, counted
    for j, values in enumerate((teff, logg, alpha)):
        unique, counts = np.unique(values[np.isfinite(values)], return_counts=True)
        worst = np.argsort(-counts)[:3]
        text = ", ".join(f"{unique[i]:g}x{counts[i]}" for i in worst)
        axes[j].set_title(f"top repeats: {text}", fontsize=7)
    written.append(plots.save(fig, ctx.diagnostic_dir, "grid_edges"))

    # How many methods stand behind each star, and what the sample looks like.
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4), layout="constrained")
    bins = np.arange(-0.5, 5.5, 1.0)
    axes[0].hist(ngroups[keep], bins=bins, histtype="step", label="NGROUPS")
    axes[0].hist(nalphagroups[keep], bins=bins, histtype="step",
                 label="NALPHAGROUPS")
    axes[0].set_xlabel("methods in the median")
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=7)
    axes[1].hist(feh[keep], bins=100, histtype="step")
    axes[1].set_xlabel("[Fe/H] (calibrated median)")
    axes[1].set_yscale("log")
    sc = axes[2].scatter(feh[keep], alpha[keep], c=nalphagroups[keep], s=3,
                         cmap="viridis", vmin=1, vmax=4, rasterized=True)
    axes[2].set_xlabel("[Fe/H]")
    axes[2].set_ylabel(r"[$\alpha$/Fe]")
    fig.colorbar(sc, ax=axes[2], label="NALPHAGROUPS")
    written.append(plots.save(fig, ctx.diagnostic_dir, "sample"))
    return written
