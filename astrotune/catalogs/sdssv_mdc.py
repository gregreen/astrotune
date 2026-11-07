"""SDSS-V Cannon parameters for M dwarfs (elemental abundances VAC).

References
----------
Behmard et al. (2025), ApJ 982, 13

Notes
-----
This catalog provides Teff, [Fe/H] and alpha abundances but **no surface
gravity**. To keep the parameter vectors rectangular, ``logg`` is carried as a
dummy axis: the value is set to 0, its statistical uncertainty and its
regularization error to 1e9, so the axis contributes nothing to the loss.

The [alpha/H] abundances are converted to [alpha/Fe] by subtracting [Fe/H].
"""

from __future__ import annotations

import numpy as np
from astropy.io import fits
from astropy.table import Table

from ..prep import plots
from ..prep.abundances import calc_alpha_Fe_from_abundances
from ..prep.download import download as download_file
from ..prep.transforms import unmask

KEY = "sdssv_mdc"
LABEL = "SDSS-V (MDw Cannon)"
N_PARAMS = 4
REFERENCE = "Behmard et al. 2025, ApJ 982, 13"
SOURCES = (
    "https://data.sdss.org/sas/dr19/vac/mwm/m-dwarf/elemental_abundances/"
    "mdwarf_abundances-1.0.2.fits",
)
FILENAME = "mdwarf_abundances-1.0.2.fits"
HDU = 1

ID_COL = "gaia_dr3_source_id"
#: ``logg`` has no counterpart in this catalog and is filled with a dummy axis.
PARAM_COLS = ("Teff_cannon", "fe_h_cannon", "fe_h_cannon", "alpha_fe")
UNC_COLS = ("Teff_err", "fe_h_err", "fe_h_err", "alpha_fe_err")
EXTRA_COLS = ()

#: The dummy logg axis is given an enormous regularization error.
X_ERR = (0.2, 0.2, 1e9, 0.2)

#: Index of the dummy logg axis.
LOGG_INDEX = 2
DUMMY_SIGMA = 1e9

#: This catalog does not measure surface gravity, so a Kiel diagram is not
#: meaningful; the pipeline draws Teff vs [Fe/H] instead.
HAS_LOGG = False


def download(ctx) -> None:
    download_file(SOURCES[0], ctx.raw_dir / FILENAME)


def load(ctx) -> Table:
    with fits.open(ctx.raw_dir / FILENAME, mode="readonly") as f:
        return Table(f[HDU].data[:])


def prepare_columns(t: Table, ctx) -> Table:
    """Convert the mean [alpha/H] into [alpha/Fe]."""
    alpha_h, alpha_h_err, abundances = calc_alpha_Fe_from_abundances(
        t, "{x.lower()}_h_cannon", "{x.lower()}_h_err"
    )
    fe_h = unmask(t["fe_h_cannon"], fill=np.nan)
    fe_h_err = unmask(t["fe_h_err"], fill=np.nan)

    t["alpha_fe"] = alpha_h - fe_h
    t["alpha_fe_err"] = np.sqrt(alpha_h_err**2 + fe_h_err**2)
    ctx.extra["abundances"] = abundances
    return t


def cuts(t: Table, ctx) -> dict:
    return {
        "[Fe/H] error < 0.3": unmask(t["fe_h_err"]) < 0.3,
        "Teff agrees with the photometric value": unmask(t["temp_agree"]).astype(bool),
    }


def param_errs(t: Table, ctx):
    return X_ERR


def finalize(table: Table, ctx) -> Table:
    """Zero out the dummy logg axis so it cannot influence the loss."""
    table["x"][:, LOGG_INDEX] = 0.0
    table["sigma_x"][:, LOGG_INDEX] = DUMMY_SIGMA
    return table


def diagnostics(t: Table, mask, table: Table, ctx):
    abundances = ctx.extra.get("abundances")
    if abundances is None:
        return []
    keep = np.asarray(mask, dtype=bool)
    alpha_fe = unmask(t["alpha_fe"], fill=np.nan)
    good = keep & np.isfinite(alpha_fe)
    return [
        plots.plot_alpha_abundances(
            alpha_fe[good], abundances[good], outdir=ctx.diagnostic_dir
        )
    ]
