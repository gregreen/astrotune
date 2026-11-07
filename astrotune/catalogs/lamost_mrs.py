"""LAMOST Medium Resolution Survey (Cycle-StarNet parameters).

References
----------
Wang et al. (2023), ApJS 266, 40

Notes
-----
The CDS distribution is a gzipped FITS table. A decompressed copy and an
astropy-HDF5 copy of the same table are kept in ``data/raw/lamost_mrs/``; they
were verified to be identical, and the FITS file is used here.

[alpha/Fe] is the inverse-variance weighted mean of the individual
alpha-element abundances.
"""

from __future__ import annotations

import numpy as np
from astropy.table import Table

from ..prep import plots
from ..prep.abundances import calc_alpha_Fe_from_abundances
from ..prep.download import download as download_file
from ..prep.transforms import unmask

KEY = "lamost_mrs"
LABEL = "LAMOST MRS"
N_PARAMS = 4
REFERENCE = "Wang et al. 2023, ApJS 266, 40"
SOURCES = ("https://cdsarc.cds.unistra.fr/viz-bin/nph-Cat/fits.gz?J/ApJS/266/40/catalog.dat.gz",)
FILENAME = "lamost_mrs_cyclestarnet.fits"

ID_COL = "Gaia_source_id"
PARAM_COLS = ("Teff", "FeH", "logg", "alphaFe")
UNC_COLS = ("Err_Teff", "Err_FeH", "Err_logg", "Err_alphaFe")
EXTRA_COLS = ("RA", "Dec")

#: Abundances with uncertainties above this value are unusable.
ALPHA_ERR_MAX = 10.0


def download(ctx) -> None:
    download_file(SOURCES[0], ctx.raw_dir / FILENAME)


def load(ctx) -> Table:
    return Table.read(ctx.raw_dir / FILENAME)


def prepare_columns(t: Table, ctx) -> Table:
    alpha_fe, alpha_fe_err, abundances = calc_alpha_Fe_from_abundances(
        t, "{x}Fe", "Err_{x}Fe", flag_fmt="Flag_{x}Fe"
    )
    t["alphaFe"] = alpha_fe
    t["Err_alphaFe"] = alpha_fe_err
    ctx.extra["abundances"] = abundances
    return t


def cuts(t: Table, ctx) -> dict:
    return {
        "Flag_Teff == 0": unmask(t["Flag_Teff"]) == 0,
        "Flag_logg == 0": unmask(t["Flag_logg"]) == 0,
        "Flag_FeH == 0": unmask(t["Flag_FeH"]) == 0,
        "alpha/Fe error usable": unmask(t["Err_alphaFe"], fill=np.nan) < ALPHA_ERR_MAX,
        "Flag_quality == 0": unmask(t["Flag_quality"]) == 0,
    }


def param_errs(t: Table, ctx) -> np.ndarray:
    teff = unmask(t["Teff"], fill=np.nan).astype("f8")

    hot = np.clip(0.1 + 0.4 * (teff - 6000.0) / 1000.0, 0.1, np.inf)
    hot = np.where(np.isfinite(hot), hot, 0.1)

    n = len(t)
    return np.stack(
        [hot, np.full(n, 0.1), np.full(n, 0.1), np.full(n, 0.1)], axis=1
    ).astype("f4")


def diagnostics(t: Table, mask, table: Table, ctx):
    abundances = ctx.extra.get("abundances")
    if abundances is None:
        return []
    keep = np.asarray(mask, dtype=bool)
    alpha_fe = unmask(t["alphaFe"], fill=np.nan)
    good = keep & np.isfinite(alpha_fe)
    return [
        plots.plot_alpha_abundances(
            alpha_fe[good], abundances[good], outdir=ctx.diagnostic_dir
        )
    ]
