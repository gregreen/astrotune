"""LAMOST M stars from Ding et al. (2022).

References
----------
Ding et al. (2022), ApJS 260, 45

Notes
-----
The catalog publishes no uncertainties, so typical values quoted in the paper
are adopted (45 K, 0.25 dex and 0.22 dex).

The published parameters are offset relative to APOGEE DR17; the corrections
quoted in the paper are applied, separately for dwarfs and giants, *before* the
quality cuts.

No error floors are applied to this catalog, matching the original analysis.
"""

from __future__ import annotations

import numpy as np
from astropy.table import Table

from ..prep.download import download as download_file
from ..prep.transforms import unmask

KEY = "lamost_mstar_ding2022"
LABEL = "LAMOST M-Stars (Ding)"
N_PARAMS = 3
REFERENCE = "Ding et al. 2022, ApJS 260, 45"
TABLE_URL = "https://cdsarc.cds.unistra.fr/ftp/J/ApJS/260/45/tableb1.dat.gz"
README_URL = "https://cdsarc.cds.unistra.fr/ftp/J/ApJS/260/45/ReadMe"
SOURCES = (TABLE_URL, README_URL)
FILENAME = "tableb1.dat"
README_FILE = "readme.txt"

ID_COL = "Gaia"
PARAM_COLS = ("Teff", "[Fe/H]", "log(g)")
UNC_COLS = ("e_Teff", "e_[Fe/H]", "e_log(g)")
EXTRA_COLS = ()

#: Typical uncertainties quoted by the paper.
DEFAULT_ERRORS = {"e_Teff": 45.0, "e_log(g)": 0.25, "e_[Fe/H]": 0.22}
#: Offsets relative to APOGEE DR17, split at log(g) = 3.
DWARF_LOGG_MIN = 3.0
DWARF_OFFSETS = {"Teff": 176.0, "log(g)": -0.16, "[Fe/H]": 0.14}
GIANT_OFFSETS = {"Teff": 140.0, "log(g)": -0.30, "[Fe/H]": -0.18}

#: No error floors in the original analysis.
ERROR_FLOORS = (0.0, 0.0, 0.0)

X_ERR = (0.2, 0.2, 0.4)


def download(ctx) -> None:
    download_file(TABLE_URL, ctx.raw_dir / FILENAME)
    download_file(README_URL, ctx.raw_dir / README_FILE)


def load(ctx) -> Table:
    return Table.read(
        ctx.raw_dir / FILENAME, format="cds", readme=str(ctx.raw_dir / README_FILE)
    )


def prepare_columns(t: Table, ctx) -> Table:
    n = len(t)
    for name, value in DEFAULT_ERRORS.items():
        t[name] = np.full(n, value)

    # The CDS table stores the parameters as integers, so promote them before
    # applying the (fractional) offsets.
    for name in ("Teff", "log(g)", "[Fe/H]"):
        t[name] = t[name].astype("f8")

    logg = unmask(t["log(g)"], fill=np.nan)
    for mask, offsets in (
        (logg > DWARF_LOGG_MIN, DWARF_OFFSETS),
        (logg <= DWARF_LOGG_MIN, GIANT_OFFSETS),
    ):
        for name, offset in offsets.items():
            t[name][mask] += offset
    return t


def cuts(t: Table, ctx) -> dict:
    teff = unmask(t["Teff"], fill=np.nan)
    logg = unmask(t["log(g)"], fill=np.nan)

    return {
        "Teff in range": (teff > 2900.0) & (teff < 4200.0),
        "SNR > 20": unmask(t["SNR"]) > 20.0,
        "logg outside 2-4": (logg < 2.0) | (logg > 4.0),
        "logg < 5.2": logg < 5.2,
    }


def param_errs(t: Table, ctx):
    return X_ERR
