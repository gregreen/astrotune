"""PRISTINE PIGS: very metal-poor stars from the Pristine Inner Galaxy Survey.

References
----------
Ardern-Arentsen et al. (2024), MNRAS 530, 3391
https://cdsarc.cds.unistra.fr/viz-bin/cat/J/MNRAS/530/3391

Notes
-----
The FERRE parameter estimates are used. The ``goodF`` quality flag (a
fixed-width ``'true '``/``'false'`` string) is compared with surrounding
whitespace stripped, so the exact padding does not matter.
"""

from __future__ import annotations

import numpy as np
from astropy.io import fits
from astropy.table import Table

from ..prep.download import download as download_file
from ..prep.transforms import unmask

KEY = "pristine_pigs"
LABEL = "PIGS"
N_PARAMS = 3
REFERENCE = "Ardern-Arentsen et al. 2024, MNRAS 530, 3391"
TABLE_URL = "https://cdsarc.cds.unistra.fr/viz-bin/nph-Cat/fits.gz?J/MNRAS/530/3391/tabled2.dat.gz"
README_URL = "https://cdsarc.cds.unistra.fr/ftp/J/MNRAS/530/3391/ReadMe"
SOURCES = (TABLE_URL, README_URL)
FILENAME = "Pristine_PIGS_tabled2.fits.gz"
README_FILE = "Pristine_PIGS_ReadMe.txt"
HDU = "tabled2.dat"

ID_COL = "GaiaDR3"
PARAM_COLS = ("TeffF", "[Fe/H]F", "loggF")
UNC_COLS = ("e_TeffFtot", "e_[Fe/H]Ftot", "e_loggFtot")
EXTRA_COLS = ("[C/Fe]F", "e_[C/Fe]Ftot", "RAdeg", "DEdeg")
ERROR_FLOORS = (0.01, 0.01, 0.04)

X_ERR = (0.1, 0.1, 0.1)

TEFF_RANGE = (4501.0, 6999.0)
LOGG_RANGE = (1.01, 4.99)


def download(ctx) -> None:
    download_file(TABLE_URL, ctx.raw_dir / FILENAME)
    download_file(README_URL, ctx.raw_dir / README_FILE)


def load(ctx) -> Table:
    with fits.open(ctx.raw_dir / FILENAME, mode="readonly") as f:
        return Table(f[HDU].data)


def cuts(t: Table, ctx) -> dict:
    teff = unmask(t["TeffF"], fill=np.nan)
    logg = unmask(t["loggF"], fill=np.nan)
    good = np.char.strip(np.asarray(t["goodF"]).astype(str)) == "true"

    return {
        "goodF flag set": good,
        "Teff in range": (teff > TEFF_RANGE[0]) & (teff < TEFF_RANGE[1]),
        "logg in range": (logg > LOGG_RANGE[0]) & (logg < LOGG_RANGE[1]),
    }


def param_errs(t: Table, ctx):
    return X_ERR
