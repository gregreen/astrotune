"""LAMOST M dwarfs with Cycle-StarNet parameters (Zhang et al. 2025).

References
----------
Zhang et al. (2025), ApJS 277, 47
Zenodo: https://doi.org/10.5281/zenodo.14030249

Notes
-----
The catalog is distributed as a CSV with no Gaia identifiers; a DR3 crossmatch
is generated once and cached in ``data/interim/lamost_csn_dr10/``.

The metallicity column is ``[M/H]``, not ``[Fe/H]``.
"""

from __future__ import annotations

import astropy.units as units
import numpy as np
from astropy.table import Table

from ..prep.download import download as download_file
from ..prep.gaia import apply_crossmatch, gaia_crossmatch_batched
from ..prep.transforms import unmask

KEY = "lamost_csn_dr10"
LABEL = "LAMOST M-Dwarfs (Zhang)"
N_PARAMS = 3
REFERENCE = "Zhang et al. 2025, ApJS 277, 47"
FILENAME = "DR10_Cycle-SN_M_dwarfs.csv"
SOURCES = (f"https://zenodo.org/records/14030249/files/{FILENAME}?download=1",)

CROSSMATCH_FILE = "crossmatched.h5"
SEARCH_SEPARATION = 1.0 * units.arcsec
MAX_SEPARATION = 0.5 * units.arcsec

ID_COL = "gdr3_source_id"
PARAM_COLS = ("Teff", "[M/H]", "Logg")
UNC_COLS = ("e_Teff", "e_[M/H]", "e_Logg")
EXTRA_COLS = ("ra_gdr3", "dec_gdr3")

X_ERR = (0.1, 0.1, 0.1)

LOGG_RANGE = (4.4, 5.2)


def download(ctx) -> None:
    download_file(SOURCES[0], ctx.raw_dir / FILENAME)


def load(ctx) -> Table:
    cache = ctx.interim_dir / CROSSMATCH_FILE
    if cache.exists() and not ctx.force:
        return Table.read(cache)

    catalog = Table.read(ctx.raw_dir / FILENAME)
    crossmatch = gaia_crossmatch_batched(
        catalog["RA"],
        catalog["Dec"],
        max_batch_size=1024 * 64,
        max_separation=SEARCH_SEPARATION,
        verbose=True,
    )
    matched = apply_crossmatch(catalog, crossmatch, max_separation=MAX_SEPARATION)

    cache.parent.mkdir(parents=True, exist_ok=True)
    matched.write(
        cache, serialize_metadata=True, overwrite=True,
        compression="gzip", compression_opts=3, chunks=True,
    )
    return matched


def cuts(t: Table, ctx) -> dict:
    logg = unmask(t["Logg"], fill=np.nan)
    return {
        "SNR(r) > 15": unmask(t["SNRr"]) > 15.0,
        "SNR(i) > 20": unmask(t["SNRi"]) > 20.0,
        "logg in range": (logg > LOGG_RANGE[0]) & (logg < LOGG_RANGE[1]),
    }


def param_errs(t: Table, ctx):
    return X_ERR
