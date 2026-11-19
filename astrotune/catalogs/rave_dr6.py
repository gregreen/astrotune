"""RAVE DR6: BDASP temperatures/gravities with GAUGUIN abundances.

References
----------
Steinmetz et al. (2020), AJ 160, 83
BDASP: https://doi.org/10.17876/rave/dr.6/008
GAUGUIN: https://doi.org/10.17876/rave/dr.6/010

Notes
-----
RAVE DR6 is distributed as four HDF5 tables keyed by ``rave_obs_id``; they are
merged on that key. The catalog has no Gaia identifiers, so a DR3 crossmatch
is generated once and cached in ``data/interim/rave_dr6/``.

The merged table carries an object-dtype ``cluster_flag`` column, which cannot
be written to HDF5; it is dropped before caching.
"""

from __future__ import annotations

import astropy.units as units
import numpy as np
import pandas as pd
from astropy.table import Table

from ..prep.download import download as download_file
from ..prep.gaia import crossmatch_catalog
from ..prep.transforms import unmask

KEY = "rave_dr6"
LABEL = "RAVE (DR6)"
N_PARAMS = 4
REFERENCE = "Steinmetz et al. 2020, AJ 160, 83 (BDASP + GAUGUIN)"
URL_BASE = "https://www.rave-survey.org/files/hdf5"
FNAMES = (
    "ravedr6_dr6_bdasp.h5",
    "ravedr6_dr6_gauguin_bdasp.h5",
    "ravedr6_dr6_x_gaiadr2.h5",
    "ravedr6_dr6_obsdata.h5",
)
SOURCES = tuple(f"{URL_BASE}/{fn}" for fn in FNAMES)

CROSSMATCH_FILE = "crossmatched.h5"
JOIN_KEY = "rave_obs_id"

#: Wide search radius, then a tighter cut on the nearest neighbor.
SEARCH_SEPARATION = 1.0 * units.arcsec
MAX_SEPARATION = 0.3 * units.arcsec

ID_COL = "gdr3_source_id"
PARAM_COLS = ("teff_bdasp", "fe_h_gauguin", "logg_bdasp", "alpha_fe_gauguin")
UNC_COLS = (
    "teff_error_bdasp",
    "fe_h_error_gauguin",
    "logg_error_bdasp",
    "alpha_fe_error_gauguin",
)
#: ``ra_gdr3`` only: the original analysis listed it twice by mistake.
EXTRA_COLS = ("ra_gdr3",)

#: RAVE's [alpha/Fe] is a by-product of the abundance pipeline, hence 0.5.
X_ERR = (0.1, 0.1, 0.1, 0.5)


def download(ctx) -> None:
    for fn in FNAMES:
        download_file(f"{URL_BASE}/{fn}", ctx.raw_dir / fn)


def load(ctx) -> Table:
    def read_catalog() -> Table:
        merged = None
        for fn in FNAMES:
            df = pd.read_hdf(ctx.raw_dir / fn)
            merged = df if merged is None else pd.merge(merged, df, on=JOIN_KEY, how="inner")
        return Table.from_pandas(merged)

    def drop_unwritable(catalog: Table) -> Table:
        if "cluster_flag" in catalog.colnames:  # object dtype, not HDF5-writable
            catalog.remove_column("cluster_flag")
        return catalog

    return crossmatch_catalog(
        read_catalog,
        ra_col="ra",
        dec_col="dec",
        cache_path=ctx.interim_dir / CROSSMATCH_FILE,
        search_separation=SEARCH_SEPARATION,
        max_separation=MAX_SEPARATION,
        force=ctx.force,
        allow_regression=ctx.allow_crossmatch_regression,
        key=KEY,
        verbose=True,
        postprocess=drop_unwritable,
        serialize_metadata=True,
    )


def cuts(t: Table, ctx) -> dict:
    fe_h = unmask(t["fe_h_gauguin"], fill=np.nan)

    return {
        "[Fe/H] finite": np.isfinite(fe_h),
        "GAUGUIN Fe/H chi2 < 1e-2": unmask(t["fe_h_chisq_gauguin"]) < 1e-2,
        "GAUGUIN Fe/H non-linear correction > 2": unmask(t["fe_h_nl_gauguin"]) > 2,
        "Teff error < 300 K": unmask(t["teff_error_bdasp"]) < 300.0,
        "logg error < 0.3": unmask(t["logg_error_bdasp"]) < 0.3,
        "[Fe/H] error < 0.3": unmask(t["fe_h_error_gauguin"]) < 0.3,
        "[alpha/Fe] error < 0.3": unmask(t["alpha_fe_error_gauguin"]) < 0.3,
    }


def param_errs(t: Table, ctx):
    return X_ERR
