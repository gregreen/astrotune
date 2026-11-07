"""VMP stars from Li et al. (2022).

References
----------
Li et al. (2022), ApJ 931, 147
Aoki et al. (2022), ApJ 931, 146 (survey design)

Notes
-----
The catalog ships coordinates but no Gaia identifiers, so it is crossmatched
against Gaia DR3 once and the result is cached in
``data/interim/vmp_stars_li2022/``.

[Fe/H] is trusted more than in the other catalogs: the regularization error on
that axis is 0.05 instead of the usual 0.1.
"""

from __future__ import annotations

import astropy.units as units
import numpy as np
from astropy.table import Table

from ..prep import plots
from ..prep.download import download as download_file
from ..prep.gaia import apply_crossmatch, gaia_crossmatch_batched
from ..prep.transforms import unmask

KEY = "vmp_stars_li2022"
LABEL = "VMP (Li)"
N_PARAMS = 3
REFERENCE = "Li et al. 2022, ApJ 931, 147"
TABLE_URL = "https://cdsarc.cds.unistra.fr/ftp/J/ApJ/931/147/table2.dat"
README_URL = "https://cdsarc.cds.unistra.fr/ftp/J/ApJ/931/147/ReadMe"
SOURCES = (TABLE_URL, README_URL)

TABLE_FILE = "table2.dat"
README_FILE = "readme.txt"
CROSSMATCH_FILE = "crossmatched.fits"

#: Maximum separation for the Gaia DR3 match.
MAX_SEPARATION = 0.5 * units.arcsec

ID_COL = "gdr3_source_id"
PARAM_COLS = ("Teff", "FeH", "logg")
UNC_COLS = ("e_Teff", "e_FeH", "e_logg")
EXTRA_COLS = ("ra_gdr3", "dec_gdr3")
ERROR_FLOORS = (0.01, 0.01, 0.04)

#: [Fe/H] is more trustworthy here than in the large surveys.
X_ERR = (0.1, 0.05, 0.1)


def download(ctx) -> None:
    download_file(TABLE_URL, ctx.raw_dir / TABLE_FILE)
    download_file(README_URL, ctx.raw_dir / README_FILE)


def load(ctx) -> Table:
    cache = ctx.interim_dir / CROSSMATCH_FILE
    if cache.exists() and not ctx.force:
        return Table.read(cache)

    catalog = Table.read(
        ctx.raw_dir / TABLE_FILE, format="cds", readme=str(ctx.raw_dir / README_FILE)
    )
    crossmatch = gaia_crossmatch_batched(
        catalog["RAdeg"], catalog["DEdeg"], max_separation=MAX_SEPARATION
    )
    matched = apply_crossmatch(catalog, crossmatch)

    cache.parent.mkdir(parents=True, exist_ok=True)
    matched.write(cache, overwrite=True)
    return matched


def cuts(t: Table, ctx) -> dict:
    """No additional quality cuts: the sample is small and already curated."""
    return {}


def param_errs(t: Table, ctx):
    return X_ERR


def diagnostics(t: Table, mask, table: Table, ctx):
    """Compare the catalog values with the LAMOST parameters they supersede."""
    import matplotlib.pyplot as plt

    axis_labels = ("Teff", "FeH", "logg")
    available = [c for c in (f"{p}-L" for p in axis_labels) if c in t.colnames]
    if not available:
        return []

    fig, axes = plt.subplots(1, len(available), figsize=(3 * len(available), 3), layout="constrained")
    for ax, name in zip(np.atleast_1d(axes), available):
        param = name[:-2]
        x = unmask(t[param], fill=np.nan)
        y = unmask(t[name], fill=np.nan)
        good = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[good], y[good], s=9, edgecolors="none", alpha=0.7)
        lo = np.nanmin([x[good].min(), y[good].min()])
        hi = np.nanmax([x[good].max(), y[good].max()])
        ax.plot([lo, hi], [lo, hi], c="k", alpha=0.5)
        ax.set_xlabel(f"{param} (Li+2022)")
        ax.set_ylabel(f"{param} (LAMOST)")

    return [plots.save(fig, ctx.diagnostic_dir, "li2022_vs_lamost")]
