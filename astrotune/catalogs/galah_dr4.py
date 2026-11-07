"""GALAH DR4 stellar parameters.

References
----------
Buder et al. (2024), PASA 42, 51
https://www.galah-survey.org/dr4/overview/

Notes
-----
GALAH does not provide a single [alpha/Fe] column, so it is built from the
individual alpha-element abundances (inverse-variance weighted mean).

``logg == 5`` is a placeholder used by GALAH and is removed, as are stars
hotter than 7950 K, where the ASPCAP-like grid performs poorly.
"""

from __future__ import annotations

import numpy as np
from astropy.table import Table

from ..prep import plots
from ..prep.abundances import calc_alpha_Fe_from_abundances
from ..prep.download import download as download_file
from ..prep.transforms import unmask

KEY = "galah_dr4"
LABEL = "GALAH (DR4)"
N_PARAMS = 4
REFERENCE = "Buder et al. 2024, PASA 42, 51"
SOURCES = (
    "https://cloud.datacentral.org.au/teamdata/GALAH/public/GALAH_DR4/catalogs/"
    "galah_dr4_allstar_240705.fits",
)
FILENAME = "galah_dr4_allstar_240705.fits"

ID_COL = "gaiadr3_source_id"
PARAM_COLS = ("teff", "fe_h", "logg", "alpha_fe")
UNC_COLS = ("e_teff", "e_fe_h", "e_logg", "e_alpha_fe")
EXTRA_COLS = ("ra", "dec")

#: GALAH's placeholder surface gravity for stars without a reliable value.
PLACEHOLDER_LOGG = 5.0
#: Hot limit of the parameter grid.
TEFF_MAX = 7950.0
#: Lower limit on the mean alpha abundance.
ALPHA_MIN = -0.4


def download(ctx) -> None:
    download_file(SOURCES[0], ctx.raw_dir / FILENAME)


def load(ctx) -> Table:
    return Table.read(ctx.raw_dir / FILENAME)


def prepare_columns(t: Table, ctx) -> Table:
    """Add the inverse-variance weighted mean [alpha/Fe]."""
    alpha_fe, alpha_fe_err, abundances = calc_alpha_Fe_from_abundances(
        t,
        "{x.lower()}_fe",
        "e_{x.lower()}_fe",
        flag_fmt="flag_{x.lower()}_fe",
    )
    t["alpha_fe"] = alpha_fe
    t["e_alpha_fe"] = alpha_fe_err
    ctx.extra["abundances"] = abundances
    return t


def cuts(t: Table, ctx) -> dict:
    teff = unmask(t["teff"], fill=np.nan)
    # A masked logg is not a valid measurement, so treat it as the placeholder.
    logg = unmask(t["logg"], fill=PLACEHOLDER_LOGG)
    alpha_fe = unmask(t["alpha_fe"], fill=np.nan)

    return {
        "SNR (CCD3) > 30": unmask(t["snr_px_ccd3"]) > 30.0,
        "flag_sp == 0": unmask(t["flag_sp"]) == 0,
        "flag_fe_h == 0": unmask(t["flag_fe_h"]) == 0,
        "logg is not a placeholder": logg != PLACEHOLDER_LOGG,
        "Teff below grid limit": teff < TEFF_MAX,
        "alpha/Fe above floor": alpha_fe > ALPHA_MIN,
        "e_logg finite": np.isfinite(unmask(t["e_logg"], fill=np.nan)),
    }


def param_errs(t: Table, ctx) -> np.ndarray:
    teff = unmask(t["teff"], fill=np.nan).astype("f8")

    hot = np.clip(0.1 + 0.4 * (teff - 6000.0) / 1000.0, 0.1, np.inf)
    hot = np.where(np.isfinite(hot), hot, 0.1)

    n = len(t)
    return np.stack(
        [
            hot,
            np.full(n, 0.1),
            np.full(n, 0.1),
            np.full(n, 0.05),
        ],
        axis=1,
    ).astype("f4")


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
