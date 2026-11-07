"""DESI EDR Data-Driven Payne (Zhang et al. 2024).

References
----------
Zhang et al. (2024), ApJS 273, 19

Notes
-----
The CDS catalog has no Gaia identifiers, so a DR3 crossmatch is generated
once and cached in ``data/interim/desi_ddp/``.

[alpha/Fe] is the inverse-variance weighted mean of the individual
alpha-element abundances.
"""

from __future__ import annotations

import astropy.units as units
import numpy as np
from astropy.io import fits
from astropy.table import Table

from ..prep import plots
from ..prep.abundances import calc_alpha_Fe_from_abundances
from ..prep.download import download as download_file
from ..prep.gaia import apply_crossmatch, gaia_crossmatch_batched
from ..prep.transforms import unmask

KEY = "desi_ddp"
LABEL = "DESI (Zhang)"
N_PARAMS = 4
REFERENCE = "Zhang et al. 2024, ApJS 273, 19"
TABLE_URL = "https://cdsarc.cds.unistra.fr/viz-bin/nph-Cat/fits.gz?J/ApJS/273/19/catalog.dat.gz"
SOURCES = (TABLE_URL,)
FILENAME = "desi_edr_zhang_etal_ddpayne.fits.gz"
HDU = 1

CROSSMATCH_FILE = "crossmatched.h5"
MAX_SEPARATION = 0.5 * units.arcsec

ID_COL = "gdr3_source_id"
PARAM_COLS = ("Teff", "[Fe/H]", "logg", "[alpha/Fe]")
UNC_COLS = ("e_Teff", "e_[Fe/H]", "e_logg", "e_[alpha/Fe]")
EXTRA_COLS = ("ra", "dec")

TEFF_RANGE = (4100.0, 8000.0)
#: Cool, high-gravity stars sit where the DD-Payne grid is unreliable.
COOL_DWARF_TEFF = 4400.0
COOL_DWARF_LOGG = 3.0


def download(ctx) -> None:
    download_file(TABLE_URL, ctx.raw_dir / FILENAME)


def load(ctx) -> Table:
    cache = ctx.interim_dir / CROSSMATCH_FILE
    if cache.exists() and not ctx.force:
        return Table.read(cache)

    with fits.open(ctx.raw_dir / FILENAME, mode="readonly") as f:
        catalog = Table(f[HDU].data)

    crossmatch = gaia_crossmatch_batched(
        catalog["RAdeg"], catalog["DEdeg"], max_separation=MAX_SEPARATION
    )
    matched = apply_crossmatch(catalog, crossmatch)

    cache.parent.mkdir(parents=True, exist_ok=True)
    matched.write(cache, serialize_metadata=True, overwrite=True)
    return matched


def prepare_columns(t: Table, ctx) -> Table:
    alpha_fe, alpha_fe_err, abundances = calc_alpha_Fe_from_abundances(
        t, "[{x}/Fe]", "e_[{x}/Fe]"
    )
    t["[alpha/Fe]"] = alpha_fe
    t["e_[alpha/Fe]"] = alpha_fe_err
    ctx.extra["abundances"] = abundances
    return t


def cuts(t: Table, ctx) -> dict:
    teff = unmask(t["Teff"], fill=np.nan)
    logg = unmask(t["logg"], fill=np.nan)
    snr = unmask(t["snrB"], fill=np.nan)
    chi2 = unmask(t["Chi2"], fill=np.nan)

    with np.errstate(divide="ignore", invalid="ignore"):
        reduced_chi2 = chi2 / snr**0.25

    return {
        "SNR(B) > 20": snr > 20.0,
        "chi2/SNR^0.25 < 3": reduced_chi2 < 3.0,
        "[Fe/H] > -3": unmask(t["[Fe/H]"]) > -3.0,
        "Teff in range": (teff > TEFF_RANGE[0]) & (teff < TEFF_RANGE[1]),
        "cool dwarf locus excluded": ~((teff < COOL_DWARF_TEFF) & (logg > COOL_DWARF_LOGG)),
        "uncertainties below limits": (
            (unmask(t["e_Teff"]) < 200.0)
            & (unmask(t["e_logg"]) < 0.2)
            & (unmask(t["e_[Fe/H]"]) < 0.2)
        ),
    }


def param_errs(t: Table, ctx) -> np.ndarray:
    teff = unmask(t["Teff"], fill=np.nan).astype("f8")

    hot = np.clip(0.1 + 0.4 * (teff - 6000.0) / 1000.0, 0.1, 1.0)
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
    alpha_fe = unmask(t["[alpha/Fe]"], fill=np.nan)
    good = keep & np.isfinite(alpha_fe)
    return [
        plots.plot_alpha_abundances(
            alpha_fe[good], abundances[good], outdir=ctx.diagnostic_dir
        )
    ]
