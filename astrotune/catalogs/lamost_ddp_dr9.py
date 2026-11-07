"""LAMOST LRS Data-Driven Payne DR9 (recommended June 2025 release).

References
----------
Zhang et al. (2025), arXiv:2506.02763
https://nadc.china-vo.org/res/r101614/
Zenodo: https://zenodo.org/records/15254859

Notes
-----
The catalog has no Gaia identifiers, so a DR3 crossmatch is generated once and
cached in ``data/interim/lamost_ddp_dr9/`` as the *crossmatch table*
(``match_idx`` etc.), which is then applied to the catalog.

A handful of abundances carry the sentinel value -99 without being flagged;
those flags are set to 98 before the alpha elements are averaged.

Two SNR-dependent chi^2 cuts are applied, one for the APOGEE and one for the
GALAH training grid.
"""

from __future__ import annotations

import astropy.units as units
import numpy as np
from astropy.io import fits
from astropy.table import Table

from ..prep import plots
from ..prep.abundances import calc_alpha_Fe_from_abundances
from ..prep.cuts import snr_chisq_relation
from ..prep.download import download as download_file
from ..prep.gaia import apply_crossmatch, gaia_crossmatch_batched
from ..prep.transforms import unmask

KEY = "lamost_ddp_dr9"
LABEL = "LAMOST DD-Payne (DR9)"
N_PARAMS = 4
REFERENCE = "Zhang et al. 2025, arXiv:2506.02763"
FILENAME = "LMDR9_DDPAYNE_recommend_202505.fits"
SOURCES = (f"https://zenodo.org/records/15254859/files/{FILENAME}?download=1",)
HDU = 1

CROSSMATCH_FILE = "crossmatch.h5"
MAX_SEPARATION = 0.5 * units.arcsec

ID_COL = "gdr3_source_id"
PARAM_COLS = ("TEFF", "FEH", "LOGG", "ALPHA_FE")
UNC_COLS = ("TEFF_ERR", "FEH_ERR", "LOGG_ERR", "ALPHA_FE_ERR")
EXTRA_COLS = ("ra_gdr3", "dec_gdr3")

#: Photometric bands combined into the total SNR.
SNR_BANDS = "UGRIZ"
#: Elements whose sentinel abundances need to be flagged by hand.
SENTINEL_ELEMENTS = ("O", "MG", "SI", "CA", "TI")
SENTINEL_VALUE = -99.0
SENTINEL_FLAG = 98

SNR_MIN = 40.0
#: Two excluded regions at the cool end of the grid, in (Teff, logg).
COOL_CUTS = ((4350.0, 3.6), (4200.0, 2.6))
#: Factor multiplying the fitted chi^2(SNR) relation.
CHISQ_FACTOR = 5.0


def download(ctx) -> None:
    download_file(SOURCES[0], ctx.raw_dir / FILENAME)


def total_snr(t: Table) -> np.ndarray:
    total = np.zeros(len(t))
    for band in SNR_BANDS:
        total += unmask(t[f"SNR{band}"], fill=0.0) ** 2
    return np.sqrt(total)


def load(ctx) -> Table:
    with fits.open(ctx.raw_dir / FILENAME, mode="readonly") as f:
        catalog = Table(f[HDU].data)

    cache = ctx.interim_dir / CROSSMATCH_FILE
    if cache.exists() and not ctx.force:
        crossmatch = Table.read(cache)
    else:
        crossmatch = gaia_crossmatch_batched(
            catalog["RA"], catalog["DEC"], max_separation=MAX_SEPARATION,
            max_batch_size=1024 * 1024,
        )
        cache.parent.mkdir(parents=True, exist_ok=True)
        crossmatch.write(cache, overwrite=True)

    return apply_crossmatch(catalog, crossmatch)


def prepare_columns(t: Table, ctx) -> Table:
    # Sentinel abundances that were not flagged by the pipeline.
    for element in SENTINEL_ELEMENTS:
        value_key, flag_key = f"{element}_FE", f"FLAG_{element}_FE"
        bad = (unmask(t[value_key], fill=np.nan) < SENTINEL_VALUE) & (
            unmask(t[flag_key]) == 0
        )
        if np.any(bad):
            t[flag_key][bad] = SENTINEL_FLAG

    alpha_fe, alpha_fe_err, abundances = calc_alpha_Fe_from_abundances(
        t, "{x.upper()}_FE", "{x.upper()}_FE_ERR", flag_fmt="FLAG_{x.upper()}_FE"
    )
    t["ALPHA_FE"] = alpha_fe
    t["ALPHA_FE_ERR"] = alpha_fe_err
    ctx.extra["abundances"] = abundances
    return t


def cuts(t: Table, ctx) -> dict:
    snr = total_snr(t)
    chisq_apo = unmask(t["CHISQ_LAM_APO"], fill=np.nan)
    chisq_gal = unmask(t["CHISQ_LAM_GAL"], fill=np.nan)

    f_apo = snr_chisq_relation(snr, chisq_apo)
    f_gal = snr_chisq_relation(snr, chisq_gal)
    ctx.extra["chisq_of_snr_apo"] = f_apo
    ctx.extra["chisq_of_snr_gal"] = f_gal

    teff = unmask(t["TEFF"], fill=np.nan)
    logg = unmask(t["LOGG"], fill=np.nan)

    excluded = np.zeros(len(t), dtype=bool)
    for teff_max, logg_min in COOL_CUTS:
        excluded |= (teff < teff_max) & (logg > logg_min)

    return {
        "UNIQFLAG == 1": unmask(t["UNIQFLAG"]) == 1,
        "SNR_tot > 40": snr > SNR_MIN,
        "APOGEE-grid chi2 below SNR-dependent limit": chisq_apo
        < CHISQ_FACTOR * f_apo(snr),
        "GALAH-grid chi2 below SNR-dependent limit": chisq_gal
        < CHISQ_FACTOR * f_gal(snr),
        "Teff > 3000 K": teff > 3000.0,
        "[Fe/H] > -9": unmask(t["FEH"]) > -9.0,
        "logg > -2": logg > -2.0,
        "uncertainties below limits": (
            (unmask(t["TEFF_ERR"]) < 200.0)
            & (unmask(t["LOGG_ERR"]) < 0.3)
            & (unmask(t["FEH_ERR"]) < 0.2)
            & (unmask(t["ALPHA_FE_ERR"]) < 0.2)
        ),
        "cool grid edges excluded": ~excluded,
    }


def diagnostics(t: Table, mask, table: Table, ctx):
    import matplotlib.pyplot as plt

    abundances = ctx.extra.get("abundances")
    if abundances is None:
        return []

    keep = np.asarray(mask, dtype=bool)
    alpha_fe = unmask(t["ALPHA_FE"], fill=np.nan)
    good = keep & np.isfinite(alpha_fe)

    fig, ax = plt.subplots(figsize=(5, 4), layout="constrained")
    snr = total_snr(t)
    chisq = unmask(t["CHISQ_LAM_APO"], fill=np.nan)
    f = ctx.extra.get("chisq_of_snr_apo")
    ok = keep & np.isfinite(snr) & np.isfinite(chisq) & (snr > 0) & (chisq > 0)
    ax.hist2d(np.log10(snr[ok]), np.log10(chisq[ok]), bins=80,
              norm=plt.matplotlib.colors.LogNorm(), rasterized=True)
    if f is not None:
        snr_plot = np.logspace(np.log10(max(snr[ok].min(), 1.0)), np.log10(snr[ok].max()), 128)
        ax.plot(np.log10(snr_plot), np.log10(f(snr_plot)), c="w", lw=1, alpha=0.8)
    ax.set_xlabel(r"$\log_{10}\,\mathrm{SNR}_{\mathrm{tot}}$")
    ax.set_ylabel(r"$\log_{10}\,\chi^2$ (APOGEE grid)")

    return [
        plots.save(fig, ctx.diagnostic_dir, "snr_chisq_apo"),
        plots.plot_alpha_abundances(
            alpha_fe[good], abundances[good], outdir=ctx.diagnostic_dir
        ),
    ]
