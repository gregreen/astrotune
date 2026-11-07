"""SDSS-V Milky Way Mapper (DR19): BOSSNet parameters from BOSS spectra.

References
----------
Sizemore et al. (2024), AJ 167, 173
https://sdss.org/dr19/mwm/astra/pipelines-in-astra/bossnet/

Notes
-----
The ``astra`` summary files keep the stellar parameters in HDU 1.

BOSSNet provides no [alpha/Fe], so this catalog has three parameters. Since
the parameters come from a neural-network fit to lower-resolution optical
spectra, the regularization error is set to 0.3 on every axis.
"""

from __future__ import annotations

import numpy as np
from astropy.io import fits
from astropy.table import Table

from ..prep import plots
from ..prep.download import download_sdss
from ..prep.transforms import unmask

KEY = "sdssv_bossnet"
LABEL = "SDSS-V (BOSSNet)"
N_PARAMS = 3
REFERENCE = "Sizemore et al. 2024, AJ 167, 173"
SOURCES = (
    "https://data.sdss5.org/sas/dr19/spectro/astra/0.6.0/summary/"
    "astraAllStarBossNet-0.6.0.fits.gz",
)
FILENAME = "astraAllStarBossNet-0.6.0.fits.gz"
HDU = 1

ID_COL = "gaia_dr3_source_id"
PARAM_COLS = ("teff", "fe_h", "logg")
UNC_COLS = ("e_teff", "e_fe_h", "e_logg")
EXTRA_COLS = ("ra", "dec")

#: Low confidence overall, so the transformation is weakly constrained.
X_ERR = (0.3, 0.3, 0.3)


def download(ctx) -> None:
    download_sdss(SOURCES[0], ctx.raw_dir / FILENAME)


def load(ctx) -> Table:
    with fits.open(ctx.raw_dir / FILENAME, mode="readonly") as f:
        return Table(f[HDU].data)


def cuts(t: Table, ctx) -> dict:
    teff = unmask(t["teff"])
    snr = unmask(t["snr"])

    return {
        "no warning/bad flags": ~(
            unmask(t["flag_warn"]).astype(bool) | unmask(t["flag_bad"]).astype(bool)
        ),
        "SNR > 20": snr > 20.0,
        "uncertainties below limits": (
            (unmask(t["e_teff"]) / teff < 0.1)
            & (unmask(t["e_fe_h"]) < 0.3)
            & (unmask(t["e_logg"]) < 0.3)
        ),
    }


def param_errs(t: Table, ctx):
    """Constant regularization errors; a tuple is broadcast by extract_table."""
    return X_ERR


def diagnostics(t: Table, mask, table: Table, ctx):
    import matplotlib.pyplot as plt

    keep = np.asarray(mask, dtype=bool)
    snr = unmask(t["snr"], fill=np.nan)
    good = keep & np.isfinite(snr) & (snr > 0)

    fig, ax = plt.subplots(figsize=(5, 3.5), layout="constrained")
    ax.hist(np.log10(snr[good]), bins=100, range=(0, 3), histtype="step")
    ax.set_xlabel(r"$\log_{10}\,\mathrm{SNR}$")
    ax.set_ylabel("accepted stars")
    return [plots.save(fig, ctx.diagnostic_dir, "snr")]
