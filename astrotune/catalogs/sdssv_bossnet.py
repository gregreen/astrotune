"""SDSS-V Milky Way Mapper (DR20): BOSSNet parameters from BOSS spectra.

References
----------
Sizemore et al. (2024), AJ 167, 173 (DOI: 10.3847/1538-3881/ad291d)
SDSS Collaboration (2026), arXiv:2607.26149 (DOI: 10.48550/arXiv.2607.26149)
https://dr20.sdss.org/sas/dr20/spectro/astra/0.8.1/summary/

BOSSNet is also described here:
https://sdss.org/dr19/mwm/astra/pipelines-in-astra/bossnet/

Notes
-----
The ``astra`` summary files keep the stellar parameters in HDU 1.

BOSSNet provides no [alpha/Fe], so this catalog has three parameters. Since
the parameters come from a neural-network fit to lower-resolution optical
spectra, the regularization error is set to 0.3 on every axis.

DR20 (astra 0.8.1) is the first release with southern-hemisphere BOSS spectra,
so the catalogue grows by roughly the southern sky; see
``porting_project.md`` for what that did to the cuts.

Caveat: SDSS warns that BOSSNet gives poor parameters for halo stars and that
MINESweeper or BOSS-CLAM should be used instead (DR20 paper, arXiv:2607.26149,
Sec. VI.1.2). The halo and open-fiber cartons dominate the DR20 growth, and the
astra summary files carry no carton column, so such stars cannot be removed
here.
"""

from __future__ import annotations

import numpy as np
from astropy.io import fits
from astropy.table import Table

from ..prep import plots
from ..prep.download import download as fetch
from ..prep.transforms import unmask

KEY = "sdssv_bossnet"
LABEL = "SDSS-V (BOSSNet)"
N_PARAMS = 3
REFERENCE = "Sizemore et al. 2024, AJ 167, 173"
SOURCES = (
    "https://dr20.sdss.org/sas/dr20/spectro/astra/0.8.1/summary/"
    "astraAllStarBossNet-0.8.1.fits.gz",
)
FILENAME = "astraAllStarBossNet-0.8.1.fits.gz"
HDU = 1

ID_COL = "gaia_dr3_source_id"
PARAM_COLS = ("teff", "fe_h", "logg")
UNC_COLS = ("e_teff", "e_fe_h", "e_logg")
EXTRA_COLS = ("ra", "dec")

#: Low confidence overall, so the transformation is weakly constrained.
X_ERR = (0.3, 0.3, 0.3)

#: The legacy table reproduces DR19. This module now reads DR20, which re-reduces
#: every DR19 spectrum and adds the southern hemisphere, so the sample itself
#: changed and a row-for-row comparison with the legacy table is meaningless.
PARITY_SKIP = "reads DR20 (astra 0.8.1); the legacy table reproduces DR19"


def download(ctx) -> None:
    # dr20.sdss.org is public: no SDSS-V credentials, so nothing prompts.
    fetch(SOURCES[0], ctx.raw_dir / FILENAME)


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
