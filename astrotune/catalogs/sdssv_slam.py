"""SDSS-V SLAM parameters for M dwarfs.

References
----------
Qiu et al. (2025), arXiv:2511.20005
Zhang et al. (2020), ApJS 246, 9
Li et al. (2021), ApJS 253, 45
https://sdss.org/dr20/mwm/astra/pipelines-in-astra/slam/

Notes
-----
Downloaded from the public DR20 SAS (ASTRA 0.8.1), which needs no credentials;
the previous version came from the password-protected ``data.sdss5.org`` DR19
area (ASTRA 0.6.0).

The SLAM Teff uncertainties are systematically too large, so they are scaled
by :data:`UNC_SCALES` before the error floors are added. Because that quoted
error is not trusted, :data:`TEFF_ERR_MAX` is deliberately loose.
"""

from __future__ import annotations

import numpy as np
from astropy.table import Table

from ..prep import plots
from ..prep.download import download as _download
from ..prep.transforms import PARAM_UNC_TRANSF, unmask

KEY = "sdssv_slam"
LABEL = "SDSS-V (SLAM)"
N_PARAMS = 4
REFERENCE = "Qiu et al. 2025; Zhang et al. 2020; Li et al. 2021"
SOURCES = (
    "https://dr20.sdss.org/sas/dr20/spectro/astra/0.8.1/summary/"
    "astraAllStarSlam-0.8.1.fits.gz",
)
FILENAME = "astraAllStarSlam-0.8.1.fits.gz"

ID_COL = "gaia_dr3_source_id"
PARAM_COLS = ("teff", "fe_h", "logg", "alpha_fe")
UNC_COLS = ("e_teff", "e_fe_h", "e_logg", "e_alpha_fe")
EXTRA_COLS = ("ra", "dec")

#: The [alpha/Fe] regularization error was raised from the legacy 0.2 to 0.4 on
#: purpose (SLAM's [alpha/Fe] differs from nearly all other surveys), so the
#: parity check is allowed to differ on ``x_err[:, 3]`` only.
X_ERR = (0.2, 0.2, 0.2, 0.4)
PARITY_SKIP = "reads DR20 (astra 0.8.1); the legacy table reproduces DR19 (0.6.0)"

#: Per-parameter multiplier applied to the statistical uncertainties.
UNC_SCALES = (0.05, 1.0, 1.0, 1.0)

TEFF_RANGE = (3150.0, 3750.0)
FEH_RANGE = (-0.6, 0.3)
ALPHA_RANGE = (-0.2, 0.2)

#: Upper limit on the quoted Teff uncertainty, in K. SLAM's Teff errors are
#: over-estimated by a large factor (see :data:`UNC_SCALES`) and the delta model
#: recalibrates them anyway, so this cut is only a guard against nonsense values
#: and is deliberately loose: 6000 K, ten times the original 600 K.
TEFF_ERR_MAX = 6000.0


def download(ctx) -> None:
    # Public SAS: no credentials needed (see the module docstring). The import is
    # aliased because this function shadows the name.
    _download(SOURCES[0], ctx.raw_dir / FILENAME)


def load(ctx) -> Table:
    return Table.read(ctx.raw_dir / FILENAME)


def unc_transf(d: int):
    """Rescale the uncertainties before the floors are applied."""
    return [
        (lambda x, xs, f=f, s=s: f(x, s * xs))
        for f, s in zip(PARAM_UNC_TRANSF[:d], UNC_SCALES[:d])
    ]


def cuts(t: Table, ctx) -> dict:
    teff = unmask(t["teff"], fill=np.nan)
    fe_h = unmask(t["fe_h"], fill=np.nan)
    alpha_fe = unmask(t["alpha_fe"], fill=np.nan)

    return {
        "Teff in range": (teff > TEFF_RANGE[0]) & (teff < TEFF_RANGE[1]),
        f"Teff error < {TEFF_ERR_MAX:.0f} K": unmask(t["e_teff"]) < TEFF_ERR_MAX,
        "[Fe/H] in range": (fe_h > FEH_RANGE[0]) & (fe_h < FEH_RANGE[1]),
        "[alpha/Fe] in range": (alpha_fe > ALPHA_RANGE[0]) & (alpha_fe < ALPHA_RANGE[1]),
        "SNR > 15": unmask(t["snr"]) > 15.0,
    }


def param_errs(t: Table, ctx):
    return X_ERR


def diagnostics(t: Table, mask, table: Table, ctx):
    import matplotlib.pyplot as plt

    keep = np.asarray(mask, dtype=bool)
    snr = unmask(t["snr"], fill=np.nan)
    good = keep & np.isfinite(snr) & (snr > 0)

    fig, ax = plt.subplots(figsize=(5, 3.5), layout="constrained")
    ax.hist(np.log10(snr[good]), bins=50, range=(0, 3), histtype="step")
    ax.set_xlabel(r"$\log_{10}\,\mathrm{SNR}$")
    ax.set_ylabel("accepted stars")
    return [plots.save(fig, ctx.diagnostic_dir, "snr")]
