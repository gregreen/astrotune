"""LAMOST HotPayne parameters for hot stars.

References
----------
Xiang et al. (2022), A&A 662, A66

Notes
-----
The reduced chi^2 of the fit grows with SNR, so the quality cut uses an
SNR-dependent threshold, ``chisqred < 4 f(4 snrg)``, where ``f`` is fitted to
the chi^2-SNR relation of the catalog itself.

Careful: the original analysis used the *value* column ``[alpha/Fe]`` as the
uncertainty column for that axis (``param_unc_cols[3] == '[alpha/Fe]'`` instead
of ``'e_[alpha/Fe]'``), so the sigma_x of that axis was set to the abundance
itself rather than to its uncertainty. That typo is **fixed** here, which makes
``sigma_x[:, 3]`` differ from the legacy table; see :data:`PARITY_EXEMPT`.
"""

from __future__ import annotations

import numpy as np
from astropy.table import Table

from ..prep import plots
from ..prep.abundances import calc_alpha_Fe_from_abundances
from ..prep.cuts import finite_mask, snr_chisq_relation
from ..prep.download import download as download_file
from ..prep.transforms import unmask

KEY = "lamost_hotpayne"
LABEL = "LAMOST HotPayne"
N_PARAMS = 4
REFERENCE = "Xiang et al. 2022, A&A 662, A66"
SOURCES = ("https://cdsarc.cds.unistra.fr/viz-bin/nph-Cat/fits.gz?J/A+A/662/A66/table5.dat.gz",)
FILENAME = "lamost_hot_payne.fits.gz"

ID_COL = "GaiaEDR3"
PARAM_COLS = ("Teff", "[Fe/H]", "logg", "[alpha/Fe]")
UNC_COLS = ("e_Teff", "e_[Fe/H]", "e_logg", "e_[alpha/Fe]")
EXTRA_COLS = ("RAdeg", "DEdeg")
X_ERR = (0.05, 0.2, 0.1, 0.5)

#: The legacy table used the ``[alpha/Fe]`` value column as the uncertainty for
#: that axis. This is fixed here on purpose, so the parity check is allowed to
#: fail on ``sigma_x[:, 3]`` only.
PARITY_EXEMPT = (
    "sigma_x[:, 3] uses e_[alpha/Fe]; the legacy table used [alpha/Fe] by mistake"
)

SNR_MIN = 16.0
FEH_MIN = -3.95


def download(ctx) -> None:
    download_file(SOURCES[0], ctx.raw_dir / FILENAME)


def load(ctx) -> Table:
    return Table.read(ctx.raw_dir / FILENAME)


def prepare_columns(t: Table, ctx) -> Table:
    """Convert the mean [alpha/H] into [alpha/Fe]."""
    alpha_h, alpha_h_err, abundances = calc_alpha_Fe_from_abundances(
        t, "[{x}/H]", "e_[{x}/H]"
    )
    fe_h = unmask(t["[Fe/H]"], fill=np.nan)
    fe_h_err = unmask(t["e_[Fe/H]"], fill=np.nan)

    t["[alpha/Fe]"] = alpha_h - fe_h
    t["e_[alpha/Fe]"] = np.sqrt(alpha_h_err**2 + fe_h_err**2)
    ctx.extra["abundances"] = abundances
    return t


def cuts(t: Table, ctx) -> dict:
    snr = unmask(t["snrg"], fill=np.nan)
    chisqred = unmask(t["chisqred"], fill=np.nan)

    chisq_of_snr = snr_chisq_relation(snr, chisqred)
    ctx.extra["chisq_of_snr"] = chisq_of_snr

    return {
        "reduced chi2 below SNR-dependent limit": chisqred < 4 * chisq_of_snr(4 * snr),
        "SNR > 16": snr > SNR_MIN,
        "uncertainties finite": finite_mask(t, ("e_Teff", "e_logg", "e_[Fe/H]")),
        "[Fe/H] > -3.95": unmask(t["[Fe/H]"]) > FEH_MIN,
    }


def param_errs(t: Table, ctx):
    return X_ERR


def diagnostics(t: Table, mask, table: Table, ctx):
    import matplotlib.pyplot as plt

    keep = np.asarray(mask, dtype=bool)
    snr = unmask(t["snrg"], fill=np.nan)
    chisqred = unmask(t["chisqred"], fill=np.nan)
    f = ctx.extra.get("chisq_of_snr")

    good = keep & np.isfinite(snr) & np.isfinite(chisqred) & (snr > 0) & (chisqred > 0)
    fig, ax = plt.subplots(figsize=(5, 4), layout="constrained")
    ax.scatter(snr[good], chisqred[good], s=2, alpha=0.2, edgecolors="none", label="accept", rasterized=True)
    rejected = (~keep) & np.isfinite(snr) & np.isfinite(chisqred) & (snr > 0) & (chisqred > 0)
    ax.scatter(snr[rejected], chisqred[rejected], s=2, alpha=0.1, edgecolors="none", label="reject", rasterized=True)
    if f is not None:
        snr_plot = np.logspace(np.log10(5.0), 3.0, 100)
        ax.plot(snr_plot, f(snr_plot), c="k", alpha=0.6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-2, 1e5)
    ax.set_xlabel("snrg")
    ax.set_ylabel("chisqred")
    ax.legend(loc="upper left")
    return [plots.save(fig, ctx.diagnostic_dir, "snr_chisq")]
