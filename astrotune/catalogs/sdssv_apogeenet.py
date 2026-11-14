"""SDSS-V Milky Way Mapper (DR20): APOGEENet parameters from APOGEE spectra.

References
----------
Olney et al. (2020), AJ 159, 182 (DOI: 10.3847/1538-3881/ab7a97)
Sprague et al. (2022), AJ 163, 152 (DOI: 10.3847/1538-3881/ac4de7)
Sizemore et al. (2024), AJ 167, 173 (DOI: 10.3847/1538-3881/ad291d)
SDSS Collaboration (2026), arXiv:2607.26149 (DOI: 10.48550/arXiv.2607.26149)
https://www.sdss.org/dr20/mwm/astra/pipelines-in-astra/apogeenet/

Notes
-----
The ``astra`` summary files keep the stellar parameters in HDU 2; the
visit-level table in HDU 1 is empty in this release.

DR20 adds no new APOGEE spectra: this is the ``astra`` 0.6.0 product first
released with DR19 (the file predates DR20), and it mixes SDSS-IV DR17 ``apStar``
spectra (``release == "dr17"``, 733,901 rows, the same spectra that feed
``apogee_dr17``) with SDSS-V spectra (``release == "sdss5"``, 388,416 rows, the
same stars as ``sdssv_aspcap``). Expect heavy overlap with both.

About a tenth of the rows are repeat measurements of a star already in the table
(different spectra, so the project's duplicate-id policy keeps them).

APOGEENet v3 predicts from the spectrum alone -- no photometry, no astrometry --
so a missing-photometry warning says nothing about the parameters, and in this
file ``flag_warn`` in fact never fires on its own (12,836 rows, all of them also
``flag_bad``). Dropping ``flag_warn`` therefore costs nothing.

We use the *recommended* uncertainties (``e_*``), not the ``raw_*`` ones: astra
inflates them as ``1.25 raw + 10 K`` (Teff), ``1.25 raw + 0.01`` (logg) and
``raw + 0.01`` ([Fe/H]). Redoing the SDSS validation with the repeat pairs in our
own cut sample gives median ``|z|`` of 0.77 (Teff), 0.82 (logg) and 0.63
([Fe/H]), with 3-5% of pairs beyond ``|z| = 3``: the inflated errors are close to
calibrated, if slightly optimistic in the tails.

``spectrum_flags`` is deliberately *not* cut on, unlike in ``sdssv_aspcap``. The
mask that driver uses (``2**0 | 2**2 | 2**19``) would drop 11.7% of this sample,
but in the repeat pairs the flagged rows are no less self-consistent (Teff rms
318 K against 421 K, logg 0.100 against 0.083, [Fe/H] 0.059 against 0.061), so
the bits carry no information about the APOGEENet parameters.

APOGEENet's own flags already bound the domain at the cool and low-gravity ends:
no row that clears ``flag_bad`` has Teff < 3200 K or logg > 5.0, so the rounded
edges in the prepared table are the pipeline's ``TEFF_UNRELIABLE`` and
``LOGG_UNRELIABLE`` limits, not ours. The hot end is not guarded in the same way:
the file reaches 61,301 K and 1,582 survivors sit above 20,000 K with no flag
set. That tail is left in on purpose -- HotPayne is otherwise our only hot
anchor.

Comparing against ``apogee_dr17`` on the 510,699 shared stars shows what the
bijector has to learn and how large the residual scatter is. APOGEENet runs cool
by 2-3.5% below 5500 K and hot by +12% at 7000-8000 K and +21% at 8000-10,000 K,
with [Fe/H] 0.10-0.14 dex lower between 5500 and 7000 K; those offsets are real
structure, not error. The scatter -- which is what ``x_err`` models -- is 1.9-2.7%
in Teff and 0.06-0.10 dex in [Fe/H] below 7000 K, matching the inherited ``x_err``
of 0.1 on those axes, but 0.17-0.28 dex in logg (0.218 overall), twice the 0.1
used here and by both sibling catalogs. That one is left as inherited rather than
changed unilaterally, since it sets the weight of the catalog in the fit.

APOGEENet has no [alpha/Fe], so this catalog has three parameters.
"""

from __future__ import annotations

import numpy as np
from astropy.io import fits
from astropy.table import Table

from ..prep import plots
from ..prep.cuts import finite_mask
from ..prep.download import download as fetch
from ..prep.transforms import unmask

KEY = "sdssv_apogeenet"
LABEL = "SDSS-V (APOGEENet)"
N_PARAMS = 3
REFERENCE = "Olney et al. 2020, AJ 159, 182; Sprague et al. 2022, AJ 163, 152; Sizemore et al. 2024, AJ 167, 173"
SOURCES = (
    "https://dr20.sdss.org/sas/dr20/spectro/astra/0.6.0/summary/"
    "astraAllStarApogeeNet-0.6.0.fits.gz",
)
FILENAME = "astraAllStarApogeeNet-0.6.0.fits.gz"
HDU = 2

ID_COL = "gaia_dr3_source_id"
PARAM_COLS = ("teff", "fe_h", "logg")
UNC_COLS = ("e_teff", "e_fe_h", "e_logg")
EXTRA_COLS = ("ra", "dec")

#: ASPCAP's useful domain; APOGEENet does not flag [Fe/H] outside it itself.
FE_H_RANGE = (-2.3, 0.9)


def download(ctx) -> None:
    # dr20.sdss.org is public: no SDSS-V credentials, so nothing prompts.
    fetch(SOURCES[0], ctx.raw_dir / FILENAME)


def load(ctx) -> Table:
    with fits.open(ctx.raw_dir / FILENAME, mode="readonly") as f:
        return Table(f[HDU].data)


def cuts(t: Table, ctx) -> dict:
    teff = unmask(t["teff"])
    fe_h = unmask(t["fe_h"])
    snr = unmask(t["snr"])

    return {
        "no warning/bad flags": ~(
            unmask(t["flag_warn"]).astype(bool) | unmask(t["flag_bad"]).astype(bool)
        ),
        "parameters finite": finite_mask(t, PARAM_COLS),
        "[Fe/H] in range": (fe_h > FE_H_RANGE[0]) & (fe_h < FE_H_RANGE[1]),
        "SNR > 20": snr > 20.0,
        "uncertainties below limits": (
            (unmask(t["e_teff"]) / teff < 0.1)
            & (unmask(t["e_fe_h"]) < 0.3)
            & (unmask(t["e_logg"]) < 0.3)
        ),
    }


def param_errs(t: Table, ctx) -> np.ndarray:
    """Hot-star ramp on Teff and [Fe/H], as in ``sdssv_aspcap``."""
    teff = unmask(t["teff"], fill=np.nan).astype("f8")

    hot = np.clip(0.1 + 0.4 * (teff - 6000.0) / 1000.0, 0.1, 1.0)
    hot = np.where(np.isfinite(hot), hot, 0.1)

    n = len(t)
    return np.stack(
        [
            hot,               # Teff
            hot,               # [Fe/H]: APOGEENet metallicities degrade in the same regime
            np.full(n, 0.1),   # logg
        ],
        axis=1,
    ).astype("f4")


def _repeat_pairs(ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Index pairs ``(i, j)`` of the first two rows that share an identifier."""
    order = np.argsort(ids, kind="stable")
    sorted_ids = ids[order]
    starts = np.flatnonzero(np.r_[True, sorted_ids[1:] != sorted_ids[:-1]])
    counts = np.diff(np.r_[starts, len(sorted_ids)])
    first = starts[counts >= 2]
    return order[first], order[first + 1]


def diagnostics(t: Table, mask, table: Table, ctx):
    """SNR distribution, and a check of the errors against repeat observations."""
    import matplotlib.pyplot as plt

    keep = np.asarray(mask, dtype=bool)
    snr = unmask(t["snr"], fill=np.nan)
    good = keep & np.isfinite(snr) & (snr > 0)

    fig, ax = plt.subplots(figsize=(5, 3.5), layout="constrained")
    ax.hist(np.log10(snr[good]), bins=100, range=(0, 4), histtype="step")
    ax.axvline(np.log10(20.0), c="k", ls=":", alpha=0.4)
    ax.set_xlabel(r"$\log_{10}\,\mathrm{SNR}$")
    ax.set_ylabel("accepted stars")
    paths = [plots.save(fig, ctx.diagnostic_dir, "snr")]

    # Two spectra of the same star must agree to within the quoted errors. The
    # z-scores are computed in physical units, so the Teff scaling of x does not
    # apply: sigma_z = sqrt(sigma_1^2 + sigma_2^2).
    ids = unmask(t[ID_COL], fill=0).astype("i8")
    teff = unmask(t["teff"], fill=np.nan)
    fe_h = unmask(t["fe_h"], fill=np.nan)
    logg = unmask(t["logg"], fill=np.nan)
    e_teff = unmask(t["e_teff"], fill=np.nan)
    e_fe_h = unmask(t["e_fe_h"], fill=np.nan)
    e_logg = unmask(t["e_logg"], fill=np.nan)
    usable = (
        keep
        & (ids > 0)
        & np.isfinite(teff)
        & np.isfinite(fe_h)
        & np.isfinite(logg)
        & np.isfinite(e_teff)
        & np.isfinite(e_fe_h)
        & np.isfinite(e_logg)
    )
    idx = np.flatnonzero(usable)
    i, j = _repeat_pairs(ids[idx])
    i, j = idx[i], idx[j]

    series = {
        "Teff": (teff, e_teff),
        "[Fe/H]": (fe_h, e_fe_h),
        "logg": (logg, e_logg),
    }
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), layout="constrained")
    for ax, (name, (value, err)) in zip(axes, series.items()):
        sigma = np.sqrt(err[i] ** 2 + err[j] ** 2)
        z = (value[i] - value[j]) / np.where(sigma > 0, sigma, np.nan)
        z = z[np.isfinite(z)]
        # A range (not a np.clip) keeps outliers out of the edge bins, where they
        # would look like a spike in the distribution.
        ax.hist(z, bins=100, range=(-5, 5), histtype="step")
        ax.set_xlabel(rf"$z$ ({name})")
        ax.set_title(
            f"{name}: median $|z|$ = {np.median(np.abs(z)):.2f}, "
            f"{100 * np.mean(np.abs(z) > 3):.1f}% beyond $3\\sigma$",
            fontsize=8,
        )
    axes[0].set_ylabel(f"repeat pairs ({len(i):,})")
    paths.append(plots.save(fig, ctx.diagnostic_dir, "repeat_pair_zscores"))

    return paths
