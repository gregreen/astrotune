"""LAMOST M-type stars, "refined" catalog of Li et al. (2025), built on LAMOST DR10.

References
----------
Li, S., et al. 2025, ApJS 281, 58, "Refined M-type Star Catalog from LAMOST
DR10: Measurements of Radial Velocities, Teff, logg, [M/H] and [alpha/M]"
(doi:10.3847/1538-4365/ae1586).
Data (the paper's Table 9, "Recommended Catalog"): NADC, doi:10.12149/101668,
``Recommended_Catalog_Upload_v1.fits``. The file is public (no login).

Notes
-----
* **This is not the official LAMOST DR12 M-star catalog.** That file
  (``dr12_v2.0_LRS_mstellar.fits.gz``) has LASP parameters, three parameters and
  no quality flag; the catalog here is a CNN refinement of the *DR10* M-type
  catalog (870,518 spectra), hence the key has no ``_dr12``.
* One row per LAMOST spectrum, for dwarfs and giants (``luminosity_class``
  ``dM``/``dM_checked`` or ``gM_checked``). 24% of the rows are repeat spectra
  of a star already in the catalog (115,875 stars have two or more; up to 34),
  which the many-to-many join of the pair builder multiplies (project policy:
  duplicates are kept).
* Parameters: Teff, logg, [M/H], [alpha/M], with uncertainties. As for the
  other M-star catalogs [M/H] is used as the metallicity ([Fe/H]), and as for
  APOGEE and SDSS-V ASPCAP [alpha/M] is used as [alpha/Fe].
* **The CNN was trained on APOGEE labels**, so the agreement with APOGEE
  (scatter 0.06 / 0.04 / 0.13 / 0.025 in 5 ln Teff, [M/H], logg, [alpha/M]) is
  not an independent test, and pairs with APOGEE are not independent
  measurements; against the other catalogs the scatter is 2-5 times larger. It
  also uses the same LAMOST spectra as the other LAMOST catalogs (DD-Payne, the
  Ding and Zhang M-star catalogs).
* **Uncertainties.** They are the spread of an ensemble of ten cross-validated
  CNNs (internal precision), and too small: the scatter against the other
  catalogs in excess of its high-SNR plateau (an estimate of the random error)
  is 1.3-3.4 times the quoted error, depending on the parameter. They are
  scaled by :data:`UNC_SCALES` before the error floors are added.

Cuts (chosen by looking at the Kiel and [M/H]-[alpha/M] diagrams and at the
stars shared with APOGEE, ASPCAP, SLAM, MDw Cannon, DESI and LAMOST DD-Payne):

* ``parameter_flag == 1`` (the paper's recommended sample; 87% of the rows).
  It already contains the paper's reasonable-error cut and its "recommended
  ranges" (Table 10: dM Teff 3225-4125 K, logg 4.35-5.45, [M/H] -0.85...0.35,
  [alpha/M] -0.13...0.25; gM Teff 3425-4075 K, logg -0.05...1.55, [M/H]
  -1.35...0.45, [alpha/M] -0.03...0.31). Inside it the distributions are
  truncated by the flag at those limits rather than piled up against them
  (the highest fraction within 5% of an edge is 3%), so no further cuts on
  extreme values are needed.
* A total SNR, the quadrature sum of the five bands (negative values, the
  ``-9999`` sentinel, count as zero), above :data:`SNR_MIN`. For M stars it is
  ~1.4 x ``snr_i`` (the u and g bands are empty), so 30 is close to the paper's
  own ``snr_i > 20``. The scatter against the other catalogs is flat above it
  and 30-40% larger below.
* Gaia source id; and a class consistent with the Gaia CMD (the paper's own
  criterion for reliable members): stars outside both M-star regions
  (``CMD_pos == 'other'``, Teff scatter 0.39 against 0.14) or classified dM by
  LAMOST and gM by the CMD (or the reverse) are removed. Stars without Gaia
  astrometry (``CMD_pos`` empty, 18%) are kept.
* Finite uncertainties with loose upper limits (this removes no star: the
  flag already contains the paper's own error-reasonableness cut).
"""

from __future__ import annotations

import numpy as np
from astropy.table import Table

from ..prep import plots
from ..prep.download import download as download_file
from ..prep.transforms import PARAM_UNC_TRANSF, unmask

KEY = "lamost_mstar_li2025"
LABEL = "LAMOST M-Stars (Li)"
N_PARAMS = 4
REFERENCE = "Li et al. 2025, ApJS 281, 58 (LAMOST DR10 spectra)"
SOURCES = ("https://nadc.china-vo.org/res/file_upload/download?id=52019",)
FILENAME = "Recommended_Catalog_Upload_v1.fits"

ID_COL = "source_id"
PARAM_COLS = ("teff", "m_h", "logg", "alpha_m")
UNC_COLS = ("teff_err", "m_h_err", "logg_err", "alpha_m_err")
EXTRA_COLS = ("ra", "dec")

SNR_BANDS = "ugriz"
SNR_MIN = 30.0

#: Upper limits on the catalog's own uncertainties (Teff in K)
MAX_ERR = {"teff_err": 300.0, "logg_err": 0.5, "m_h_err": 0.4, "alpha_m_err": 0.15}

#: Factors applied to the catalog uncertainties before the error floors
UNC_SCALES = (2.0, 2.0, 2.0, 2.0)

#: Regularization errors: a first guess, like the other M-star catalogs
#: (SLAM, MDw Cannon 0.2; Ding 0.2/0.4; Zhang 0.1). Not tuned.
X_ERR = (0.2, 0.2, 0.2, 0.2)


def download(ctx) -> None:
    download_file(SOURCES[0], ctx.raw_dir / FILENAME, timeout=180.0)


def load(ctx) -> Table:
    t = Table.read(ctx.raw_dir / FILENAME)
    for name in ("luminosity_class", "CMD_pos", "subclass"):
        t[name] = np.char.strip(np.asarray(t[name]).astype(str))
    return t


def total_snr(t: Table) -> np.ndarray:
    """Quadrature sum of the SNR of the five bands (a negative SNR counts as 0)."""
    total = np.zeros(len(t))
    for band in SNR_BANDS:
        total += np.clip(unmask(t[f"snr_{band}"], fill=0.0).astype("f8"), 0.0, None) ** 2
    return np.sqrt(total)


def unc_transf(d: int):
    """Scale the catalog uncertainties before the floors are applied."""
    return [
        (lambda x, xs, f=f, s=s: f(x, s * xs))
        for f, s in zip(PARAM_UNC_TRANSF[:d], UNC_SCALES[:d])
    ]


def class_consistent_with_cmd(t: Table) -> np.ndarray:
    """False where the Gaia CMD contradicts the dM/gM class (or the star is not an M star)."""
    giant = np.char.startswith(np.asarray(t["luminosity_class"]).astype(str), "gM")
    cmd = np.asarray(t["CMD_pos"]).astype(str)
    contradicts = (cmd == "other") | (giant & (cmd == "dM")) | (~giant & (cmd == "gM"))
    return ~contradicts


def cuts(t: Table, ctx) -> dict:
    err = {k: unmask(t[k]).astype("f8") for k in MAX_ERR}
    errors_ok = np.ones(len(t), dtype=bool)
    for k, limit in MAX_ERR.items():
        errors_ok &= np.isfinite(err[k]) & (err[k] > 0) & (err[k] < limit)

    return {
        "Gaia source id": np.asarray(t[ID_COL]) > 0,
        "parameter_flag == 1": np.asarray(t["parameter_flag"]) == 1,
        f"SNR_tot > {SNR_MIN:g}": total_snr(t) > SNR_MIN,
        "class consistent with the Gaia CMD": class_consistent_with_cmd(t),
        "uncertainties finite and below limits": errors_ok,
    }


def param_errs(t: Table, ctx):
    return X_ERR


def diagnostics(t: Table, mask, table: Table, ctx):
    """SNR and the flag, and the quoted uncertainties against SNR."""
    import matplotlib.pyplot as plt

    keep = np.asarray(mask, dtype=bool)
    flagged = np.asarray(t["parameter_flag"]) == 1
    snr = total_snr(t)
    has_snr = snr > 0
    written = []

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
    bins = np.linspace(0, 3, 100)
    for sel, label, kw in (
        (has_snr, "all spectra", dict(histtype="step")),
        (has_snr & flagged, "parameter_flag == 1", dict(histtype="step")),
        (keep, "selected", dict(histtype="stepfilled", alpha=0.5)),
    ):
        ax0.hist(np.log10(snr[sel]), bins=bins, label=label, **kw)
    ax0.axvline(np.log10(SNR_MIN), c="k", ls="--", lw=0.8)
    ax0.set_yscale("log")
    ax0.set_xlabel(r"$\log_{10}\,\mathrm{SNR}_{\mathrm{tot}}$")
    ax0.legend(fontsize=7)
    edges = np.logspace(0, 2.8, 25)
    centers = np.sqrt(edges[1:] * edges[:-1])
    frac = [
        np.mean(flagged[(snr >= lo) & (snr < hi)]) if np.any((snr >= lo) & (snr < hi)) else np.nan
        for lo, hi in zip(edges[:-1], edges[1:])
    ]
    ax1.semilogx(centers, frac, "o-")
    ax1.axvline(SNR_MIN, c="k", ls="--", lw=0.8)
    ax1.set_xlabel(r"$\mathrm{SNR}_{\mathrm{tot}}$")
    ax1.set_ylabel("fraction with parameter_flag == 1")
    written.append(plots.save(fig, ctx.diagnostic_dir, "snr_flag"))

    fig, axes = plt.subplots(1, 4, figsize=(14, 3), layout="constrained")
    for ax, (col, label, scale) in zip(
        axes,
        (("teff_err", "Teff error (K)", 1), ("logg_err", "logg error", 1),
         ("m_h_err", "[M/H] error", 1), ("alpha_m_err", "[alpha/M] error", 1)),
    ):
        e = unmask(t[col]).astype("f8")
        ok = keep & np.isfinite(e)
        med = [np.median(e[ok & (snr >= lo) & (snr < hi)]) if np.any(ok & (snr >= lo) & (snr < hi)) else np.nan
               for lo, hi in zip(edges[:-1], edges[1:])]
        ax.loglog(centers, med, "o-", label="quoted")
        ax.loglog(centers, UNC_SCALES[0] * np.asarray(med), "o--", label=f"x{UNC_SCALES[0]:g} (used)")
        ax.set_xlabel(r"$\mathrm{SNR}_{\mathrm{tot}}$")
        ax.set_ylabel(label)
    axes[0].legend(fontsize=7)
    written.append(plots.save(fig, ctx.diagnostic_dir, "errors_vs_snr"))
    return written
