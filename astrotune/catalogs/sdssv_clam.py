"""SDSS-V BOSS-CLAM stellar parameters (DR20 value-added catalog).

References
----------
Medan, I., Casey, A. R., Ji, A. P., et al. 2026, "BOSS-CLAM: Utilizing a
Constrained Linear Absorption Model to Infer Stellar Parameters from BOSS
Spectra", arXiv:2607.22822, https://scixplorer.org/abs/2026ApJS..287....5M/abstract
VAC: https://www.sdss.org/dr20/data_access/value-added-catalogs/?vac_id=10036
Data model: https://data.sdss.org/datamodel/files/MWM_STELLAR_PARAMS/boss_clam/boss_clam_lite.html

Notes
-----
* One row per BOSS spectrum (1,708,214), covering M dwarfs to OB stars. The
  file is public (no SDSS login) and has Gaia DR3 source ids; no crossmatch is
  needed.
* Teff, logg, [Fe/H] and [alpha/M]. As for APOGEE and SDSS-V ASPCAP, [alpha/M]
  is used as the fourth parameter ([alpha/Fe]); the catalog gives no [M/H], so
  the two cannot be told apart, and the difference is small where [M/H] ~ [Fe/H].
* Uncertainties are the square roots of the diagonal of ``param_covariance``.
  **They are far too small**, and the diagonal is only half the story.
  *Scale*: against DESI DR1 and LAMOST DD-Payne the [Fe/H] scatter is 0.25 at
  SNR 15-22 and 0.18 above SNR 45, where the catalog quotes 0.034 and 0.018;
  against APOGEE and ASPCAP (bright stars only) it is 0.07-0.10 at 0.013.
  ``astrotune-error-scale sdssv_clam --partners desi_dr1 lamost_ddp_dr9
  apogee_dr17 sdssv_aspcap`` fits s = 2.9-3.9 (Teff), 3.5-3.7 ([Fe/H]),
  2.2-3.9 (logg), 3.5-4.2 ([alpha/M]) with plateaus 0.07-0.13; ``UNC_SCALES``
  and ``ERROR_FLOORS`` below apply those numbers. The floors are what the bulk
  of the sample needs -- the overlap sample the fit uses is biased bright.
  **Why it matters**: with the old 1.5 scale this catalog was **44% of the whole
  data term** while being 11% of the observations (mean chi^2 per row 183
  against 6-14 for the large catalogs), so the map and the latent ``y`` were
  being fitted to its noise; that is what made the map distort and the training
  loss spike. ``scripts/catalog_loss_weight.py --run-dir RUN`` measures the
  share for every catalog. Sizing the floors to the measured scatter puts CLAM
  at ~7%.
  *Shape*: the quoted covariance is strongly non-diagonal -- corr(Teff, [Fe/H])
  = +0.85, corr([Fe/H], [alpha/M]) = -0.92, corr(Teff, logg) = +0.70 over the
  kept rows, with 85-96% of rows above |rho| = 0.5. Only the diagonal is used,
  and the model's residuals reproduce the tilt (corr(f(x) - y) = 0.91 in
  Teff-[Fe/H]), which no single smooth map can remove. Correcting the scale
  makes that a second-order effect; representing the covariance itself needs a
  full-covariance data term, which the trainer does not have.
* **Flags.** ``flag_bad`` (the catalog's own bad-fit flags, 43% of the rows)
  removes the star. ``flag_warn`` (``IN_OB`` or ``FE_H_ALPHA_SUSPECT``: every
  star above 6500 K -- all 30,982 of them in the prepared sample -- plus stars
  of the OB target program at any Teff) means that [alpha/M] should not be
  trusted, and the paper says the same of [Fe/H] for those stars. **Those rows
  are removed**, so that the catalog is four-parameter for every star it
  contributes (57,729 rows, 6.5%; see the note below).

* **Why partial coverage cannot be kept.** A catalog in which only *some* stars
  have [alpha/M] forces the learned map to be [alpha/Fe]-dependent, and the map
  is then not invertible for the rest. The data term only constrains
  ``f^-1(y)`` on the coordinates a row measures, so at inference ``x[alpha]`` is
  an exact one-dimensional degeneracy of ``p(x | y)``, and evaluating the map at
  the placeholder ``x[alpha] = 0`` gives a harmonized position that is *not* the
  star's ``y`` -- measured at 0.2-0.3 dex in [Fe/H] for the rows dropped here,
  against 0.02 dex for the ones kept. Dropping the 6.5% of rows costs ~2% of the
  stars in the merged sample (those left with a single catalog) and makes CLAM
  behave like the fifteen other four-parameter catalogs.

Cuts (chosen by looking at the Kiel and [Fe/H]-[alpha/M] diagrams and at the
stars shared with APOGEE, GALAH, ASPCAP and the cool-star catalogs):

* ``flag_bad`` clear and SNR >= 15. The catalog already requires SNR >= 10. The
  scatter against the references is flat above SNR ~ 30 and grows to ~1.7x at
  SNR 10-20; 20 would remove a third of the cool DESI anchors.
* Finite, positive-variance uncertainties and loose upper limits on them.
* **[Fe/H] and [alpha/M] measured for every kept row**: ``flag_warn`` clear, a
  finite value and an uncertainty inside :data:`MAX_ERR`. The prepared table
  therefore has no placeholder in either column, which is what makes the catalog
  uniformly four-parameter.
* Teff > 3400 K, a cut just inside the cool edge of the sample. Below ~3500 K
  the distribution drops steeply and the agreement with SLAM/MDC/DESI degrades
  toward it (the [alpha/M] offset grows from ~0 at 3600-3700 K to +0.11 below
  3400 K). That trend is measured in bins of CLAM's own noisy Teff, so part of
  it is an edge/selection effect and the cut is a judgment call; 3400 K rather
  than 3500 K keeps ~530 SLAM anchors (3400-3500 K) and removes only the sparse
  tail (33 shared stars).
* **No upper Teff cut.** None is needed now: ``flag_warn`` is set for every
  star above 6500 K and those rows are removed, so the hot end leaves the sample
  whatever :data:`TEFF_MAX` says (setting it to 8000 would cut nothing; only a
  value below 6500 K would bite). The caveats recorded here before the warning
  cut still describe the rows that are dropped, and ``sample_edges.png`` still
  shows them: above ~8000 K (6,990 rows) logg is pinned near 4.1 (horizontal
  stripes in the Kiel diagram) and the Teff and [Fe/H] offsets against other
  catalogs are large (e.g. Teff +7-15%, [Fe/H] +0.6 against DESI and BOSSNet at
  6500-8000 K), although part of that is the other catalogs.
* Loose ranges in logg, [Fe/H] and [alpha/M] against extreme values.
"""

from __future__ import annotations

import h5py
import numpy as np
from astropy.table import Table

from ..prep import plots
from ..prep.download import download as download_file
from ..prep.transforms import PARAM_UNC_TRANSF, unmask

KEY = "sdssv_clam"
LABEL = "SDSS-V (CLAM)"
N_PARAMS = 4
REFERENCE = "Medan et al. 2026, arXiv:2607.22822 (SDSS DR20 VAC)"
SOURCES = (
    "https://data.sdss.org/sas/dr20/vac/mwm/stellar-params/boss_clam/boss_clam_lite-1.0.0.h5",
)
FILENAME = "boss_clam_lite-1.0.0.h5"

ID_COL = "id"
PARAM_COLS = ("teff", "fe_h", "logg", "alpha_m")
UNC_COLS = ("e_teff", "e_fe_h", "e_logg", "e_alpha_m")
EXTRA_COLS = ("ra", "dec", "flag_warn")

#: Bit positions in ``clam_flags`` (for the diagnostics)
FLAG_NAMES = (
    "IN_WD", "IN_YSO", "IN_OB", "FE_H_ALPHA_SUSPECT", "BAD_SPEC_FIT",
    "LOW_MASS_SUSPECT", "FE_H_ALPHA_CORR", "TEFF_LOGG_CORR",
    "LOW_MASS_TEFF_LOGG_CORR", "EXTREME_CORR", "ALTER_CORR",
)

SNR_MIN = 15.0

TEFF_MIN = 3400.0
#: Moot while the warning cut removes everything above 6500 K; only a value below
#: that would cut anything.
TEFF_MAX = None
LOGG_RANGE = (0.0, 5.3)
FEH_RANGE = (-2.5, 0.8)
ALPHA_RANGE = (-0.35, 0.75)

#: Upper limits on the catalog's own uncertainties (Teff in K)
MAX_ERR = {"e_teff": 300.0, "e_logg": 0.4, "e_fe_h": 0.25, "e_alpha_m": 0.15}

#: Factors applied to the catalog uncertainties before the error floors
UNC_SCALES = (3.0, 3.0, 3.0, 4.0)     # Teff, [Fe/H], logg, [alpha/Fe]

#: Floors on uncertainties, added after scaling. Experience shows that the
#: uncertainties reported by the catalog are too small.
ERROR_FLOORS = (0.05, 0.05, 0.05, 0.02)

#: Regularization errors: larger for Teff and [Fe/H] above HOT_START, where
#: CLAM disagrees most with the other catalogs. Tune once it is in the model.
#: With the warning cut the hot branch only covers 6000-6500 K, since every star
#: above 6500 K is removed.
X_ERR = (0.1, 0.1, 0.1, 0.1)
HOT_START = 6000.0
HOT_SLOPE = 0.2  # added per 1000 K above HOT_START


def download(ctx) -> None:
    download_file(SOURCES[0], ctx.raw_dir / FILENAME)


def load(ctx) -> Table:
    with h5py.File(ctx.raw_dir / FILENAME, "r") as f:
        clam, meta = f["clam"], f["meta"]
        # Variances are the diagonal of the covariance in the order
        # [teff, logg, fe_h, alpha_m]; a non-positive variance is invalid.
        variance = np.einsum("nii->ni", clam["param_covariance"][:])
        err = np.sqrt(np.where(variance > 0, variance, np.nan))
        t = Table(
            {
                "id": meta["gaia_dr3_source_id"][:].astype("i8"),
                "ra": meta["ra"][:].astype("f8"),
                "dec": meta["dec"][:].astype("f8"),
                "snr": meta["snr"][:].astype("f8"),
                "telescope": np.asarray(meta["telescope"][:]).astype(str),
                "teff": clam["teff"][:],
                "logg": clam["logg"][:],
                "fe_h": clam["fe_h"][:],
                "alpha_m": clam["alpha_m"][:],
                "e_teff": err[:, 0],
                "e_logg": err[:, 1],
                "e_fe_h": err[:, 2],
                "e_alpha_m": err[:, 3],
                "rchi2": clam["rchi2"][:],
                "clam_flags": clam["clam_flags"][:],
                "flag_bad": clam["flag_bad"][:],
                "flag_warn": clam["flag_warn"][:],
            }
        )
    return t


def unc_transf(d: int):
    """Scale the catalog uncertainties before the floors are applied."""
    return [
        (lambda x, xs, f=f, s=s: f(x, s * xs))
        for f, s in zip(PARAM_UNC_TRANSF[:d], UNC_SCALES[:d])
    ]


def measured(t: Table) -> dict:
    """Which rows carry a [Fe/H] and an [alpha/M] that may be used.

    The catalog reports both for every row, but ``flag_warn`` marks the rows whose
    [alpha/M] the paper says not to trust (and [Fe/H] with it). Those are the rows
    that would make this catalog *partially* four-dimensional, which the map
    cannot invert (see the module notes), so they are removed rather than kept
    with a placeholder.
    """
    warn = np.asarray(t["flag_warn"], dtype=bool)
    feh = unmask(t["fe_h"]).astype("f8")
    alpha = unmask(t["alpha_m"]).astype("f8")
    return {
        "[Fe/H] measured (no flag_warn)": ~warn & np.isfinite(feh),
        "[alpha/M] measured (no flag_warn)": ~warn & np.isfinite(alpha),
    }


def cuts(t: Table, ctx) -> dict:
    teff = unmask(t["teff"]).astype("f8")
    logg = unmask(t["logg"])
    feh = unmask(t["fe_h"])
    alpha = unmask(t["alpha_m"])
    err = {k: unmask(t[k]).astype("f8") for k in MAX_ERR}

    def ok(values, limits):
        return (values > limits[0]) & (values < limits[1])

    def small(k):
        return np.isfinite(err[k]) & (err[k] < MAX_ERR[k])

    named = {
        "Gaia source id": np.asarray(t[ID_COL]) > 0,
        "flag_bad clear": ~np.asarray(t["flag_bad"], dtype=bool),
        **measured(t),
        f"SNR >= {SNR_MIN:g}": unmask(t["snr"]) >= SNR_MIN,
        "Teff inside the cool edge": teff > TEFF_MIN,
        "logg in range": ok(logg, LOGG_RANGE),
        # Both are required for every kept row, so a NaN (an invalid variance in
        # the catalog) fails here rather than being masked later.
        "[Fe/H] in range": ok(feh, FEH_RANGE),
        "[alpha/M] in range": ok(alpha, ALPHA_RANGE),
        "uncertainties finite and below limits": (
            small("e_teff") & small("e_logg")
            & small("e_fe_h") & small("e_alpha_m")
        ),
    }
    if TEFF_MAX is not None:
        named["Teff below the hot limit"] = teff < TEFF_MAX
    return named


def param_errs(t: Table, ctx) -> np.ndarray:
    teff = unmask(t["teff"]).astype("f8")
    out = np.tile(np.asarray(X_ERR, dtype="f8"), (len(t), 1))
    hot = np.clip(HOT_SLOPE * (teff - HOT_START) / 1000.0, 0.0, None)
    out[:, 0] += hot
    out[:, 1] += hot
    return out.astype("f4")


def diagnostics(t: Table, mask, table: Table, ctx):
    """Flags, SNR, the edges of the sample, and [Fe/H]-[alpha/M] for the warned rows."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    keep = np.asarray(mask, dtype=bool)
    has_id = np.asarray(t[ID_COL]) > 0
    warn = np.asarray(t["flag_warn"], dtype=bool)
    flags = np.asarray(t["clam_flags"]).astype("i8")
    snr = unmask(t["snr"])
    teff = unmask(t["teff"]).astype("f8")
    written = []

    # Flags (all rows vs selected) and SNR
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 3.6), layout="constrained")
    n_all = [int(np.count_nonzero((flags >> b) & 1)) for b in range(len(FLAG_NAMES))]
    n_sel = [int(np.count_nonzero(((flags >> b) & 1) & keep)) for b in range(len(FLAG_NAMES))]
    y = np.arange(len(FLAG_NAMES))
    ax0.barh(y, n_all, color="C0", label="all rows")
    ax0.barh(y, n_sel, color="C1", label="selected")
    ax0.set_yticks(y, FLAG_NAMES, fontsize=7)
    ax0.invert_yaxis()
    ax0.set_xscale("log")
    ax0.set_xlabel("stars with the flag set")
    ax0.legend(fontsize=7)
    bins = np.linspace(1.0, 2.8, 100)
    ax1.hist(np.log10(snr[has_id]), bins=bins, histtype="step", label="with Gaia id")
    ax1.hist(np.log10(snr[keep]), bins=bins, histtype="stepfilled", alpha=0.5, label="selected")
    ax1.axvline(np.log10(SNR_MIN), c="k", ls="--", lw=0.8)
    ax1.set_yscale("log")
    ax1.set_xlabel(r"$\log_{10}\,\mathrm{SNR}$")
    ax1.legend(fontsize=7)
    written.append(plots.save(fig, ctx.diagnostic_dir, "flags_snr"))

    # The edges of the sample: Teff (cool end, hot end), logg of hot stars
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.4), layout="constrained")
    base = has_id & ~np.asarray(t["flag_bad"], dtype=bool)
    for ax, (lo, hi, limit, title) in zip(
        axes[:2], ((3000, 4200, TEFF_MIN, "cool edge"), (5000, 20000, TEFF_MAX, "hot end"))
    ):
        ax.hist(teff[base], bins=np.linspace(lo, hi, 120), histtype="step", label="flag_bad clear")
        ax.hist(teff[keep], bins=np.linspace(lo, hi, 120), histtype="stepfilled", alpha=0.5, label="selected")
        if limit is not None:
            ax.axvline(limit, c="k", ls="--", lw=0.8)
        ax.set_yscale("log")
        ax.set_xlabel(f"Teff (K), {title}")
    axes[0].legend(fontsize=7)
    # Every row above 8000 K is removed by flag_warn, so this panel is drawn from
    # all rows with an id: it documents what the warning cut takes out.
    hot = has_id & (teff > 8000)
    axes[2].hist2d(
        teff[hot], unmask(t["logg"])[hot],
        bins=[np.linspace(8000, 20000, 100), np.linspace(1, 5.5, 90)], norm=LogNorm(), rasterized=True,
    )
    axes[2].set_title(f"above 8000 K, all removed: {int(hot.sum()):,}", fontsize=8)
    axes[2].set_xlabel("Teff (K); rows above 8000 K")
    axes[2].set_ylabel("logg (pinned near 4.1)")
    axes[3].hist(unmask(t["alpha_m"])[keep], bins=100, histtype="step", label="[alpha/M]")
    axes[3].hist(unmask(t["fe_h"])[keep], bins=100, histtype="step", label="[Fe/H]")
    axes[3].set_yscale("log")
    axes[3].legend(fontsize=7)
    written.append(plots.save(fig, ctx.diagnostic_dir, "sample_edges"))

    # [Fe/H]-[alpha/M] for the rows the sample keeps and the rows the warning cut
    # removes; the latter are the only ones without a usable [alpha/M].
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
    for ax, sel, title in ((axes[0], keep, "kept"), (axes[1], has_id & warn, "flag_warn set (removed)")):
        ax.hist2d(
            unmask(t["fe_h"])[sel], unmask(t["alpha_m"])[sel],
            bins=[np.linspace(-2.6, 0.9, 100), np.linspace(-0.6, 1.0, 100)], norm=LogNorm(), rasterized=True,
        )
        ax.set_title(f"{title}: {int(sel.sum()):,}", fontsize=9)
        ax.set_xlabel(r"[Fe/H]")
        ax.set_ylabel(r"[$\alpha$/M] (catalog value)")
    written.append(plots.save(fig, ctx.diagnostic_dir, "feh_alpha_by_warn"))
    return written
