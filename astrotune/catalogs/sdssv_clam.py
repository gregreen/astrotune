"""SDSS-V BOSS-CLAM stellar parameters (DR20 value-added catalog).

References
----------
Medan, I., Casey, A. R., Ji, A. P., et al. 2026, "BOSS-CLAM: Utilizing a
Constrained Linear Absorption Model to Infer Stellar Parameters from BOSS
Spectra", arXiv:2607.22822
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
  **They are too small**: the paper's own wide-binary test gives
  sigma_[Fe/H] ~ 0.15 and sigma_[alpha/M] ~ 0.06 at SNR = 10, 3-4 times the
  catalog values (0.04 and 0.019 at SNR 10-15), and the scatter against
  APOGEE/GALAH/ASPCAP at SNR 10-30 is 5-8 times the quoted error (3-5 times once
  the systematic floor measured at high SNR is subtracted). They are therefore
  scaled by :data:`UNC_SCALES` before the error floors are added (this keeps the
  SNR dependence). ``astrotune-error-scale sdssv_clam`` fits the scale and a
  systematic plateau to the scatter against APOGEE, ASPCAP and GALAH; compared in the
  harmonized space (``--run-dir``) the scale is about 3 for Teff, [Fe/H] and logg and
  4-4.5 for [alpha/M]. The project owner chose a scale of 3 for all parameters and the
  default error floors (2026-11-09): the plateau in the harmonized space differs between
  partners (0.06-0.07 against APOGEE and ASPCAP, none against GALAH).
* **Flags.** ``flag_bad`` (the catalog's own bad-fit flags, 43% of the rows)
  removes the star. ``flag_warn`` (``IN_OB`` or ``FE_H_ALPHA_SUSPECT``: every
  star above 6500 K, plus stars of the OB target program at any Teff) means that
  [alpha/M] should not be trusted: those stars are kept
  but their [alpha/M] is treated as *not measured* (``x = 0``,
  ``sigma_x = x_err = 1e9``), exactly like a missing parameter in a catalog
  without [alpha/Fe]. The paper says [Fe/H] is not trustworthy for these stars
  either; set :data:`WARN_IGNORES_FEH` to drop it as well (default: keep it, at
  the project owner's request).

Cuts (chosen by looking at the Kiel and [Fe/H]-[alpha/M] diagrams and at the
stars shared with APOGEE, GALAH, ASPCAP and the cool-star catalogs):

* ``flag_bad`` clear and SNR >= 15. The catalog already requires SNR >= 10. The
  scatter against the references is flat above SNR ~ 30 and grows to ~1.7x at
  SNR 10-20; 20 would remove a third of the cool DESI anchors.
* Finite, positive-variance uncertainties and loose upper limits on them.
* Teff > 3400 K, a cut just inside the cool edge of the sample. Below ~3500 K
  the distribution drops steeply and the agreement with SLAM/MDC/DESI degrades
  toward it (the [alpha/M] offset grows from ~0 at 3600-3700 K to +0.11 below
  3400 K). That trend is measured in bins of CLAM's own noisy Teff, so part of
  it is an edge/selection effect and the cut is a judgment call; 3400 K rather
  than 3500 K keeps ~530 SLAM anchors (3400-3500 K) and removes only the sparse
  tail (33 shared stars).
* **No upper Teff cut**, at the project owner's request: ``flag_bad`` and
  ``flag_warn`` handle the hot stars (every star above 6500 K has
  ``flag_warn``, so its [alpha/M] is ignored). Be aware that above ~8000 K logg
  is pinned near 4.1 (horizontal stripes in the Kiel diagram, see
  ``sample_edges.png``) and the Teff and [Fe/H] offsets against other catalogs
  are large (e.g. Teff +7-15%, [Fe/H] +0.6 against DESI and BOSSNet at
  6500-8000 K), although part of that is the other catalogs. Those stars are
  ~1% of the catalog (1,881 shared with BOSSNet, 476 with LAMOST HotPayne). If
  they harm the hot end, restore an upper limit in :data:`TEFF_MAX`.
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
TEFF_MAX = None  # no upper cut; set e.g. 8000.0 to restore one
LOGG_RANGE = (0.0, 5.3)
FEH_RANGE = (-2.5, 0.8)
ALPHA_RANGE = (-0.35, 0.75)

#: Upper limits on the catalog's own uncertainties (Teff in K)
MAX_ERR = {"e_teff": 300.0, "e_logg": 0.4, "e_fe_h": 0.25, "e_alpha_m": 0.15}

#: Factors applied to the catalog uncertainties before the error floors
UNC_SCALES = (3.0, 3.0, 3.0, 3.0)

#: Regularization errors: a first guess, as for the other BOSS-based catalogs
#: (SLAM 0.2, BOSSNet 0.3), larger for Teff and [Fe/H] above HOT_START, where
#: CLAM disagrees most with the other catalogs. Tune once it is in the model.
X_ERR = (0.2, 0.2, 0.2, 0.2)
HOT_START = 6000.0
HOT_SLOPE = 0.2  # added per 1000 K above HOT_START

#: Also treat [Fe/H] as not measured for ``flag_warn`` stars (see the notes)
WARN_IGNORES_FEH = False

MISSING = 1e9
FEH_INDEX, ALPHA_INDEX = 1, 3


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


def _used(t: Table) -> tuple[np.ndarray, np.ndarray]:
    """Which stars have a usable [Fe/H] and [alpha/M]."""
    warn = np.asarray(t["flag_warn"], dtype=bool)
    return (~warn if WARN_IGNORES_FEH else np.ones(len(t), bool)), ~warn


def cuts(t: Table, ctx) -> dict:
    teff = unmask(t["teff"]).astype("f8")
    logg = unmask(t["logg"])
    feh = unmask(t["fe_h"])
    alpha = unmask(t["alpha_m"])
    err = {k: unmask(t[k]).astype("f8") for k in MAX_ERR}
    feh_used, alpha_used = _used(t)

    def ok(values, limits):
        return (values > limits[0]) & (values < limits[1])

    def small(k):
        return np.isfinite(err[k]) & (err[k] < MAX_ERR[k])

    named = {
        "Gaia source id": np.asarray(t[ID_COL]) > 0,
        "flag_bad clear": ~np.asarray(t["flag_bad"], dtype=bool),
        f"SNR >= {SNR_MIN:g}": unmask(t["snr"]) >= SNR_MIN,
        "Teff inside the cool edge": teff > TEFF_MIN,
        "logg in range": ok(logg, LOGG_RANGE),
        # Parameters that are ignored for a star cannot reject it
        "[Fe/H] in range": ok(feh, FEH_RANGE) | ~feh_used,
        "[alpha/M] in range": ok(alpha, ALPHA_RANGE) | ~alpha_used,
        "uncertainties finite and below limits": (
            small("e_teff") & small("e_logg")
            & (small("e_fe_h") | ~feh_used)
            & (small("e_alpha_m") | ~alpha_used)
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


def finalize(table: Table, ctx) -> Table:
    """Mark the parameters that are not to be used for ``flag_warn`` stars as missing."""
    warn = np.asarray(table["flag_warn"], dtype=bool)
    missing = [ALPHA_INDEX] + ([FEH_INDEX] if WARN_IGNORES_FEH else [])
    for i in missing:
        table["x"][warn, i] = 0.0
        table["sigma_x"][warn, i] = MISSING
        table["x_err"][warn, i] = MISSING
    return table


def diagnostics(t: Table, mask, table: Table, ctx):
    """Flags, SNR, the edges of the sample, and [Fe/H]-[alpha/M] for warned stars."""
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
    hot = keep & (teff > 8000)
    axes[2].hist2d(
        teff[hot], unmask(t["logg"])[hot],
        bins=[np.linspace(8000, 20000, 100), np.linspace(1, 5.5, 90)], norm=LogNorm(), rasterized=True,
    )
    axes[2].set_xlabel("Teff (K); selected stars above 8000 K")
    axes[2].set_ylabel("logg (pinned near 4.1)")
    axes[3].hist(unmask(t["alpha_m"])[keep & ~warn], bins=100, histtype="step", label="[alpha/M]")
    axes[3].hist(unmask(t["fe_h"])[keep], bins=100, histtype="step", label="[Fe/H]")
    axes[3].set_yscale("log")
    axes[3].legend(fontsize=7)
    written.append(plots.save(fig, ctx.diagnostic_dir, "sample_edges"))

    # [Fe/H]-[alpha/M] for stars with and without the warning flag
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
    for ax, sel, title in ((axes[0], keep & ~warn, "flag_warn clear"), (axes[1], keep & warn, "flag_warn set ([alpha/M] ignored)")):
        ax.hist2d(
            unmask(t["fe_h"])[sel], unmask(t["alpha_m"])[sel],
            bins=[np.linspace(-2.6, 0.9, 100), np.linspace(-0.6, 1.0, 100)], norm=LogNorm(), rasterized=True,
        )
        ax.set_title(f"{title}: {int(sel.sum()):,}", fontsize=9)
        ax.set_xlabel(r"[Fe/H]")
        ax.set_ylabel(r"[$\alpha$/M] (catalog value)")
    written.append(plots.save(fig, ctx.diagnostic_dir, "feh_alpha_by_warn"))
    return written
