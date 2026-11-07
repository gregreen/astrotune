"""Diagnostic figures for catalog preparation.

Figures are always written to disk (``diagnostic_plots/<catalog>/`` by
default) so that the preparation step is fully reproducible in a headless
environment.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import numpy as np

if not os.environ.get("DISPLAY"):  # headless by default
    import matplotlib

    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.colors import LogNorm
from matplotlib.ticker import AutoMinorLocator, FixedLocator

from ..config import PARAM_NAMES
from .transforms import TEFF_REF_K, TEFF_SCALE

COLOR_CYCLE = (
    "#019422",
    "#ff5fd9",
    "#3cdeb0",
    "#da005a",
    "#9f9300",
    "#0147c7",
    "#f65135",
    "#b587ff",
    "#784702",
    "#6d4081",
)

PARAM_LATEX = (
    r"$5\ln\left(T_{\mathrm{eff}}/5000\,\mathrm{K}\right)$",
    r"$\left[\mathrm{Fe/H}\right]$",
    r"$\log g$",
    r"$\left[\alpha/\mathrm{Fe}\right]$",
)


def setup_style() -> None:
    """Consistent, publication-friendly defaults."""
    plt.rc("axes", prop_cycle=cycler(color=list(COLOR_CYCLE)))
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 150,
            "axes.grid": True,
            "grid.alpha": 0.2,
            "axes.titlesize": "medium",
            "legend.fontsize": "small",
        }
    )


def save(fig, outdir: Path, name: str) -> Path:
    """Save ``fig`` into ``outdir`` and close it."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# --- Generic helpers --------------------------------------------------------
def choose_lim(x, pct=(1.0, 99.0), expand=0.2):
    x = np.asarray(x, dtype="f8")
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return (-1.0, 1.0)
    x0, x1 = np.percentile(finite, pct)
    w = x1 - x0
    if w == 0:
        w = max(abs(x1), 1.0) * 0.1
    return x0 - expand * w, x1 + expand * w


def hist2d_reduce(x, y, c, xlim=None, ylim=None, ax=None, hist_kw=None, imshow_kw=None):
    """2-D histogram of ``x``, ``y`` where each bin shows a statistic of ``c``."""
    from scipy.stats import binned_statistic_2d

    if ax is None:
        ax = plt.gca()

    x = np.asarray(x, dtype="f8")
    y = np.asarray(y, dtype="f8")
    c = np.asarray(c, dtype="f8")
    idx = np.isfinite(x) & np.isfinite(y) & np.isfinite(c)

    xlim = choose_lim(x) if xlim is None else xlim
    ylim = choose_lim(y) if ylim is None else ylim
    xlim_s, ylim_s = np.sort(xlim).tolist(), np.sort(ylim).tolist()

    kw = dict(range=[xlim_s, ylim_s], statistic="median", bins=100)
    kw.update(hist_kw or {})
    img, _, _, _ = binned_statistic_2d(x[idx], y[idx], c[idx], **kw)

    kw = dict(extent=xlim_s + ylim_s, origin="lower", aspect="auto", interpolation="nearest")
    kw.update(imshow_kw or {})
    return ax.imshow(img.T, **kw)


def nearly_square_grid(n_elements: int):
    if n_elements <= 0:
        return (0, 0)
    cols = int(np.ceil(np.sqrt(n_elements)))
    rows = int(np.ceil(n_elements / cols))
    return rows, cols


def _style_kiel_axis(ax, xlabel=None, ylabel=None):
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xlabel(PARAM_LATEX[0] if xlabel is None else xlabel)
    ax.set_ylabel(PARAM_LATEX[2] if ylabel is None else ylabel)
    ax.grid(True, alpha=0.2)
    _set_teff_ticks(ax)


def _set_teff_ticks(ax):
    teff_ticks = np.array(list(range(2, 10, 1)) + list(range(10, 51, 10)))
    ax.xaxis.set_major_locator(FixedLocator(TEFF_SCALE * np.log(teff_ticks / (TEFF_REF_K / 1000.0))))
    ax.set_xticklabels([f"${t:.0f}$" for t in teff_ticks])
    teff_minor = np.hstack([np.arange(2, 10, 0.2), np.arange(10, 51, 1)])
    ax.xaxis.set_minor_locator(
        FixedLocator(TEFF_SCALE * np.log(teff_minor / (TEFF_REF_K / 1000.0)))
    )
    ax.set_xlabel(r"$T_\mathrm{eff} / 1000\,\mathrm{K}$")


# --- Kiel / abundance planes ------------------------------------------------
def hist2d_x_logg(x_teff, logg, ax=None):
    """Kiel diagram in transformed temperature coordinates."""
    if ax is None:
        ax = plt.gca()
    ax.hist2d(x_teff, logg, bins=100, norm=LogNorm(), rasterized=True)
    _style_kiel_axis(ax)
    return ax


def hist2d_teff_logg(teff, logg, ax=None):
    """Kiel diagram from physical temperatures."""
    return hist2d_x_logg(TEFF_SCALE * np.log(np.asarray(teff) / TEFF_REF_K), logg, ax=ax)


def hist2d_teff_feh(teff, feh, ax=None):
    if ax is None:
        ax = plt.gca()
    ax.hist2d(
        TEFF_SCALE * np.log(np.asarray(teff) / TEFF_REF_K),
        feh,
        bins=100,
        norm=LogNorm(),
        rasterized=True,
    )
    _style_kiel_axis(ax, ylabel=PARAM_LATEX[1])
    return ax


def plot_teff_logg_feh(teff, logg, feh, outdir=None, name="teff_logg_feh", label_as_mh=False):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), layout="tight")
    hist2d_teff_logg(teff, logg, ax=axes[0])
    hist2d_teff_feh(teff, feh, ax=axes[1])
    if label_as_mh:
        axes[1].set_ylabel(r"$\left[\mathrm{M/H}\right]$")
    return (save(fig, outdir, name) if outdir is not None else fig)


def plot_teff_logg_feh_abundance(
    teff,
    logg,
    feh,
    abundance,
    outdir=None,
    name="teff_logg_feh_alpha",
    abundance_label=r"$\left[\alpha/\mathrm{Fe}\right]$",
    label_as_mh=False,
):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), layout="tight")
    hist2d_teff_logg(teff, logg, ax=axes[0])

    idx = np.isfinite(abundance) & np.isfinite(feh)
    axes[1].hist2d(feh[idx], abundance[idx], bins=100, norm=LogNorm(), rasterized=True)
    axes[1].set_xlabel(r"$\left[\mathrm{M/H}\right]$" if label_as_mh else PARAM_LATEX[1])
    axes[1].set_ylabel(abundance_label)
    return (save(fig, outdir, name) if outdir is not None else fig)


def plot_alpha_abundances(
    alpha_Fe, abundances, outdir=None, name="alpha_abundances", xlabel=None
):
    """Compare the mean [alpha/Fe] with each individual alpha element."""
    n_rows, n_cols = nearly_square_grid(len(abundances.colnames))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 2.5 * n_rows), layout="tight")
    axes = np.atleast_1d(axes).ravel()

    for element, ax in zip(abundances.colnames, axes):
        x = np.asarray(alpha_Fe, dtype="f8")
        y = np.asarray(abundances[element], dtype="f8")
        good = np.isfinite(x) & np.isfinite(y)
        ax.hist2d(x[good], y[good], bins=100, norm=LogNorm(), rasterized=True)
        ax.axline((0, 0), slope=1, c="w", alpha=0.3)
        ax.set_xlabel(xlabel or PARAM_LATEX[3])
        ax.set_ylabel(rf"$\left[\mathrm{{{element}/Fe}}\right]$")

    for ax in axes[len(abundances.colnames):]:
        ax.axis("off")

    return (save(fig, outdir, name) if outdir is not None else fig)


# --- Table-level diagnostics ------------------------------------------------
def plot_catalog_params(table, outdir, names: Sequence[str] = PARAM_NAMES):
    """Histogram of every parameter column."""
    x = np.asarray(table["x"], dtype="f8")
    n = x.shape[1]
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 2.6), layout="constrained")

    for i, ax in enumerate(np.atleast_1d(axes)):
        values = x[:, i]
        lo, hi = choose_lim(values, pct=(0.1, 99.9), expand=0.1)
        ax.hist(values, bins=100, range=(lo, hi), histtype="step", log=True)
        ax.set_xlabel(PARAM_LATEX[i] if i < len(PARAM_LATEX) else names[i])
        ax.set_ylabel("count")
        ax.xaxis.set_minor_locator(AutoMinorLocator())

    fig.suptitle(f"{len(table):,} stars")
    return save(fig, outdir, "params")


def plot_catalog_uncertainties(table, outdir, names: Sequence[str] = PARAM_NAMES):
    """Uncertainties as a function of position in the Kiel diagram, plus histograms."""
    x = np.asarray(table["x"], dtype="f8")
    sigma = np.asarray(table["sigma_x"], dtype="f8")
    n = sigma.shape[1]

    xlim = choose_lim(x[:, 0], expand=0.35)
    ylim = choose_lim(x[:, 2], expand=0.35) if n > 2 else choose_lim(x[:, 1], expand=0.35)
    y_index = 2 if n > 2 else 1

    fig, axes = plt.subplots(
        1,
        n + 1,
        figsize=(2.5 * n + 0.7, 2.9),
        width_ratios=[1] * n + [0.06],
        layout="constrained",
    )
    norm = LogNorm(vmin=2e-2, vmax=5e-1)
    im = None
    for i, ax in enumerate(np.atleast_1d(axes)[:-1]):
        im = hist2d_reduce(
            x[:, 0],
            x[:, y_index],
            sigma[:, i],
            xlim=xlim,
            ylim=ylim,
            ax=ax,
            imshow_kw=dict(norm=norm),
        )
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.set_title(f"param {i}")
        ax.set_xlabel(PARAM_LATEX[0])
        if i == 0:
            ax.set_ylabel(PARAM_LATEX[y_index])
        else:
            ax.set_yticklabels([])

    if im is not None:
        fig.colorbar(im, cax=np.atleast_1d(axes)[-1], label=r"$\sigma$")
    save(fig, outdir, "uncertainties_vs_params")

    fig, ax = plt.subplots(figsize=(5, 3), layout="constrained")
    for i in range(n):
        values = sigma[:, i]
        values = values[np.isfinite(values) & (values > 0)]
        ax.hist(np.log10(values), bins=100, range=(-3, 0), histtype="step", label=f"param {i}")
    ax.set_xlabel(r"$\log_{10}(\mathrm{uncertainty})$")
    ax.legend()
    return save(fig, outdir, "uncertainties_hist")


def plot_kiel(table, outdir, name="kiel"):
    x = np.asarray(table["x"], dtype="f8")
    fig, ax = plt.subplots(figsize=(5, 4), layout="constrained")
    hist2d_x_logg(x[:, 0], x[:, 2] if x.shape[1] > 2 else np.zeros(len(x)), ax=ax)
    ax.set_title(f"{len(table):,} stars")
    return save(fig, outdir, name)


def plot_teff_feh(table, outdir, name="teff_feh"):
    """Teff-[Fe/H] plane, for catalogs that do not measure logg."""
    x = np.asarray(table["x"], dtype="f8")
    fig, ax = plt.subplots(figsize=(5, 4), layout="constrained")
    ax.hist2d(x[:, 0], x[:, 1], bins=100, norm=LogNorm(), rasterized=True)
    ax.invert_xaxis()
    _set_teff_ticks(ax)
    ax.set_ylabel(PARAM_LATEX[1])
    ax.grid(True, alpha=0.2)
    ax.set_title(f"{len(table):,} stars")
    return save(fig, outdir, name)


def plot_feh_alpha(table, outdir, name="feh_alpha"):
    x = np.asarray(table["x"], dtype="f8")
    if x.shape[1] < 4:
        return None
    good = np.isfinite(x[:, 1]) & np.isfinite(x[:, 3])
    fig, ax = plt.subplots(figsize=(4.5, 4), layout="constrained")
    ax.hist2d(x[good, 1], x[good, 3], bins=100, norm=LogNorm(), rasterized=True)
    ax.set_xlabel(PARAM_LATEX[1])
    ax.set_ylabel(PARAM_LATEX[3])
    return save(fig, outdir, name)


def plot_sky(table, outdir, name="sky"):
    if "ra" not in table.colnames or "dec" not in table.colnames:
        return None
    ra = np.asarray(table["ra"], dtype="f8")
    dec = np.asarray(table["dec"], dtype="f8")
    good = np.isfinite(ra) & np.isfinite(dec)
    if not np.any(good):
        return None

    fig = plt.figure(figsize=(5.5, 3.2), layout="constrained")
    ax = fig.add_subplot(111, projection="mollweide")
    ra_rad = np.deg2rad((ra[good] + 180.0) % 360.0 - 180.0)
    ax.scatter(ra_rad, np.deg2rad(dec[good]), s=0.5, alpha=0.2, edgecolors="none")
    ax.grid(True, alpha=0.3)
    return save(fig, outdir, name)


def plot_cut_flow(counts: dict, outdir, name="cut_flow"):
    """Horizontal bar chart of the number of stars surviving each cut."""
    labels = list(counts)
    values = [counts[label] for label in labels]

    fig, ax = plt.subplots(figsize=(5.5, 0.35 * len(labels) + 1.2), layout="constrained")
    y = np.arange(len(labels))
    ax.barh(y, values, color=COLOR_CYCLE[0], alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("stars")
    for yi, value in zip(y, values):
        ax.text(value, yi, f" {value:,}", va="center", fontsize="small")
    return save(fig, outdir, name)
