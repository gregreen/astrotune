"""Figures used to monitor training.

A single figure summarises the state of a run: the loss history (top row) and
the catalogs before and after the learned transformation (bottom, 2x2).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence

import os

if not os.environ.get("DISPLAY"):  # headless by default
    import matplotlib

    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
from matplotlib.ticker import AutoMinorLocator

log = logging.getLogger(__name__)

PARAM_LABELS = (
    r"$5\ln\left(T_{\rm eff}/5000\,\mathrm{K}\right)$",
    r"$\left[\mathrm{Fe/H}\right]$",
    r"$\log g$",
    r"$\left[\alpha/\mathrm{Fe}\right]$",
)

_PENALTIES = (
    ("penalty_pair", "penalty (pairs)"),
    ("penalty_ctrl", "penalty (control)"),
    ("penalty_div", "penalty (div)"),
    ("penalty_l2", "penalty (l2)"),
)


def catalog_colors(n: int):
    """A fixed set of ``n`` distinct colors, used consistently in all figures."""
    from distinctipy import distinctipy

    return distinctipy.get_colors(
        n,
        exclude_colors=[(1, 1, 1), (0, 0, 0)],
        pastel_factor=0.0,
        n_attempts=1000,
        rng=1,
    )


def catalog_samples(ctrl_dict, n_cat: int, n_plot: int = 2048) -> List[np.ndarray]:
    """Random samples of each catalog, taken from the catalog control points.

    Returns a list of ``(n_i, 4)`` arrays (``n_i <= n_plot``). [alpha/Fe] is
    already padded with zeros for catalogs that do not have it.
    """
    samples = []
    for k in range(n_cat):
        idx = (ctrl_dict["c_control"] == k) & ctrl_dict["from_catalog"]
        samples.append(ctrl_dict["x_control"][idx][:n_plot])
    return samples


def catalog_sample_valid(ctrl_dict, n_cat: int, n_plot: int = 2048) -> List[np.ndarray]:
    """``(n_i, 4)`` boolean masks matching :func:`catalog_samples`: parameter measured?"""
    out = []
    for k in range(n_cat):
        idx = (ctrl_dict["c_control"] == k) & ctrl_dict["from_catalog"]
        x = ctrl_dict["x_control"][idx][:n_plot]
        if "missing_control" in ctrl_dict:
            out.append(~ctrl_dict["missing_control"][idx][:n_plot])
        else:  # hand-built dictionaries: assume everything is measured
            out.append(np.ones(x.shape, dtype=bool))
    return out


def _smooth(x, n_smooth: Optional[int] = None):
    """Smooth a time series with a Kaiser window."""
    x = np.asarray(x, dtype="f8")
    if n_smooth is None:
        n_smooth = int(np.clip(len(x) // 16, 4, 128))
    if len(x) < 2 * n_smooth:
        return x
    w = np.kaiser(2 * n_smooth, 5)
    w /= np.sum(w)
    return scipy.ndimage.convolve(x, w, mode="reflect")


def plot_loss_history(metrics, fig) -> None:
    """Draw three loss panels on ``fig``: everything, loss, and penalties."""
    ax0, ax1, ax2 = fig.subplots(1, 3)

    n = len(metrics["loss"])
    step = np.arange(n) / 1000.0
    half = n // 2
    step_half = step[half:]

    for key, label in (("loss", "loss"),) + _PENALTIES:
        alpha = 1.0 if key == "loss" else 0.7
        ax0.semilogy(step, np.maximum(metrics[key], 1e-12), label=label, alpha=alpha)
    leg = ax0.legend(fontsize=6)
    for line in leg.get_lines():
        line.set_alpha(1)

    # Second half of training so far: smoothed training/validation loss
    loss_smooth = _smooth(metrics["loss"])
    (line,) = ax1.plot(step_half, metrics["loss"][half:], alpha=0.25)
    ax1.plot(step_half, loss_smooth[half:], c=line.get_color(), label="training loss")
    if len(metrics.get("loss_val", ())) > 1:
        step_val = np.asarray(metrics["step_val"]) / 1000.0
        loss_val_smooth = _smooth(metrics["loss_val"])
        sel = step_val >= (step_half[0] if len(step_half) else 0.0)
        ax1.plot(step_val[sel], loss_val_smooth[sel], label="validation loss")
    ax1.legend(fontsize=6)

    ax2.plot([], [])  # advance the color cycle past the "loss" color
    for key, label in _PENALTIES:
        ax2.plot(step_half, _smooth(metrics[key])[half:], label=label)
    ax2.legend(fontsize=6)

    ax0.set_title("history", fontsize=8)
    ax1.set_title("loss (2nd half)", fontsize=8)
    ax2.set_title("penalties (2nd half)", fontsize=8)
    for ax in (ax0, ax1, ax2):
        ax.set_xlabel("step / 1000", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.2)
        ax.xaxis.set_minor_locator(AutoMinorLocator())


def plot_transformed_params(
    bij, samples: Sequence[np.ndarray], labels, colors, fig,
    valid: Optional[Sequence[np.ndarray]] = None,
) -> None:
    """Scatter plots of catalog samples in the original and harmonized spaces.

    ``valid`` (from :func:`catalog_sample_valid`) marks the parameters each
    catalog measures. Placeholders for the others are not plotted, in either
    space (they are still passed through the model, as zeros).

    Top row: Teff-logg. Bottom row: [Fe/H]-[alpha/Fe]. Left column: original
    coordinates. Right column: harmonized coordinates.
    """
    import tensorflow as tf

    paths = []
    for k, x in enumerate(samples):
        c = tf.fill([len(x)], tf.constant(k, dtype=tf.int32))
        path = bij.return_path(tf.constant(x, dtype=tf.float32), c, 2).numpy()
        if valid is not None:
            path = np.where(valid[k][None], path, np.nan)
        paths.append(path)

    # Plot limits from the original coordinates, with a margin
    x_all = np.concatenate([p[0] for p in paths], axis=0)
    with np.errstate(all="ignore"):
        lim = np.nanpercentile(x_all, [1.0, 99.0], axis=0)
    lim = np.where(np.isfinite(lim), lim, np.array([[-1.0], [1.0]]))  # nothing measured
    w = lim[1] - lim[0]
    lim[0] -= 0.3 * w
    lim[1] += 0.3 * w
    lim[:, 0] = lim[::-1, 0]  # Teff and logg increase toward the lower left
    lim[:, 2] = lim[::-1, 2]

    param_idx = [(0, 2), (1, 3)]
    ax_arr = fig.subplots(2, 2)
    for i, ax_row in enumerate(ax_arr):
        p0, p1 = param_idx[i]
        for j, ax in enumerate(ax_row):
            for k, label in enumerate(labels):
                x = paths[k][j]
                ax.scatter(
                    x[:, p0], x[:, p1], color=colors[k], label=label,
                    edgecolors="none", s=4, alpha=0.2,
                )
            ax.set_xlabel(PARAM_LABELS[p0], fontsize=8)
            if j == 0:
                ax.set_ylabel(PARAM_LABELS[p1], fontsize=8)
            else:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=7)
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            ax.grid(True, alpha=0.2)
            ax.grid(True, which="minor", alpha=0.08)
            ax.set_xlim(lim[:, p0])
            ax.set_ylim(lim[:, p1])

    ax_arr[0, 0].legend(loc="upper left", markerscale=3, fontsize=5)
    ax_arr[0, 0].set_title("Original", fontsize=9)
    ax_arr[0, 1].set_title("Transformed", fontsize=9)


def save_progress_figure(
    path: Path,
    metrics,
    bij,
    samples: Sequence[np.ndarray],
    labels: Sequence[str],
    colors,
    title: str = "",
    dpi: int = 120,
    valid: Optional[Sequence[np.ndarray]] = None,
) -> Path:
    """Write a figure summarizing training progress to ``path``."""
    fig = plt.figure(figsize=(8, 10), layout="constrained")
    top, bottom = fig.subfigures(2, 1, height_ratios=[1, 3.2])
    plot_loss_history(metrics, top)
    plot_transformed_params(bij, samples, labels, colors, bottom, valid)
    if title:
        fig.suptitle(title, fontsize=10)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path
