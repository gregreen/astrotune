#!/usr/bin/env python
"""The plane in 4-D ``x`` that best separates the two CLAM residual modes.

    python scripts/clam_mode_plane.py

The modes are defined exactly as in ``scripts/clam_offmode.py`` (two-means on
``f(x) - y`` in ``([Fe/H], [a/Fe])``). This asks the next question: is there a
*direction in the catalog's own four parameters* along which the two classes
separate, i.e. is the split a function of ``x`` at all?

That is Fisher's linear discriminant: with two classes the separating
hyperplane has normal

    w  ~  S_W^-1 (m_1 - m_2),

with ``S_W`` the pooled within-class covariance of ``x`` and ``m_i`` the class
means. The weights are reported twice: standardised (each parameter divided by
its overall standard deviation, so the numbers are comparable and the plane is
comparable to a Mahalanobis one) and raw, in the project's units.

Writes ``diagnostic_plots/<catalog>_selection/mode_plane.png``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

NAMES = ("Teff (5 ln T/5000)", "[Fe/H]", "log g", "[a/Fe]")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--file", type=Path, default=Path("data/sdssv_clam_subset.h5"))
    parser.add_argument("--out-dir", type=Path,
                        default=Path("diagnostic_plots/sdssv_clam_selection"))
    args = parser.parse_args(argv)

    with h5py.File(args.file, "r") as f:
        keys = str(f.attrs["catalog_keys"]).split(",")
        k = keys.index(str(f.attrs["catalog"]))
        c = np.asarray(f["obs/catalog"][:])
        star = np.asarray(f["obs/star"][:])
        x = np.asarray(f["obs/x"][:], dtype="f8")
        fx = np.asarray(f["obs/fx"][:], dtype="f8")
        y = np.asarray(f["stars/y"][:], dtype="f8")

    mine = c == k
    x = x[mine]
    d = np.c_[fx[mine, 1] - y[star[mine], 1], fx[mine, 3] - y[star[mine], 3]]
    cen = np.array([[0.0, 0.0], [-0.3, -0.2]])
    for _ in range(50):
        label = np.argmin(((d[:, None, :] - cen[None]) ** 2).sum(-1), 1)
        new = np.array([d[label == i].mean(0) for i in range(2)])
        if np.allclose(new, cen):
            break
        cen = new
    off = label == 1
    print(f"{len(x):,} rows: {(~off).sum():,} on the origin mode, {off.sum():,} off-centre")

    # --- Fisher's linear discriminant -----------------------------------------
    mean = x.mean(0)
    sd = x.std(0)
    z = (x - mean) / sd  # standardised, so Sw is a correlation structure
    m0, m1 = z[~off].mean(0), z[off].mean(0)
    Sw = 0.5 * (np.cov(z[~off].T) + np.cov(z[off].T))
    Sb = np.outer(m1 - m0, m1 - m0)
    w_std = np.linalg.solve(Sw, m1 - m0)
    w_std /= np.linalg.norm(w_std)
    if w_std[np.argmax(np.abs(w_std))] < 0:  # sign convention: biggest weight positive
        w_std = -w_std
    w_raw = w_std / sd
    w_raw /= np.abs(w_raw).max()

    proj = z @ w_std
    p0, p1 = proj[~off], proj[off]
    # Separation of the projected distributions, and how well a single cut works.
    threshold = 0.5 * (p0.mean() + p1.mean())
    pred = (proj > threshold)
    accuracy = max(np.mean(pred == off), np.mean(~pred == off))
    pooled = np.sqrt(0.5 * (p0.var() + p1.var()))
    # Mahalanobis distance between the class means in x
    dm = np.sqrt(max((m1 - m0) @ np.linalg.solve(Sw, m1 - m0), 0.0))
    # AUC (rank based), robust to a non-Gaussian projection
    order = np.argsort(proj)
    ranks = np.empty(len(proj))
    ranks[order] = np.arange(1, len(proj) + 1)
    n1, n0 = off.sum(), (~off).sum()
    auc = (ranks[off].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    auc = max(auc, 1 - auc)

    print("\nseparating hyperplane (standardised weights; largest = 1)"
          if False else "\nseparating hyperplane")
    print(f"{'parameter':22s} {'standardised':>13s} {'raw (project units)':>21s}")
    for name, a, b in zip(NAMES, w_std, w_raw):
        print(f"{name:22s} {a:13.3f} {b:13.3f}")
    print(f"\nclass means in x:      " + "  ".join(f"{v:+.3f}" for v in mean))
    print(f"x standard deviations: " + "  ".join(f"{v:.3f}" for v in sd))
    print("\nprojected onto the discriminant (standardised units):")
    print(f"  on origin  {p0.mean():+.2f} +- {p0.std():.2f}")
    print(f"  off-centre {p1.mean():+.2f} +- {p1.std():.2f}")
    print(f"  distance between the means {abs(p1.mean() - p0.mean()) / pooled:.2f} pooled sigma")
    print(f"  Mahalanobis distance between the class means in x: {dm:.3f}")
    print(f"  best single cut: accuracy {100 * accuracy:.1f}%, AUC {auc:.3f}")

    # --- figures ---------------------------------------------------------------
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 4.2), layout="constrained")
    axes = fig.subplots(1, 3, width_ratios=[1.1, 1.0, 1.2])

    axes[0].hist(p0, bins=200, histtype="step", density=True, color="C0",
                 label=f"on origin ({(~off).sum():,})")
    axes[0].hist(p1, bins=200, histtype="step", density=True, color="C1",
                 label=f"off-centre ({off.sum():,})")
    axes[0].axvline(threshold, color="k", ls="--", lw=0.8)
    axes[0].set_xlabel("projection on the discriminant")
    axes[0].set_ylabel("density")
    axes[0].set_title(f"AUC {auc:.3f}, best cut {100 * accuracy:.0f}%", fontsize=9)
    axes[0].legend(fontsize=7)

    axes[1].barh(np.arange(4), w_std, color=["C0" if v < 0 else "C1" for v in w_std])
    axes[1].set_yticks(np.arange(4))
    axes[1].set_yticklabels([n.replace(" (5 ln T/5000)", "") for n in NAMES], fontsize=8)
    axes[1].axvline(0, color="k", lw=0.6)
    axes[1].set_xlabel("standardised weight")
    axes[1].set_title("normal of the separating plane", fontsize=9)
    axes[1].grid(alpha=0.2, axis="x")

    big = np.argsort(-np.abs(w_std))[:2]
    if big[1] == 1:  # [Fe/H] belongs on the x axis
        big = big[::-1]
    ax = axes[2]
    for subset, colour, lab in ((~off, "C0", "on origin"), (off, "C1", "off-centre")):
        ax.scatter(x[subset, big[0]], x[subset, big[1]], s=0.5, alpha=0.05,
                   c=colour, rasterized=True, label=lab)
    ax.set_xlabel(NAMES[big[0]])
    ax.set_ylabel(NAMES[big[1]])
    ax.set_title("two largest weights", fontsize=9)
    ax.legend(fontsize=7, markerscale=12)
    ax.set_xlim(*np.percentile(x[:, big[0]], [0.5, 99.5]))
    ax.set_ylim(*np.percentile(x[:, big[1]], [0.5, 99.5]))
    if big[0] == 0:  # Teff: log ticks in kK
        ax.set_xscale("log")
        ax.set_xticks([3500, 4000, 5000, 6000, 8000])
        ax.set_xticklabels(["3.5", "4", "5", "6", "8"])
        ax.set_xlabel(r"$T_\mathrm{eff}$ / 1000 K")
    fig.suptitle("CLAM residual modes: is there a plane in $x$ that separates them?")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "mode_plane.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
