#!/usr/bin/env python
"""Plot [Fe/H] vs [alpha/Fe] for one prepared catalog, with and without a
cut on the [alpha/Fe] uncertainty.

Reads a prepared table (``training_data/<key>.h5``), which stores the four
parameters in the canonical order, so column 1 is [Fe/H] and column 3 is
[alpha/Fe]. The left panel uses every star; the right one keeps
``sigma_x[3] < --sigma-cut``.

Usage:
    scripts/plot_feh_alpha.py training_data/desi_dr1.h5 --sigma-cut 0.25
"""

import argparse
from pathlib import Path

import numpy as np
from astropy.table import Table

FEH, AFE = 1, 3
PARAM_NAMES = ("5 ln(Teff/5000 K)", "[Fe/H]", "logg", "[alpha/Fe]")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", type=Path, help="prepared table (h5)")
    parser.add_argument("--sigma-cut", type=float, default=0.25,
                        help="keep stars with sigma_[alpha/Fe] below this")
    parser.add_argument("--feh-range", nargs=2, type=float, default=None)
    parser.add_argument("--afe-range", nargs=2, type=float, default=None)
    parser.add_argument("--bins", type=int, default=200)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    t = Table.read(args.path)
    x = np.asarray(t["x"], dtype="f8")
    sigma = np.asarray(t["sigma_x"], dtype="f8")
    feh, afe, safe = x[:, FEH], x[:, AFE], sigma[:, AFE]

    finite = np.isfinite(feh) & np.isfinite(afe) & np.isfinite(safe)
    keep = finite & (safe < args.sigma_cut)

    feh_range = args.feh_range or tuple(np.nanpercentile(feh[finite], [0.1, 99.9]))
    afe_range = args.afe_range or tuple(np.nanpercentile(afe[finite], [0.1, 99.9]))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained",
                             sharex=True, sharey=True)
    panels = (
        (axes[0], finite, f"all {finite.sum():,} stars"),
        (axes[1], keep,
         f"$\\sigma_{{[\\alpha/\\mathrm{{Fe}}]}} < {args.sigma_cut:g}$: "
         f"{keep.sum():,} stars ({keep.mean():.2%})"),
    )
    for ax, mask, label in panels:
        h = ax.hist2d(
            feh[mask], afe[mask],
            bins=args.bins, range=[feh_range, afe_range],
            norm=LogNorm(), rasterized=True, cmap="viridis",
        )
        ax.set_xlabel(r"$\left[\mathrm{Fe}/\mathrm{H}\right]$")
        ax.set_title(label, fontsize=10)
        # median and 16-84 band of [alpha/Fe] in [Fe/H] bins
        edges = np.linspace(*feh_range, 61)
        idx = np.digitize(feh[mask], edges) - 1
        cen, med, lo, hi = [], [], [], []
        for i in range(len(edges) - 1):
            values = afe[mask][idx == i]
            if values.size < 50:
                continue
            cen.append(0.5 * (edges[i] + edges[i + 1]))
            p16, p50, p84 = np.percentile(values, [16, 50, 84])
            med.append(p50)
            lo.append(p16)
            hi.append(p84)
        ax.plot(cen, med, c="w", lw=1.2)
        ax.plot(cen, lo, c="w", lw=0.7, ls="--")
        ax.plot(cen, hi, c="w", lw=0.7, ls="--")
        fig.colorbar(h[3], ax=ax, label="stars per bin")

    axes[0].set_ylabel(r"$\left[\alpha/\mathrm{Fe}\right]$")
    if args.out is None:
        args.out = (Path("diagnostic_plots")
                    / f"{args.path.stem}_feh_alpha_sigma{args.sigma_cut:g}.png")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print(f"wrote {args.out}")

    # a couple of numbers the figure cannot show
    removed = finite & ~keep
    if np.any(removed):
        print(f"cut removes {removed.sum():,} stars; of those "
              f"{np.percentile(feh[removed], 50):+.2f} median [Fe/H], "
              f"{np.percentile(afe[removed], 50):+.2f} median [alpha/Fe]")
    for name, mask in (("all", finite), ("kept", keep)):
        print(f"  {name:5s} [alpha/Fe] median {np.median(afe[mask]):+.3f}, "
              f"16-84 = {np.percentile(afe[mask], 16):+.3f} .. "
              f"{np.percentile(afe[mask], 84):+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
