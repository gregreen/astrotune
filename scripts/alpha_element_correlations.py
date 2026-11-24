#!/usr/bin/env python
"""How well does each alpha element track the catalogue's [alpha/Fe]?

Runs one catalogue's ``prepare_columns`` (so the element columns are in the same
units and flag handling the pipeline uses) and reports

* per element: the correlation with [alpha/Fe], the slope of [alpha/Fe] against
  it, and the median offset -- a slope well below 1 or an offset of a tenth of a
  dex means the element is not measuring the same thing as the others;
* the correlation restricted to the stars the catalogue calls precise, which
  separates a definitional mismatch (correlation stays low) from plain noise
  (correlation rises);
* the element-element correlations plus a variance decomposition: the covariance
  between two different elements is the component they share, so what is left of
  each element's own variance is its individual noise.

A catalog can be passed as a registry key. The crossmatch cache is reused, so
after a prepare run this takes a couple of minutes rather than an hour.

Usage:
    scripts/alpha_element_correlations.py lamost_ddp_dr9
"""

import argparse

import numpy as np

from astrotune import registry
from astrotune.prep.pipeline import make_context

PRECISE = 0.03  # [alpha/Fe] uncertainty defining the "precise" subsample


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("catalog", help="registry key, e.g. lamost_ddp_dr9")
    parser.add_argument("--precise", type=float, default=PRECISE,
                        help="[alpha/Fe] uncertainty below which a star is 'precise'")
    parser.add_argument("--alpha-key", default=None,
                        help="column holding the catalogue's own [alpha/Fe] "
                             "(default: ALPHA_FE, or the alpha_fe that prepare_columns adds)")
    args = parser.parse_args(argv)

    module = registry.get_module(args.catalog)
    ctx = make_context(module.KEY, make_plots=False, force=False,
                       allow_crossmatch_regression=False, gaia_archive="esa")
    t = module.prepare_columns(module.load(ctx), ctx) if hasattr(module, "prepare_columns") else module.load(ctx)

    abundances = ctx.extra.get("abundances")
    if abundances is None:
        parser.error(f"{args.catalog} does not keep per-element abundances")

    alpha_key = args.alpha_key
    if alpha_key is None:
        alpha_key = next(k for k in ("ALPHA_FE", "alpha_fe") if k in t.colnames)
    alpha = np.asarray(t[alpha_key], dtype="f8")
    aerr_key = "ALPHA_FE_ERR" if "ALPHA_FE_ERR" in t.colnames else "alpha_fe_err"
    aerr = np.asarray(t[aerr_key], dtype="f8")

    names = list(abundances.colnames)
    Y = np.column_stack([np.asarray(abundances[c], dtype="f8") for c in names])
    finite_alpha = np.isfinite(alpha)

    print(f"{args.catalog}: {len(t):,} rows, elements {names}")
    print(f"[alpha/Fe] from {alpha_key}: median {np.nanmedian(alpha):+.3f}, "
          f"sd {np.nanstd(alpha):.3f}, over {finite_alpha.sum():,} stars")

    print(f"\n{'element':8s}{'n':>11s}{'r':>8s}{'slope':>8s}{'median':>9s}"
          f"{'offset':>9s}{'sd':>8s}{f'r (err<{args.precise:g})':>14s}")
    for i, name in enumerate(names):
        ok = np.isfinite(Y[:, i]) & finite_alpha
        precise = ok & (aerr < args.precise)
        r = np.corrcoef(Y[ok, i], alpha[ok])[0, 1]
        r_precise = np.corrcoef(Y[precise, i], alpha[precise])[0, 1]
        slope = np.polyfit(Y[ok, i], alpha[ok], 1)[0]
        print(f"{name:8s}{ok.sum():11,d}{r:8.3f}{slope:8.3f}"
              f"{np.nanmedian(Y[ok, i]):9.3f}"
              f"{np.nanmedian(alpha[ok] - Y[ok, i]):9.3f}"
              f"{np.nanstd(Y[ok, i]):8.3f}{r_precise:14.3f}")

    k = len(names)
    cov = np.empty((k, k))
    for i in range(k):
        for j in range(i, k):
            ok = np.isfinite(Y[:, i]) & np.isfinite(Y[:, j])
            cov[i, j] = cov[j, i] = np.cov(Y[ok, i], Y[ok, j])[0, 1]

    print("\nelement-element correlation")
    print(f"  {'':8s}" + "".join(f"{n:>8s}" for n in names))
    for i, name in enumerate(names):
        print(f"  {name:8s}" + "".join(
            f"{cov[i, j] / np.sqrt(cov[i, i] * cov[j, j]):8.3f}" for j in range(k)))

    shared = np.median(cov[~np.eye(k, dtype=bool)])
    print(f"\nshared component: sd {np.sqrt(shared):.3f} (covariance between two "
          f"different elements)")
    print(f"  {'element':8s}{'own sd':>9s}{'noise sd':>10s}")
    for i, name in enumerate(names):
        noise = max(cov[i, i] - shared, 0.0)
        print(f"  {name:8s}{np.sqrt(cov[i, i]):9.3f}{np.sqrt(noise):10.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
