#!/usr/bin/env python
"""What did a continuation change? Compare two runs' maps on the same stars.

The star table of the *old* run is used as the carrier of the inputs (ids, x, the
quoted sigma, the masks) -- those come from the prepared catalog tables and are
identical for both runs -- so every difference reported here is a difference
between the two bijectors and their fitted per-star values.

Reported: the change in each catalog's harmonized values, and, for the focus
catalog and parameter, the pair pull (the sum over the stars it shares with each
partner of the Student-t likelihood's derivative with respect to its value) plus
its slope against the star's latent value. Both are the quantities that were
measured for the original run, so the two can be compared directly.

Usage:
    scripts/bijector_delta.py --old models/runA --new models/runA_cont32 \\
        --delta-dir models/run1/delta --focus sdssv_mdc --param 3
"""

import argparse
from pathlib import Path

import numpy as np

PARAMS = ("Teff", "[Fe/H]", "logg", "[a/Fe]")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--delta-dir", type=Path, default=None,
                        help="the delta model that supplied the sigmas and nu")
    parser.add_argument("--focus", default="sdssv_mdc")
    parser.add_argument("--param", type=int, default=3, choices=range(4))
    args = parser.parse_args(argv)

    from astrotune.apply import make_forward_and_jacobian
    from astrotune.delta.stars import load_star_table, with_groups
    from astrotune.model import load_bijector
    from astrotune.model.io import enable_memory_growth

    enable_memory_growth()
    p, dname = args.param, PARAMS[args.param]
    stars, keys = load_star_table(args.old / "delta" / "star_table.npz")
    keys = tuple(keys)
    stars = with_groups(stars, keys)
    fwd = {name: make_forward_and_jacobian(load_bijector(d))
           for name, d in (("old", args.old), ("new", args.new))}

    y = {}
    for name, f in fwd.items():
        out = []
        for lo in range(0, len(stars["id"]), 200_000):
            sl = slice(lo, min(lo + 200_000, len(stars["id"])))
            out.append(np.asarray(f(stars["x"][sl], stars["c"][sl])[0]))
        y[name] = np.concatenate(out)

    print(f"{args.new} against {args.old}: change in the harmonized values "
          f"({dname} unless said otherwise)")
    print(f"{'catalog':22s}{'n':>9s}{'median dy_' + dname:>16s}{'median |dy|':>13s}"
          f"{'median |dy| all params':>24s}")
    for j, key in enumerate(keys):
        sel = stars["c"] == j
        if sel.sum() < 200:
            continue
        d = y["new"][sel] - y["old"][sel]
        m = stars["m"][sel]
        print(f"{key:22s}{int(sel.sum()):9,d}{np.median(d[m[:, p], p]):+16.5f}"
              f"{np.median(np.abs(d[m[:, p], p])):13.5f}"
              f"{np.median(np.abs(d[m])):24.5f}")

    if args.delta_dir is None:
        return 0

    # The pull, exactly as measured for the original run, but on the new map.
    import json

    from astrotune import registry
    from astrotune.delta.model import load_delta_model, model_nu, numpy_delta_fn

    model = load_delta_model(args.delta_dir)
    fn = numpy_delta_fn(model)
    ln_scale, ln_add = fn(stars["x"], stars["ls"], stars["g"])
    sig = np.sqrt(np.exp(ln_scale.astype("f8")) * stars["sig"].astype("f8") ** 2
                  + np.exp(ln_add.astype("f8")))
    nu_group = model_nu(model)
    gidx = registry.group_index(keys)
    nu = np.asarray([nu_group[gi] for gi in gidx])
    f = keys.index(args.focus)
    with np.load(args.new / "latent_y.npz") as z:
        Y = z["y"][np.searchsorted(z["id"], stars["id"])].astype("f8")

    n_star = len(stars["id"])
    pres = np.zeros((n_star, len(keys)), dtype=bool)
    row = np.full((len(keys), n_star), -1, dtype="i8")
    sel = np.flatnonzero(stars["m"][:, p])
    pres[stars["seg"][sel], stars["c"][sel]] = True
    row[stars["c"][sel], stars["seg"][sel]] = sel

    def pull(name, rows_idx, star_idx):
        yy = y[name][rows_idx].astype("f8")
        m = stars["m"][rows_idx]
        d = m.sum(axis=1).astype("f8")
        delta = np.where(m, yy - Y[star_idx], 0.0)
        chi2 = np.sum(np.where(m, delta ** 2 / sig[rows_idx].astype("f8") ** 2, 0.0), axis=1)
        w = (nu[f] + d) / (nu[f] + chi2)
        g = 2.0 * w * delta[:, p] / sig[rows_idx][:, p].astype("f8") ** 2
        h = 2.0 * w * (1.0 / sig[rows_idx][:, p].astype("f8") ** 2
                       - 2.0 * (delta[:, p] / sig[rows_idx][:, p].astype("f8") ** 2) ** 2
                       / (nu[f] + chi2))
        return g, h

    print(f"\npull on {args.focus} {dname}, dNLL/d eps over the pairs with each partner")
    print(f"{'partner':22s}{'n':>9s}{'dNLL/d eps (old)':>18s}{'dNLL/d eps (new)':>18s}"
          f"{'implied shift (new)':>21s}")
    scale = float(np.median(np.abs(np.asarray(fwd["new"](stars["x"][:1000], stars["c"][:1000])[1])[:, p, p]))) or 1.0
    have_f = pres[:, f]
    for k in range(len(keys)):
        if k == f:
            continue
        pair = have_f & pres[:, k]
        if pair.sum() < 20:
            continue
        g_old, _ = pull("old", row[f][pair], pair)
        g_new, h_new = pull("new", row[f][pair], pair)
        implied = -g_new.sum() / h_new.sum() / scale if h_new.sum() > 0 else np.nan
        print(f"{keys[k]:22s}{int(pair.sum()):9,d}{g_old.sum():18.1f}"
              f"{g_new.sum():18.1f}{implied:+21.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
