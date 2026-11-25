#!/usr/bin/env python
"""Why is one catalog's parameter being pulled? Start from the pairs.

Given a trained run and the delta model that calibrated the sigmas it was fitted
with, this looks at every star that ``--focus`` observes together with another
catalog, and reports three things:

1. how many such pairs exist per catalog;
2. the distribution of the *recalibrated* sigma of the other catalog on those
   pairs (and, dashed, of ``--focus`` on the same stars) -- who is precise and who
   is not is what decides who moves whom;
3. the pull each catalog exerts on ``--focus``'s value for that parameter, as

       dNLL/d eps = sum over the pairs of d rho_i / d(shift of focus's value by eps)

   i.e. the *sum*, not the mean, of the Student-t likelihood's derivative with
   respect to the focus catalog's observation. Positive means the likelihood rises
   as the focus value is raised, so that set of pairs is pulling it down.

   Why the partial derivative is the total one: in the latent fit the star's value
   ``Y`` is a free parameter, stationary at the solution, so by the envelope
   theorem shifting an observation changes the loss at first order only through the
   term it appears in. The other catalogs enter only through the ``Y`` they helped
   set.

The likelihood used is the one the latent fit itself minimised -- independent
observations with the calibrated sigma and the per-catalog ``nu`` the delta model
supplied -- not the delta model's own hierarchical star likelihood.

Usage:
    scripts/mdc_alpha_pull.py --run-dir models/latent_student_t_iter2_20cat \\
        --delta-dir models/latent_student_t_lr1e4_20cat/delta \\
        --focus sdssv_mdc --param 3
"""

import argparse
import os
from pathlib import Path

import numpy as np

PARAMS = ("Teff", "[Fe/H]", "logg", "[a/Fe]")


def entry_rows(stars, param: int, n_cat: int) -> np.ndarray:
    """``(n_cat, n_star)`` rows of the entry each catalog contributes, or -1."""
    n_star = len(stars["id"])
    rows = np.full((n_cat, n_star), -1, dtype="i8")
    sel = np.flatnonzero(stars["m"][:, param])
    rows[stars["c"][sel], stars["seg"][sel]] = sel
    return rows


def presence(stars, param: int, n_cat: int) -> np.ndarray:
    """``(n_star, n_cat)`` whether that star has a measured entry in a catalog."""
    n_star = len(stars["id"])
    pres = np.zeros((n_star, n_cat), dtype=bool)
    sel = np.flatnonzero(stars["m"][:, param])
    pres[stars["seg"][sel], stars["c"][sel]] = True
    return pres


def pull(y, Y, sig, nu, measured, param: int):
    """``d rho / d eps`` and ``d2 rho / d eps2`` for shifting one observation.

    ``rho = (nu + d) ln(1 + chi2/nu)`` over the parameters the entry measures, so
    with ``w = d rho / d chi2 = (nu + d)/(nu + chi2)``,

        d rho/d eps   = 2 w (y_p - Y_p) / sigma_p^2
        d2 rho/d eps2 = 2 w [1/sigma_p^2 - 2 ((y_p - Y_p)/sigma_p^2)^2/(nu + chi2)],

    for a shift of the ``param`` component of the observation.
    """
    d = measured.sum(axis=1).astype("f8")
    delta = np.where(measured, y - Y, 0.0)
    chi2 = np.sum(np.where(measured, delta ** 2 / sig ** 2, 0.0), axis=1)
    w = (nu + d) / (nu + chi2)
    g = 2.0 * w * delta[:, param] / sig[:, param] ** 2
    h = 2.0 * w * (1.0 / sig[:, param] ** 2
                   - 2.0 * (delta[:, param] / sig[:, param] ** 2) ** 2 / (nu + chi2))
    return g, h


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--delta-dir", type=Path, default=None,
                        help="the delta model whose sigmas this run was fitted with "
                             "(default: <run-dir>/delta)")
    parser.add_argument("--delta-model-type", default=None,
                        help="which model to use when that directory holds several "
                        "(const_add, const_scale_add, network_scale_add)")
    parser.add_argument("--focus", default="sdssv_mdc")
    parser.add_argument("--param", type=int, default=3, choices=range(4))
    parser.add_argument("--validation", action="store_true",
                        help="restrict to the validation stars")
    parser.add_argument("--eps", type=float, default=None,
                        help="check the analytic derivative by finite differences")
    parser.add_argument("--out", type=Path, default=None, help="figure path")
    args = parser.parse_args(argv)

    from astrotune.delta.feed import resolve_delta_model_dir
    from astrotune.delta.model import load_delta_model, model_nu, numpy_delta_fn
    from astrotune.delta.stars import load_star_table, split, with_groups

    run_dir = Path(args.run_dir)
    root = Path(args.delta_dir) if args.delta_dir else run_dir / "delta"
    delta_dir = resolve_delta_model_dir(root, args.delta_model_type)
    p = args.param
    dname = PARAMS[p]

    model = load_delta_model(delta_dir)
    # The star table is shared by every model type, so it sits at the root.
    table = delta_dir / "star_table.npz"
    if not table.exists():
        table = root / "star_table.npz"
    stars, keys = load_star_table(table)
    stars = with_groups(stars, keys)
    keys = tuple(keys)
    if args.validation:
        _, stars = split(stars)
    n_cat = len(keys)
    f = keys.index(args.focus)

    # The sigma this run was actually fitted with: the delta model's calibration of
    # the quoted sigma the star table carries.
    fn = numpy_delta_fn(model)
    ln_scale, ln_add = fn(stars["x"], stars["ls"], stars["g"])
    sig = np.sqrt(np.exp(ln_scale.astype("f8")) * stars["sig"].astype("f8") ** 2
                  + np.exp(ln_add.astype("f8")))
    sig[~stars["m"]] = np.inf

    # Per-catalog nu: one value per group, mapped back to the catalogs. A Gaussian
    # model (no learned tail) is the nu -> infinity limit.
    nu_group = model_nu(model)
    from astrotune import registry

    gidx = registry.group_index(keys)
    nu = (np.asarray([nu_group[gi] for gi in gidx])
          if nu_group is not None else np.full(n_cat, 1e6))

    # The star's fitted latent value, which is what the observations were pulled to.
    with np.load(run_dir / "latent_y.npz") as z:
        ylat, ids = z["y"], z["id"]
    order = np.searchsorted(ids, stars["id"])
    Y = ylat[order].astype("f8")  # (n_star, 4)

    rows = entry_rows(stars, p, n_cat)
    pres = presence(stars, p, n_cat)
    have_f = pres[:, f]
    print(f"{run_dir} fitted with {delta_dir}")
    print(f"{int(have_f.sum()):,} stars have a measured {dname} in {args.focus}")
    print(f"median recalibrated sigma of {args.focus} {dname}: "
          f"{np.median(sig[rows[f][have_f], p]):.4f}\n")

    # ---- 1. pairs, 2. sigma distributions, 3. pull -----------------------------
    head = (f"{'catalog':22s}{'n_pairs':>9s}{'sig X':>8s}{'sig focus':>10s}"
            f"{'med (focus-X)':>14s}{'dNLL/d eps':>12s}{'per pair':>10s}"
            f"{'implied shift':>14s}")
    print(f"pairs of {args.focus} in {dname}; sigma in dex, shift in dex")
    print(head)
    print("-" * len(head))
    results = {}
    for k in range(n_cat):
        if k == f:
            continue
        pair = have_f & pres[:, k]
        n = int(pair.sum())
        if n < 20:
            continue
        rf, rk = rows[f][pair], rows[k][pair]
        sig_f = sig[rf, p]
        sig_k = sig[rk, p]
        # difference in the harmonised space, expressed in dex through the focus
        # catalog's own Jacobian for that parameter
        scale = float(np.median(np.abs(stars["J"][rf, p, p]))) or 1.0
        diff = (stars["y"][rf, p] - stars["y"][rk, p]) / scale
        g, h = pull(stars["y"][rf].astype("f8"), Y[pair], sig[rf].astype("f8"),
                    nu[f], stars["m"][rf], p)
        implied = -g.sum() / h.sum() / scale if h.sum() > 0 else np.nan
        results[keys[k]] = dict(n=n, g=g.sum(), h=h.sum(), g_pair=g,
                                sig_f=sig_f, sig_k=sig_k, diff=diff, pair=pair,
                                ys=stars["y"][rk, p] / scale)
        print(f"{keys[k]:22s}{n:9,d}{np.median(sig_k):8.4f}{np.median(sig_f):10.4f}"
              f"{np.median(diff):+14.3f}{g.sum():12.1f}{g.mean():10.2f}{implied:+14.3f}")

    # the same over every star the focus observes, for scale
    g_all, h_all = pull(stars["y"][rows[f][have_f]].astype("f8"), Y[have_f],
                        sig[rows[f][have_f]].astype("f8"), nu[f],
                        stars["m"][rows[f][have_f]], p)
    scale = float(np.median(np.abs(stars["J"][rows[f][have_f], p, p]))) or 1.0
    print("-" * len(head))
    print(f"{'all focus entries':22s}{int(have_f.sum()):9,d}{'':18s}{'':14s}"
          f"{g_all.sum():12.1f}{g_all.mean():10.2f}"
          f"{-g_all.sum() / h_all.sum() / scale:+14.3f}")

    # ---- the hypothesis: is one end of the other catalog's range doing it? -----
    worst = min(results, key=lambda k: results[k]["g"])
    r = results[worst]
    q = np.quantile(r["ys"], [0.25, 0.5, 0.75])
    print(f"\nwhere the pull comes from, {worst} pairs (strongest dNLL/d eps < 0):")
    print(f"{'quartile of ' + worst + ' ' + dname:34s}{'n':>9s}{'med X':>9s}"
          f"{'med (focus-X)':>15s}{'dNLL/d eps':>12s}{'share':>9s}")
    for lo, hi, name in ((-np.inf, q[0], "lowest"), (q[0], q[1], "2nd"),
                         (q[1], q[2], "3rd"), (q[2], np.inf, "highest")):
        sel = (r["ys"] >= lo) & (r["ys"] < hi)
        if not sel.any():
            continue
        g = r["g_pair"][sel]
        print(f"{name:34s}{int(sel.sum()):9,d}{np.median(r['ys'][sel]):9.3f}"
              f"{np.median(r['diff'][sel]):+15.3f}{g.sum():12.1f}"
              f"{100 * g.sum() / r['g_pair'].sum():8.0f}%")

    # ---- is the discrepancy a zero point or a slope? --------------------------
    # Fit focus = a + b * partner on the pairs. b < 1 means the focus catalog's
    # range is compressed -- agreement at one end of the distribution and a growing
    # shortfall at the other, which is what "it only agrees with the low-alpha
    # catalog" looks like when the low-alpha catalog is the one at that end.
    print(f"\nfocus against partner on the pairs: {args.focus} = a + b * partner")
    print(f"{'partner':22s}{'n':>9s}{'b':>8s}{'+-':>7s}{'a':>9s}{'rms':>8s}")
    for name, r in sorted(results.items(), key=lambda kv: -kv[1]["n"]):
        xp = np.asarray(r["ys"], dtype="f8")
        yf = np.full(len(xp), np.nan)
        rf = rows[f][r["pair"]]
        scale = float(np.median(np.abs(stars["J"][rf, p, p]))) or 1.0
        yf = stars["y"][rf, p] / scale
        ok = np.isfinite(xp) & np.isfinite(yf)
        b, a = np.polyfit(xp[ok], yf[ok], 1)
        resid = yf[ok] - (a + b * xp[ok])
        sb = resid.std() / (xp[ok].std() * np.sqrt(ok.sum()))
        print(f"{name:22s}{int(ok.sum()):9,d}{b:8.3f}{sb:7.3f}{a:+9.3f}{resid.std():8.3f}")

    # ---- check the analytic derivative against a finite difference -----------
    if args.eps:
        eps = args.eps
        sub = results[worst]["pair"]
        rf = rows[f][sub]
        y0 = stars["y"][rf].astype("f8")
        mk = stars["m"][rf]
        tot = []
        for sign in (+1, -1):
            y1 = y0.copy()
            y1[:, p] += sign * eps
            d = mk.sum(axis=1).astype("f8")
            delta = np.where(mk, y1 - Y[sub], 0.0)
            chi2 = np.sum(np.where(mk, delta ** 2 / sig[rf].astype("f8") ** 2, 0.0), axis=1)
            tot.append(float(np.sum((nu[f] + d) * np.log1p(chi2 / nu[f]))))
        fd = (tot[0] - tot[1]) / (2 * eps)
        print(f"\nfinite-difference check over the {worst} pairs: analytic "
              f"{results[worst]['g']:.2f} vs {fd:.2f} nats per unit shift")

    # ---- figure: who is precise on the shared stars ---------------------------
    import matplotlib

    if not os.environ.get("DISPLAY"):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    showed = sorted(results, key=lambda k: -np.median(results[k]["sig_k"]))
    edges = np.linspace(0.0, 0.08, 81)
    fig, ax = plt.subplots(figsize=(10, 6.5), layout="constrained")
    for k, name in enumerate(showed):
        r = results[name]
        color = f"C{k}"
        ax.hist(np.clip(r["sig_k"], edges[0], edges[-1]), bins=edges, log=True,
                histtype="step", color=color, lw=1.5, label=f"{name} ({r['n']:,})")
        ax.hist(np.clip(r["sig_f"], edges[0], edges[-1]), bins=edges, log=True,
                histtype="step", ls="--", color=color, lw=0.8, alpha=0.7)
    ax.set(xlabel=rf"recalibrated $\sigma$ on {dname}", ylabel="number of pairs",
           title=(f"{args.focus} {dname} uncertainty vs its partners, on the stars "
                  f"they share\nsolid: the partner (labeled); dashed: {args.focus} "
                  f"on the same stars"))
    ax.legend(fontsize=8, ncols=2, title="partner (number of pairs)")
    ax.grid(alpha=0.25)
    out = args.out or (run_dir / "evaluation" / f"{args.focus}_{p}_pairs.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
