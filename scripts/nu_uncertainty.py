#!/usr/bin/env python
"""How well is each group's ``nu`` actually measured?

A learned ``nu`` means one of two things, and the number alone does not say
which: the data prefer it, or the prior was simply echoed back. The only way to
tell them apart is to ask how far the likelihood moves when ``nu`` is perturbed,
i.e. how wide the error bar on ``nu`` is compared with the distance from the
prior mean.

This computes that error bar from the curvature of the log likelihood,

    sigma_c = 1 / sqrt(d2 NLL / d raw_c^2),

evaluated at the fitted parameters with the rest of the model frozen. The
displacement ``nu_c - nu_prior`` in units of that sigma is the pull the data
exerted; 1 means the prior let go by one sigma, and anything under ~1 means the
reported ``nu`` is the prior.

Two caveats, in opposite directions. Freezing the network overstates the
curvature (the profile likelihood is flatter), while the fitted ``nu`` is a
posterior mode, pulled short of where the data alone would put it. So treat the
signed pull as an order-of-magnitude statement, not a p-value.

Because the pull alone can be large on a difference that does not matter, the
same pass also reports what it would cost, in nats per star, to insist on the
prior mean with everything else frozen. Compare the two runs' cost columns with
care: ``nu`` and the scale/add network are partly degenerate (the network can
reshuffle mass between tail and width), which is why the cost of moving a group
to the prior mean is not the same in a run whose prior sits at 8 and one whose
prior sits at 16. A two-stage fit -- network first with ``nu`` fixed, then ``nu``
alone with the network frozen -- is the way to pin ``nu`` down reproducibly.

Usage:
    scripts/nu_uncertainty.py models/run/delta_e64 [--max-stars 200000]
"""

import argparse
import json
from pathlib import Path

import numpy as np


def star_mask_of_group(stars, group: int) -> np.ndarray:
    """A mask over stars with at least one entry in ``group``."""
    n_star = len(stars["id"])
    m = np.zeros(n_star, dtype=bool)
    m[stars["seg"][stars["g"] == group]] = True
    return m


def group_nll(stars, model, group: int, max_stars: int, seed: int = 0) -> tuple:
    """Sum of the per-star NLL over the stars that have an entry in ``group``.

    Returns ``(total, n_star_used, n_star_total)``: the total is over the sampled
    stars only, so the caller rescales the curvature by ``n_total / n_used``.
    """
    from astrotune.delta.model import star_nll
    from astrotune.delta.stars import subset

    m = star_mask_of_group(stars, group)
    n_total = int(m.sum())
    idx = np.flatnonzero(m)
    if max_stars and len(idx) > max_stars:
        idx = np.random.default_rng(seed).choice(idx, max_stars, replace=False)
    nu = model.nu()
    shared = model.shared.variance() if hasattr(model, "shared") else None
    n_group = int(stars["g"].max()) + 1
    total, n_used = 0.0, 0
    for lo in range(0, len(idx), 2000):
        # Blocks are whole stars, so a star's value is never split across two
        # likelihood evaluations.
        sub = subset(stars, idx[lo:lo + 2000])
        nll, _ = star_nll(_batch(sub), model, nu=nu, n_group=n_group, shared=shared)
        total += float(np.sum(nll.numpy()))
        n_used += len(sub["id"])
    return total, n_used, n_total


def _batch(stars) -> dict:
    """The keys ``star_nll`` reads, as tensors."""
    import tensorflow as tf

    return {k: tf.constant(np.asarray(stars[k])) for k in
            ("x", "ls", "sig", "J", "c", "g", "y", "m", "seg")}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--max-stars", type=int, default=200_000,
                        help="stars per group used for the curvature")
    parser.add_argument("--step", type=float, default=0.25,
                        help="finite-difference step in raw (nu moves about as much)")
    parser.add_argument("--validation", action="store_true",
                        help="use the validation stars (default: the training ones)")
    args = parser.parse_args(argv)

    import tensorflow as tf

    from astrotune.delta.model import load_delta_model
    from astrotune.delta.stars import load_star_table, split, with_groups

    out_dir = Path(args.out_dir)
    with open(out_dir / "delta_model_config.json") as f:
        saved = json.load(f)
    cfg = saved["config"]
    model = load_delta_model(out_dir)
    stars, keys = load_star_table(out_dir / "star_table.npz")
    stars = with_groups(stars, keys)
    if args.validation:
        _, stars = split(stars)

    raw0 = model.tail.raw.numpy().copy()
    prior_mean = float(cfg.get("nu_prior_mean", 6.0))
    prior_sd = float(cfg.get("nu_prior_sd", 2.0))
    nu_prior = 2.0 + float(np.logaddexp(0.0, prior_mean))
    labels = saved.get("group_labels") or [str(i) for i in range(len(raw0))]

    rows = []
    for g in range(len(raw0)):
        def nll_at(delta: float) -> tuple:
            raw = raw0.copy()
            raw[g] += np.float32(delta)
            model.tail.raw.assign(raw)
            total, n_used, n_total = group_nll(stars, model, g, args.max_stars)
            # Rescaled to the whole group, and per star for readability.
            return total * (n_total / max(n_used, 1)), total / max(n_used, 1)

        base, base_ps = nll_at(0.0)
        up, _ = nll_at(args.step)
        dn, _ = nll_at(-args.step)
        # What it would cost to insist on the prior mean with everything else fixed:
        # the honest "does the data care at all?" number, in nats per star.
        _, prior_ps = nll_at(prior_mean - float(raw0[g]))
        model.tail.raw.assign(raw0)
        curv = (up - 2.0 * base + dn) / args.step ** 2
        sigma = 1.0 / np.sqrt(curv) if curv > 0 else np.inf
        # nu = 2 + softplus(raw); d nu / d raw is ~1 at these values.
        nu = 2.0 + float(np.logaddexp(0.0, float(raw0[g])))
        rows.append(dict(group=labels[g], nu=nu, sigma=float(sigma),
                         pull=(nu - nu_prior) / sigma if np.isfinite(sigma) else np.nan,
                         cost=float(prior_ps - base_ps)))

    head = (f"{'group':38s}{'nu':>7s}{'sigma_nu':>10s}{'nu - prior':>12s}"
            f"{'pull (sigma)':>14s}{'cost (nats/star)':>18s}")
    print(f"prior on nu: {nu_prior:.2f} +- {prior_sd:.2f} (raw {prior_mean:.2f} +- "
          f"{prior_sd:.2f}); {out_dir}")
    print(head)
    for r in sorted(rows, key=lambda r: -abs(r["pull"]) if np.isfinite(r["pull"]) else 0):
        print(f"{r['group']:38s}{r['nu']:7.2f}{r['sigma']:10.2f}"
              f"{r['nu'] - nu_prior:12.2f}{r['pull']:14.2f}{r['cost']:16.4f}")
    print()
    print("sigma_nu is conditional on the network (a lower bound on the error bar);")
    print("cost is the validation NLL per star of forcing nu back to the prior mean.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
