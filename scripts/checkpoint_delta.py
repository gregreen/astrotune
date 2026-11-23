#!/usr/bin/env python
"""Compare two checkpoints of the same architecture, for one catalog's parameter.

A continuation is only saved as a SavedModel when it finishes, so during the run the
only way to see what the map has done is to restore the raw checkpoint. This does
that for two prefixes and reports, on the stars of one catalog:

* the change in the *harmonized* value the map produces (``dy``), and
* the change in the fitted per-star value (``dY``), which is where the map is being
  pulled to.

Both are the numbers a pair analysis talks about, so a continuation can be checked
against the pull computed before it started.

Usage:
    scripts/checkpoint_delta.py \\
        --config models/run/config.json \\
        --old models/run/checkpoints/ckpt-20750 \\
        --new models/run_cont/checkpoints/ckpt-20985 \\
        --star-table models/run/delta/star_table.npz \\
        --focus sdssv_mdc --param 3
"""

import argparse
import json
from pathlib import Path

import numpy as np

PARAMS = ("Teff", "[Fe/H]", "logg", "[a/Fe]")


def load_bijector_at(prefix: str, cfg: dict, n_star: int):
    """The map and the free values as they were at ``prefix``."""
    import tensorflow as tf

    from astrotune.model.ode_bijector import ODEBijector

    bij = ODEBijector(
        4, int(cfg["n_cat"]), int(cfg["hidden_size"]), int(cfg["n_hidden"]),
        rtol=float(cfg["ode_rtol"]), atol=float(cfg["ode_atol"]),
        vector_network_kwargs=dict(use_residual=True, use_film=True, use_layer_norm=True),
    )
    # Built to be restored, never trained: the shapes have to match the checkpoint's.
    y = tf.Variable(np.zeros((n_star, 4), dtype="f4"), name="latent_y", dtype=tf.float32)
    y_val = tf.Variable(np.zeros((n_star, 4), dtype="f4"), name="latent_y_val",
                        dtype=tf.float32)
    tf.train.Checkpoint(bij=bij, latent_y=y, latent_y_val=y_val).restore(prefix) \
        .expect_partial()
    return bij, y


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, nargs="+", required=True,
                        help="one config.json per checkpoint, in the same order")
    parser.add_argument("--ckpt", nargs="+", required=True)
    parser.add_argument("--star-table", type=Path, required=True)
    parser.add_argument("--focus", default="sdssv_mdc")
    parser.add_argument("--param", type=int, default=3, choices=range(4))
    parser.add_argument("--steps", type=int, nargs="+", default=None,
                        help="step number of each checkpoint, for the labels")
    args = parser.parse_args(argv)

    from astrotune.delta.stars import load_star_table
    from astrotune.model.io import enable_memory_growth

    enable_memory_growth()
    p, dname = args.param, PARAMS[args.param]
    stars, keys = load_star_table(args.star_table)
    keys = tuple(keys)
    n_star = len(stars["id"])
    f = keys.index(args.focus)
    sel = np.flatnonzero((stars["c"] == f) & stars["m"][:, p])
    x, c = stars["x"][sel], stars["c"][sel]
    print(f"{args.focus}: {len(sel):,} entries with a measured {dname}")

    out = []
    for cfg_path, ckpt in zip(args.config, args.ckpt):
        with open(cfg_path) as fh:
            cfg = json.load(fh)["config"]
        cfg.setdefault("n_cat", len(keys))
        cfg["n_cat"] = len(keys)
        bij, y = load_bijector_at(ckpt, cfg, n_star)
        xy, _ = bij.forward_with_s(x, c=c)
        yy = np.asarray(xy)
        Y = y.numpy()[stars["seg"][sel]]
        out.append((yy[:, p].copy(), Y[:, p].copy()))
        print(f"  {Path(ckpt).name}: median {dname} (map) {np.median(yy[:, p]):+.4f}, "
              f"spread {yy[:, p].std():.4f}")

    (y0, Y0), (y1, Y1) = out[0], out[1]
    dy, dY = y1 - y0, Y1 - Y0
    print(f"\n{'':24s}{'change':>10s}{'|change| median':>18s}{'90th pct':>10s}")
    print(f"{'map output dy':24s}{np.median(dy):+10.5f}{np.median(np.abs(dy)):18.5f}"
          f"{np.percentile(np.abs(dy), 90):10.5f}")
    print(f"{'fitted Y of their stars':24s}{np.median(dY):+10.5f}"
          f"{np.median(np.abs(dY)):18.5f}{np.percentile(np.abs(dY), 90):10.5f}")
    moved_up = float(np.mean(dy > 0))
    print(f"fraction of the {args.focus} entries whose {dname} rose: {moved_up:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
