#!/usr/bin/env python
"""Extract every observation of the stars that one catalog observes, plus their y.

    python scripts/clam_subset.py --run-dir models/<run>
    python scripts/clam_subset.py --run-dir models/<run> --catalog sdssv_clam

Writes an HDF5 file with

``obs/``
    ``id``, ``catalog`` (index into ``attrs["catalog_keys"]``), ``x``,
    ``sigma_x``, ``x_err``, ``star`` (index into ``stars/id``) and ``fx`` for
    **every** observation of every star that ``--catalog`` observes, from every
    catalog. ``fx`` is the trained map applied to that observation,
    ``f_c(x)``, which is what the data term compares with the star's ``y`` --
    the residual ``fx - y[star]`` is the catalog's contribution to the loss.
``stars/``
    ``id`` and ``y``: the harmonized parameter vector the run has inferred for
    each of those stars, taken from the latest checkpoint.

The point is to be able to look at one catalog's pathology in the context of the
stars it shares with the others, without carrying the full 6.3 M-observation
file: the training loss of a star is a sum over its rows, and the map has to fit
all of them at once.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

from astrotune import registry


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--observations",
        type=Path,
        default=Path("data/observations_and_ctrl_points_11m26b.h5"),
    )
    parser.add_argument("--catalog", default="sdssv_clam")
    parser.add_argument(
        "--output", type=Path, default=None,
        help="default: data/<catalog>_subset.h5",
    )
    args = parser.parse_args(argv)

    output = args.output or Path("data") / f"{args.catalog}_subset.h5"

    with h5py.File(args.observations, "r") as f:
        group = f["obs_dict"]
        ids = np.asarray(group["id"][:], dtype="i8")
        catalog = np.asarray(group["c"][:], dtype="i4")
        x = np.asarray(group["x"][:], dtype="f4")
        sigma = np.asarray(group["xs"][:], dtype="f4")
        x_err = np.asarray(group["xe"][:], dtype="f4")
        keys = str(f.attrs["catalogs"]).split(",")
    if args.catalog not in keys:
        print(f"{args.catalog} is not one of {keys}", file=sys.stderr)
        return 1
    c_index = keys.index(args.catalog)
    print(f"{len(ids):,} observations of {len(np.unique(ids)):,} stars; "
          f"{args.catalog} is category {c_index} of {len(keys)}")

    # The star index of every observation, exactly as ``load_observations`` makes it:
    # the observations are sorted by id, so the unique ids are already in order.
    star_ids, star = np.unique(ids, return_inverse=True)

    # Every star that the catalog observes, and then every observation of it.
    in_catalog = np.zeros(len(star_ids), dtype=bool)
    in_catalog[star[catalog == c_index]] = True
    keep = in_catalog[star]
    print(f"{int(in_catalog.sum()):,} stars are observed by {args.catalog}; "
          f"they carry {int(keep.sum()):,} observations from all catalogs")

    # Which catalogs those stars come from, for the record.
    n_by = np.bincount(catalog[keep], minlength=len(keys))

    import tensorflow as tf

    from astrotune.model.io import enable_memory_growth, load_bijector

    enable_memory_growth()
    newest = max(
        (args.run_dir / "checkpoints").glob("ckpt-*.index"),
        key=lambda p: int(p.name.split("-")[1].split(".")[0]),
    )
    step = int(newest.name.split("-")[1].split(".")[0])
    y_var = tf.Variable(tf.zeros([len(star_ids), x.shape[1]], tf.float32), name="y")
    tf.train.Checkpoint(latent_y=y_var).restore(str(newest)[: -len(".index")]).expect_partial()
    y = y_var.numpy()

    # The map applied to every observation, in the run's own ODE tolerance: this is
    # the ``f_c(x)`` the data term compares with ``y``.
    bij = load_bijector(args.run_dir)
    fx = np.empty_like(x)
    chunk = 16384
    for start in range(0, len(x), chunk):
        stop = min(start + chunk, len(x))
        fx[start:stop] = bij.forward_with_s(
            tf.constant(x[start:stop]), tf.constant(catalog[start:stop])
        )[0].numpy()
        if start % (chunk * 16) == 0:
            print(f"  f(x): {stop:,} of {len(x):,}", flush=True)

    # Rows of the subset, with the star index remapped to the subset.
    sub_stars = np.flatnonzero(in_catalog)
    remap = np.full(len(star_ids), -1, dtype="i4")
    remap[sub_stars] = np.arange(len(sub_stars), dtype="i4")
    rows = np.flatnonzero(keep)

    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "w") as f:
        obs = f.create_group("obs")
        for name, values in (
            ("id", ids[rows]),
            ("catalog", catalog[rows]),
            ("star", remap[star[rows]]),
            ("x", x[rows]),
            ("sigma_x", sigma[rows]),
            ("x_err", x_err[rows]),
            ("fx", fx[rows]),
        ):
            obs.create_dataset(name, data=values, compression="gzip", compression_opts=4)
        stars = f.create_group("stars")
        stars.create_dataset("id", data=star_ids[sub_stars], compression="gzip",
                             compression_opts=4)
        stars.create_dataset("y", data=y[sub_stars], compression="gzip",
                             compression_opts=4)
        f.attrs["catalog"] = args.catalog
        f.attrs["catalog_index"] = c_index
        f.attrs["catalog_keys"] = ",".join(keys)
        f.attrs["n_by_catalog"] = ",".join(str(int(v)) for v in n_by)
        f.attrs["run_dir"] = str(args.run_dir)
        f.attrs["checkpoint_step"] = step
        f.attrs["observations"] = str(args.observations)
        f.attrs["param_order"] = "Teff(5 ln T/5000), [Fe/H], logg, [alpha/Fe]"
        f.attrs["created_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    size = output.stat().st_size / 1e6
    print(f"wrote {output} ({size:.1f} MB): y from {args.run_dir} at step {step}")
    print("observations per catalog: " + ", ".join(
        f"{k} {int(n)}" for k, n in zip(keys, n_by) if n))
    return 0


if __name__ == "__main__":
    sys.exit(main())
