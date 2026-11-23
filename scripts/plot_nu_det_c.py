#!/usr/bin/env python
"""nu against the volume of the corrected covariance, one dot per catalog.

For each delta output directory this writes ``<out_dir>/evaluation/nu_det_c.png``,
the same figure a training run now draws for itself.

``nu`` is one number per group of catalogs, so the members of a group share an x.
``dln det C`` is per catalog: the median over its validation entries of
``ln det(S_a + diag(tau_g^2)) - ln det C_m,a``, i.e. how much volume the group's
recalibration and its shared systematic add to (or remove from) that catalog's
quoted covariance, on the parameters the entry measures. That is the covariance the
merge will use for the entry, so the number is the one that matters downstream --
with the caveat that the shared part is present in every member of the group.

Usage:
    scripts/plot_nu_det_c.py models/run/delta_e64_nus8_2 ...
"""

import argparse
import json
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out_dirs", nargs="+", type=Path)
    parser.add_argument("--validation", action="store_true", default=True,
                        help="use the validation stars (the default, as in evaluation/)")
    parser.add_argument("--all", dest="validation", action="store_false",
                        help="use every star in the table")
    args = parser.parse_args(argv)

    from astrotune.delta.model import load_delta_model
    from astrotune.delta.stars import load_star_table, split, with_groups
    from astrotune.delta.train import catalog_summary, plot_nu_det_c

    for out_dir in args.out_dirs:
        with open(out_dir / "delta_model_config.json") as f:
            saved = json.load(f)
        model = load_delta_model(out_dir)
        stars, keys = load_star_table(out_dir / "star_table.npz")
        stars = with_groups(stars, keys)
        if args.validation:
            _, stars = split(stars)
        summary = catalog_summary(model, stars, keys)
        path = plot_nu_det_c(
            summary, out_dir / "evaluation" / "nu_det_c.png",
            title=f"{out_dir.parent.name}: nu against the volume of the corrected "
                  f"covariance",
        )
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
