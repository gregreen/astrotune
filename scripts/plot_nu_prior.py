#!/usr/bin/env python
"""Draw the prior on nu, with each group's learned nu marked on it.

One page per delta output directory, written to ``<out_dir>/evaluation/nu_prior.png``
-- the same figure a training run now writes for itself. Runs that differ only in
the prior can then be flipped through one after another.

Usage:
    scripts/plot_nu_prior.py models/run/delta_e64 models/run/delta_e64_nu14_6 ...
"""

import argparse
import json
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out_dirs", nargs="+", type=Path)
    parser.add_argument("--name", default=None,
                        help="output file name inside evaluation/ (default nu_prior.png)")
    args = parser.parse_args(argv)

    from astrotune.delta.model import load_delta_model
    from astrotune.delta.train import plot_nu_prior

    for out_dir in args.out_dirs:
        with open(out_dir / "delta_model_config.json") as f:
            saved = json.load(f)
        cfg = saved["config"]
        space = cfg.get("nu_prior_space", "raw")
        model = load_delta_model(out_dir)
        nu = model.nu().numpy().ravel()
        path = plot_nu_prior(
            nu, saved["keys"], cfg["nu_prior_mean"], cfg["nu_prior_sd"],
            out_dir / "evaluation" / (args.name or "nu_prior.png"),
            title=(f"{out_dir.parent.name}: prior on nu "
                   f"({cfg['nu_prior_mean']:g} $\\pm$ {cfg['nu_prior_sd']:g} "
                   f"in {space})"),
            space=space,
        )
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
