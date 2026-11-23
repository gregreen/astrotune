#!/usr/bin/env python
"""Continue a finished latent run with different optimiser settings.

The point of a continuation rather than a fresh run is to change one thing about
the *optimiser* while keeping the fitted state: the map's weights, the free
harmonized values and Adam's moments all start where the previous run left them,
so any movement afterwards is the new setting acting on the old solution.

Only the final checkpoint (``ckpt-N`` plus the ``state-N.npz`` that carries the
optimizer variables and the metric history) is copied, not the whole 4 GB of
checkpoints: the continuation starts from the end.

The learning-rate schedule is made *constant* (``lr_schedule_phases=1``,
``lr_warmup_frac=0``) by default, since the run is a short controlled probe and a
cosine phase would otherwise re-introduce a warmup in the middle of it.

Usage:
    scripts/continue_latent_run.py \\
        --from models/latent_student_t_iter2_20cat \\
        --to models/latent_student_t_iter2_20cat_cont32_l2e5 \\
        --epochs 32 --lr 1e-6 --l2-weights 1e-5
"""

import argparse
import dataclasses
import json
import shutil
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--from", dest="src", type=Path, required=True)
    parser.add_argument("--to", dest="dst", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=32,
                        help="how many more epochs to train")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--l2-weights", type=float, default=None)
    parser.add_argument("--keep-schedule", action="store_true",
                        help="keep the run's cosine schedule instead of a constant rate")
    parser.add_argument("--dry-run", action="store_true",
                        help="set everything up and stop before training")
    args = parser.parse_args(argv)

    src, dst = Path(args.src), Path(args.dst)
    ckpts = sorted((src / "checkpoints").glob("ckpt-*.index"),
                   key=lambda p: int(p.name.split("-")[1].split(".")[0]))
    if not ckpts:
        raise SystemExit(f"no checkpoints in {src}/checkpoints")
    step = int(ckpts[-1].name.split("-")[1].split(".")[0])
    state = src / "checkpoints" / f"state-{step}.npz"
    if not state.exists():
        raise SystemExit(f"no {state}; the checkpoint cannot be resumed without it")

    with open(src / "config.json") as f:
        snapshot = json.load(f)
    cfgd = dict(snapshot["config"])
    n_epochs = int(cfgd["n_epochs"])
    steps_per_epoch, rem = divmod(step, n_epochs)
    if rem:
        raise SystemExit(
            f"{step} steps is not a whole number of {n_epochs} epochs; "
            "the epoch arithmetic below would be off"
        )
    # ``n_steps = steps_per_epoch * n_epochs``: extending n_epochs by the number
    # asked for is what adds exactly that many more epochs on top of ``step``.
    cfgd["n_epochs"] = n_epochs + args.epochs
    if args.lr is not None:
        cfgd["learning_rate"] = args.lr
    if args.l2_weights is not None:
        cfgd["l2_weights"] = args.l2_weights
    if not args.keep_schedule:
        cfgd["lr_schedule_phases"] = 1
        cfgd["lr_warmup_frac"] = 0.0
    snapshot["config"] = cfgd
    snapshot["continued_from"] = str(src)
    snapshot["continued_at_step"] = step

    (dst / "checkpoints").mkdir(parents=True, exist_ok=True)
    for suffix in (".index", ".data-00000-of-00001"):
        s = src / "checkpoints" / f"ckpt-{step}{suffix}"
        if s.exists():
            shutil.copy2(s, dst / "checkpoints" / s.name)
    shutil.copy2(state, dst / "checkpoints" / state.name)
    with open(dst / "config.json", "w") as f:
        json.dump(snapshot, f, indent=2)

    print(f"{src} at step {step:,} ({n_epochs} epochs) -> {dst}")
    print(f"  {args.epochs} more epochs, {cfgd['n_epochs']} in total "
          f"({cfgd['n_epochs'] * steps_per_epoch:,} steps)")
    print(f"  learning_rate {cfgd['learning_rate']:g}, "
          f"l2_weights {cfgd['l2_weights']:g}, "
          f"phases {cfgd['lr_schedule_phases']}, warmup {cfgd['lr_warmup_frac']:g}")

    if args.dry_run:
        return 0

    from astrotune.train_latent import LatentTrainConfig, _config_from_json, run_training

    cfg = LatentTrainConfig(**_config_from_json(cfgd))
    run_training(cfg, dst, Path(snapshot["observations_path"]), resume=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
