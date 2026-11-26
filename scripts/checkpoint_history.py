#!/usr/bin/env python
"""Recover the training history of a run that never finished.

``metrics_history.json`` is only written when a run completes, but every
checkpoint carries the history so far: ``state-<step>.npz`` holds a ``meta``
string with ``metrics`` (the same per-step lists) and ``agreement``. So a run
that was killed still has its curves -- read them from the newest state file.

Usage:
    scripts/checkpoint_history.py latent_student_t_iter1_11m24d_20cat
    scripts/checkpoint_history.py models/<run> --out diagnostic_plots/x.png

With ``--out`` the four most useful series are plotted; without it only the
summary and the requested window are printed.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def run_dir(name: str) -> Path:
    path = Path(name)
    if path.is_dir():
        return path
    for candidate in (Path("models") / name, Path("models") / f"latent_student_t_{name}"):
        if candidate.is_dir():
            return candidate
    raise SystemExit(f"no run directory for {name!r}")


def newest_state(d: Path) -> Path:
    states = sorted((d / "checkpoints").glob("state-*.npz"))
    if not states:
        raise SystemExit(f"{d} has no checkpoints")

    def step(p: Path) -> int:
        try:
            return int(p.stem.split("-")[1])
        except (IndexError, ValueError):
            return -1

    return max(states, key=step)


def load_history(state: Path) -> tuple[dict, dict]:
    z = np.load(state, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    return meta.get("metrics", {}), meta


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", help="run name under models/ or a path")
    parser.add_argument("--state", type=Path, default=None,
                        help="a specific state-*.npz (default: the newest)")
    parser.add_argument("--metric", default="loss", help="metric to summarise")
    parser.add_argument("--window", nargs=2, type=int, default=None,
                        help="step range to print in full")
    parser.add_argument("--out", type=Path, default=None, help="figure to write")
    args = parser.parse_args(argv)

    d = run_dir(args.run)
    state = args.state or newest_state(d)
    metrics, meta = load_history(state)
    if not metrics:
        raise SystemExit(f"{state} carries no metrics")

    step = np.asarray(metrics["step_val"], dtype=float)
    n = len(metrics[args.metric])
    print(f"{d.name}: history from {state.name}")
    print(f"  checkpoint at step {meta.get('step')}, complete={meta.get('complete')}, "
          f"elapsed {meta.get('elapsed', 0) / 3600:.2f} h")
    print(f"  {len(step):,} validation records up to step {step[-1]:.0f}; "
          f"{n:,} training records")

    values = np.asarray(metrics[args.metric], dtype=float)
    print(f"\n{args.metric}: min {values.min():.4g} (step {values.argmin() + 1}), "
          f"max {values.max():.4g} (step {values.argmax() + 1}), "
          f"median {np.median(values):.4g}")
    for key in ("chi2", "penalty_div", "penalty_ctrl"):
        if key in metrics:
            v = np.asarray(metrics[key], dtype=float)
            print(f"  {key:14s} at the {args.metric} maximum: {v[values.argmax()]:.4g}"
                  f"   median {np.median(v):.4g}")

    if args.window:
        lo, hi = args.window
        cols = [k for k in ("loss", "chi2", "penalty_div", "penalty_ctrl", "norm")
                if k in metrics]
        print(f"\nsteps {lo}-{hi}")
        print(f"{'step':>7s}" + "".join(f"{c:>14s}" for c in cols))
        for i in range(lo - 1, min(hi, len(values))):
            if i % 10 == 0:
                print(f"{i + 1:>7d}" + "".join(
                    f"{np.asarray(metrics[c], float)[i]:14.4g}" for c in cols))

    if args.out is not None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(11, 6), layout="constrained")
        panels = (("loss", "loss"), ("chi2", "chi2 (train)"),
                  ("penalty_div", "divergence penalty"), ("penalty_ctrl", "control penalty"))
        for ax, (key, title) in zip(axes.ravel(), panels):
            if key not in metrics:
                continue
            v = np.asarray(metrics[key], dtype=float)
            ax.plot(np.arange(1, len(v) + 1), v, lw=0.4, alpha=0.35, rasterized=True)
            window = max(1, len(v) // 400)
            if len(v) > 4 * window:
                smooth = np.convolve(v, np.ones(window) / window, mode="valid")
                ax.plot(np.arange(window, len(v) + 1), smooth, lw=1.0, c="C1")
            ax.set(title=title, xlabel="step")
            ax.grid(True, alpha=0.2)
        if "loss_val" in metrics:
            axes[0, 0].plot(step, metrics["loss_val"], lw=0.8, c="C2", label="validation")
            axes[0, 0].legend(fontsize=7)
        fig.suptitle(f"{d.name} (checkpoint at step {meta.get('step')})", fontsize=9)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=130)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
