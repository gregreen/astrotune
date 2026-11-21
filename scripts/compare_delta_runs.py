#!/usr/bin/env python
"""Put several delta fits side by side.

Each argument is a delta output directory (the one holding ``evaluation/``). For
every run this prints

* the headline validation numbers of the network arm (NLL per star, median PIT,
  the tail fractions, the scale ratios), next to the two constant baselines;
* the per-group ``nu``, which is the quantity that motivated the comparison;
* how much each fit was still moving when it stopped -- the validation NLL over
  the last few epochs, and the largest change in ``nu`` over the same window. A
  fit whose ``nu`` was still drifting had not converged, whatever its NLL says.

Usage:
    scripts/compare_delta_runs.py models/run/delta models/run/delta_e64 ...

Histories written before the losses and ``nu`` were recorded per epoch only have
``train_nll``/``val_nll``; the panels that need more are skipped for those.
"""

import argparse
import json
from pathlib import Path

import numpy as np

ARMS = ("quoted (as prepared)", "constant add", "constant scale+add", "network scale+add")


def load(out_dir: Path) -> dict:
    with open(Path(out_dir) / "evaluation" / "comparison.json") as f:
        return json.load(f)


def headline(runs: dict) -> str:
    head = (f"{'run':34s}{'arm':20s}{'val NLL':>9s}{'med PIT':>9s}"
            + "".join(f"{'sf<' + format(t, 'g'):>9s}" for t in (1e-3, 1e-2, 5e-2))
            + "".join(f"{'r_' + p:>9s}" for p in ("Teff", "[Fe/H]", "logg", "[a/Fe]")))
    short = ("Teff", "[Fe/H]", "logg", "[a/Fe]")
    lines = [head]
    for name, run in runs.items():
        for arm in ARMS:
            if arm not in run["results"]:
                continue
            p = run["results"][arm]["pooled"]
            lines.append(
                f"{name:34s}{arm:20s}{run['nll'][arm]:9.4f}{p['median_pit']:9.3f}"
                + "".join(f"{f:9.2%}" for f in p["frac_sf"])
                + "".join(f"{s:9.2f}" for s in p["scale_ratio"]))
        lines.append("")
    lines.append(f"{'':34s}{'(nominal)':20s}{'':9s}{0.5:9.3f}"
                 + "".join(f"{t:9.2%}" for t in (1e-3, 1e-2, 5e-2))
                 + "".join(f"{1.0:9.2f}" for _ in short))
    return "\n".join(lines)


def nu_table(runs: dict) -> str:
    """Per-group nu for every arm that learns one."""
    groups = None
    for run in runs.values():
        if run.get("group_labels"):
            groups = run["group_labels"]
            break
    if groups is None:
        return ""
    arms = [a for a in ARMS if any(run.get("nus", {}).get(a) for run in runs.values())]
    if not arms:
        return ""
    head = f"{'group':38s}" + "".join(f"{a:>22s}" for a in arms)
    lines = ["", "nu per group", head,
             "-" * (38 + 22 * len(arms))]
    for i, g in enumerate(groups):
        row = f"{g:38s}"
        for arm in arms:
            values = [run["nus"][arm][i] for run in runs.values()
                      if run.get("nus", {}).get(arm) and i < len(run["nus"][arm])]
            row += f"{values[0]:8.2f}" + "".join(f"{v:7.2f}" for v in values[1:])
            if len(values) < len(runs):
                row += " " * (7 * (len(runs) - len(values)))
        lines.append(row)
    # The runs are the columns beyond the first, so name them under the arm.
    lines.append("")
    lines.append("columns per arm: " + ", ".join(runs))
    return "\n".join(lines)


def convergence(runs: dict, window: int = 5) -> str:
    """Was the fit still moving when it stopped?"""
    lines = ["", f"convergence (last {window} epochs of the network fit)", 
             f"{'run':34s}{'epochs':>7s}{'val NLL end':>13s}"
             f"{'d val NLL':>11s}{'max |d nu|':>11s}{'|grad| end':>11s}"]
    for name, run in runs.items():
        hist = run.get("history") or []
        if not hist:
            continue
        tail = hist[-min(window, len(hist)):]
        first, last = tail[0], tail[-1]
        d_nll = (last.get("val_nll") or last["train_nll"]) - \
                (first.get("val_nll") or first["train_nll"])
        d_nu = float("nan")
        if "nu" in first and "nu" in last:
            a = np.asarray(first["nu"], dtype="f8")
            b = np.asarray(last["nu"], dtype="f8")
            d_nu = float(np.max(np.abs(b - a)))
        grad = last.get("grad_norm", float("nan"))
        lines.append(f"{name:34s}{len(hist):7d}{last.get('val_nll', last['train_nll']):13.4f}"
                     f"{d_nll:11.4f}{d_nu:11.3f}{grad:11.3g}")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out_dirs", nargs="+", type=Path)
    parser.add_argument("--names", nargs="*", default=None,
                        help="labels for the rows (default: the directory names)")
    parser.add_argument("--window", type=int, default=5,
                        help="epochs to look back over when judging convergence")
    args = parser.parse_args(argv)

    names = args.names or [d.resolve().name for d in args.out_dirs]
    runs = {}
    for name, out_dir in zip(names, args.out_dirs):
        runs[name] = load(out_dir)
    print(headline(runs))
    print(nu_table(runs))
    print(convergence(runs, args.window))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
