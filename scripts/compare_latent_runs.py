#!/usr/bin/env python
"""Put two latent (bijector) runs side by side.

The intended use is an A/B test in which the *only* difference between the two
runs is one architectural choice -- the Fourier feature block, say -- with the
same observations file, seed, schedule and batch size. For each run this prints

* which saved config fields differ, so the apples-to-apples claim is checked
  rather than assumed;
* the tail of the training loss and each penalty, as a median over the last
  few hundred steps (the raw series is far too noisy to read off the end);
* the post-harmonisation pair agreement per stellar parameter: the residual
  |bias| and scatter of a pair after both catalogues are mapped through the
  bijector, plus the fraction of pairs that improved on the prepared values.

Usage:
    scripts/compare_latent_runs.py models/latent_student_t_iter2b_20cat \
        models/latent_student_t_iter2b_fourier_20cat

The second run may still be training; whatever history exists is used and the
last step is reported. ``evaluation/`` is only written when a run finishes, so
the agreement table is skipped for a run that is still going.
"""

import argparse
import json
from pathlib import Path

import numpy as np

TRAIN_KEYS = ("loss", "chi2", "chi2_raw", "penalty_obs", "penalty_ctrl",
              "penalty_div", "penalty_l2")
VAL_KEYS = ("loss_val", "chi2_val", "chi2_raw_val", "penalty_obs_val",
            "penalty_ctrl_val", "penalty_div_val", "penalty_l2_val")
# the summarised agreement rows are parameters, not catalogs (``labels`` holds
# the twenty catalog names)
PARAM_NAMES = ("Teff", "[Fe/H]", "logg", "[a/Fe]")


def run_dir(name: str) -> Path:
    """Accept a path, a directory name, or the shorthands the tables print."""
    path = Path(name)
    if path.is_dir():
        return path
    for candidate in (Path("models") / name,
                      Path("models") / f"latent_student_t_{name}"):
        if candidate.is_dir():
            return candidate
    raise SystemExit(f"no run directory for {name!r}")


def load_config(d: Path) -> dict:
    with open(d / "config.json") as f:
        snap = json.load(f)
    return snap.get("config", snap)


def flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for key, value in d.items():
        if isinstance(value, dict):
            out.update(flatten(value, prefix + key + "."))
        else:
            out[prefix + key] = value
    return out


def load_history(d: Path) -> dict:
    path = d / "metrics_history.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def load_agreement(d: Path):
    """Post-harmonisation pair agreement: the final evaluation, else the live one."""
    final = d / "evaluation" / "agreement.json"
    if final.exists():
        with open(final) as f:
            return json.load(f), "evaluation"
    live = d / "agreement_history.json"
    if live.exists():
        with open(live) as f:
            history = json.load(f)
        if history:
            return history[-1], f"live @ step {history[-1].get('step', '?')}"
    return None, "missing"


def short_name(name: str) -> str:
    return name.replace("latent_student_t_", "")


def tail(values, window: int) -> float:
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.median(arr[-window:]))


def config_diff(cfg_a: dict, cfg_b: dict) -> str:
    fa, fb = flatten(cfg_a), flatten(cfg_b)
    rows = [(k, fa.get(k), fb.get(k)) for k in sorted(set(fa) | set(fb))
            if fa.get(k) != fb.get(k)]
    if not rows:
        return "config: identical"
    width = max(len(k) for k, _, _ in rows)
    lines = ["config differences"]
    for key, va, vb in rows:
        lines.append(f"  {key:{width}s}  {va!r} -> {vb!r}")
    return "\n".join(lines)


def metric_table(hist: dict, window: int, keys: tuple) -> dict:
    return {k: tail(hist.get(k, []), window) for k in keys}


def metric_block(runs: dict) -> str:
    names = list(runs)
    width = max(len(k) for k in TRAIN_KEYS + VAL_KEYS)
    lines = [f"loss tail (median of the last {runs[names[0]]['window']} steps)"]
    lines.append(f"  {'metric':{width}s}" + "".join(f"{n:>26s}" for n in names))
    for key in TRAIN_KEYS + VAL_KEYS:
        cells = []
        for name in names:
            value = runs[name]["train"].get(key, float("nan"))
            cells.append(f"{value:26.5g}")
        lines.append(f"  {key:{width}s}" + "".join(cells))
    lines.append("  " + f"{'last step':{width}s}"
                 + "".join(f"{runs[n]['last_step']:26d}" for n in names))
    return "\n".join(lines)


def agreement_block(runs: dict) -> str:
    avail = {n: r["agreement"] for n, r in runs.items()}
    if all(a is None for a in avail.values()):
        return "pair agreement: not available yet (runs unfinished)"
    lines = ["pair agreement after harmonisation"]
    for name, (summary, where) in avail.items():
        lines.append(f"  {name}: {where}" if summary else f"  {name}: missing")
    lines.append(f"  {'param':16s}" + "".join(f"{n:>34s}" for n in avail))
    lines.append(f"  {'':16s}" + "".join(
        f"{'abs bias/sigma  improved':>34s}" for _ in avail))
    n_param = max((len(s["summary"]) for s, _ in avail.values() if s), default=0)
    for i in range(n_param):
        cells = []
        for name in avail:
            summary = avail[name][0]
            if not summary or i >= len(summary["summary"]):
                cells.append(f"{'-':>34s}")
                continue
            row = summary["summary"][i]
            cells.append(f"{row['abs_bias_harm']:12.4f}{row['sigma_harm']:12.4f}"
                         f"{row['frac_bias_improved']:10.0%}")
        label = PARAM_NAMES[i] if i < len(PARAM_NAMES) else f"param {i}"
        lines.append(f"  {label:16s}" + "".join(cells))
    # the prepared values, which both runs start from
    for name in avail:
        summary = avail[name][0]
        if summary:
            row = summary["summary"]
            lines.append(f"  (prepared, {name}): "
                         + "  ".join(f"{r['abs_bias_orig']:.4f}/{r['sigma_orig']:.4f}"
                                     for r in row))
            break
    return "\n".join(lines)


def plot(runs: dict, out: Path, window: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(runs)
    colors = ["C0", "C3", "C2", "C4"]

    def binned(steps, values, n=400):
        """Median in equal-count bins, so 20k noisy points become a readable line."""
        values = np.asarray(values, dtype=float)
        steps = np.asarray(steps, dtype=float)
        if values.size <= n:
            return steps, values
        edges = np.linspace(0, values.size, n + 1).astype(int)
        idx = [(a + b) // 2 for a, b in zip(edges[:-1], edges[1:])]
        return (steps[idx] if steps.size == values.size else np.arange(len(idx)),
                [np.median(values[a:b]) for a, b in zip(edges[:-1], edges[1:])])

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for i, name in enumerate(names):
        hist = runs[name]["history"]
        color = colors[i % len(colors)]
        steps = hist.get("step_val") or []
        for ax, key in zip(axes[:2], ("chi2_val", "penalty_div_val")):
            series = hist.get(key) or []
            if series:
                x, y = binned(steps, series)
                ax.plot(x, y, color=color, lw=1.1, label=name)

    axes[0].set(title="chi2 (validation)", xlabel="step", ylabel="chi2")
    axes[1].set(title="divergence penalty (validation)",
                xlabel="step", ylabel="penalty")
    axes[2].set(title="final post-harmonisation scatter",
                xlabel="parameter", ylabel="sigma_harm")
    for i, name in enumerate(names):
        summary = runs[name]["agreement"][0] if runs[name]["agreement"] else None
        if not summary:
            continue
        sig = [row["sigma_harm"] for row in summary["summary"]]
        axes[2].plot(np.arange(len(sig)), sig, "o-", color=colors[i % len(colors)],
                     label=name)
    n_param = max((len(r["agreement"][0]["summary"]) for r in runs.values()
                   if r["agreement"] and r["agreement"][0]), default=0)
    if n_param:
        axes[2].set_xticks(np.arange(n_param))
        axes[2].set_xticklabels(PARAM_NAMES[:n_param], fontsize=9)
    axes[0].legend(fontsize=7)
    axes[2].legend(fontsize=7)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"\nwrote {out}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", help="run names under models/ or paths")
    parser.add_argument("--window", type=int, default=500,
                        help="steps to median over at the end of each series")
    parser.add_argument("--out", type=Path, default=None,
                        help="figure path (default: diagnostic_plots/latent_ab_<runs>.png)")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)

    runs = {}
    for name in args.runs:
        d = run_dir(name)
        history = load_history(d)
        if not history:
            # metrics_history.json is only written by LatentTrainer.save(), i.e.
            # when a run finishes; an in-flight run has config.json and nothing else.
            print(f"note: {d} has no metrics_history.json yet (run unfinished)")
        agreement, where = load_agreement(d)
        last = max((len(v) for v in history.values() if isinstance(v, list)), default=0)
        runs[short_name(d.name)] = {
            "dir": d,
            "config": load_config(d),
            "history": history,
            "agreement": (agreement, where),
            "train": metric_table(history, args.window, TRAIN_KEYS + VAL_KEYS),
            "last_step": max(history.get("step_val") or [0], default=0),
            "window": args.window,
        }

    names = list(runs)
    print(config_diff(runs[names[0]]["config"], runs[names[1]]["config"])
          if len(names) == 2 else "")
    print()
    print(metric_block(runs))
    print()
    print(agreement_block(runs))

    if not args.no_plot:
        if args.out is None:
            stem = "_vs_".join(n.replace("latent_student_t_", "") for n in names)
            args.out = Path("diagnostic_plots") / f"latent_ab_{stem}.png"
        plot(runs, args.out, args.window)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
