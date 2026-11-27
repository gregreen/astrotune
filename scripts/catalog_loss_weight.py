#!/usr/bin/env python
"""How much of the training loss does each catalog carry, and are its residuals
structured?

    python scripts/catalog_loss_weight.py --run-dir models/<run>

The data term is a sum over *observations*, so a catalog's influence is not its
share of the rows: it is ``n_rows * loss_per_row``, and the loss per row grows
with the ratio of the real scatter to the quoted uncertainty. A catalog whose
errors are under-quoted therefore dominates the fit even when it is a small
fraction of the data, and its residuals are the ones the map and the latent
``y`` are bent to reduce.

For every catalog this prints the mean chi^2 per parameter, the mean loss per row
(Student-t, ``(nu + d) / 2 * log1p(chi2 / nu)``), and the share of the whole data
term. It also measures the *shape* of the residuals: the correlation matrix of
``f_c(x) - y`` over the four parameters. A catalog whose quoted error covariance
is strongly non-diagonal -- or whose errors are too small along one direction --
shows up as a correlated residual, which no invertible map can remove, because
the map is a single smooth transformation while the correlation is per row.

Writes ``diagnostic_plots/<key>/catalog_loss_weight.{txt,png}``, or
``--out-dir``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

from astrotune import registry
from astrotune.config import PARAM_NAMES


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--observations",
        type=Path,
        default=Path("data/observations_and_ctrl_points_11m26.h5"),
        help="observation file the run was trained on",
    )
    parser.add_argument("--per-catalog", type=int, default=20000,
                        help="rows sampled per catalog (0 = all)")
    parser.add_argument("--out-dir", type=Path, default=Path("diagnostic_plots/loss_weight"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    import tensorflow as tf
    from astrotune.model.io import enable_memory_growth, load_bijector

    enable_memory_growth()
    keys = list(registry.run_keys(args.run_dir))

    with h5py.File(args.observations, "r") as f:
        group = f["obs_dict"]
        x = np.asarray(group["x"][:], dtype="f4")
        catalog = np.asarray(group["c"][:], dtype="i8")
        ids = np.asarray(group["id"][:], dtype="i8")
        sigma = np.asarray(group["xs"][:], dtype="f4")
        attrs = dict(group["x"].attrs)
    nu = float(attrs.get("nu", 25.0)) if not np.isscalar(attrs.get("nu")) else float(attrs["nu"])

    star_ids = np.unique(ids)
    star = np.searchsorted(star_ids, ids)
    del ids

    # The latent y lives in the training checkpoint, not in the exported model.
    y = tf.Variable(tf.zeros([len(star_ids), x.shape[1]], tf.float32), name="y")
    newest = max(
        (args.run_dir / "checkpoints").glob("ckpt-*.index"),
        key=lambda p: int(p.name.split("-")[1].split(".")[0]),
    )
    tf.train.Checkpoint(latent_y=y).restore(str(newest)[: -len(".index")]).expect_partial()
    y = y.numpy()

    bij = load_bijector(args.run_dir)
    rng = np.random.default_rng(args.seed)

    def forward(xx, cc):
        out = []
        for i in range(0, len(xx), 8192):
            out.append(
                bij.forward_with_s(
                    tf.constant(xx[i:i + 8192]), tf.constant(cc[i:i + 8192])
                )[0].numpy()
            )
        return np.concatenate(out).astype("f8")

    n_total = {int(k): int(v) for k, v in zip(*np.unique(catalog, return_counts=True))}
    table, resid = [], {}
    for ci, key in enumerate(keys):
        idx = np.flatnonzero(catalog == ci)
        if not len(idx):
            continue
        if args.per_catalog and len(idx) > args.per_catalog:
            idx = rng.choice(idx, args.per_catalog, replace=False)
        r = forward(x[idx], np.full(len(idx), ci, dtype="i4")) - y[star[idx]]
        s = sigma[idx].astype("f8")
        measured = s < 1e6
        chi2 = np.where(measured, (r / np.where(measured, s, 1.0)) ** 2, 0.0).sum(1)
        dof = measured.sum(1).astype("f8")
        loss = 0.5 * (nu + dof) * np.log1p(chi2 / nu)
        # only where both parameters are measured: a placeholder column holds no
        # information and its residual is arbitrary.
        corr = np.full((4, 4), np.nan)
        for i in range(4):
            for j in range(4):
                both = measured[:, i] & measured[:, j]
                if both.sum() > 100:
                    corr[i, j] = np.corrcoef(r[both, i], r[both, j])[0, 1]
        row = dict(
            key=key,
            n=len(idx),
            n_total=n_total.get(ci, len(idx)),
            chi2=chi2.mean(),
            loss=loss.mean(),
            rms=np.sqrt((r ** 2).mean(0)),
            corr=corr,
        )
        row["share"] = row["loss"] * row["n_total"]
        table.append(row)
        resid[key] = row["corr"]

    total = sum(row["share"] for row in table)
    for row in table:
        row["share"] /= total

    order = sorted(table, key=lambda r: -r["share"])
    lines = [
        f"data term of {args.run_dir} (nu = {nu:g}), sampled "
        f"{args.per_catalog or 'all'} rows per catalog",
        "",
        f"{'catalog':24s} {'rows':>9s} {'chi2':>7s} {'loss/row':>9s} {'share':>7s} "
        f"{'Teff':>7s} {'[Fe/H]':>7s} {'logg':>7s} {'[a/Fe]':>7s}",
    ]
    for row in order:
        lines.append(
            f"{row['key']:24s} {row['n_total']:9d} {row['chi2']:7.2f} {row['loss']:9.3f} "
            f"{100 * row['share']:6.1f}% "
            + " ".join(f"{v:7.3f}" for v in row["rms"])
        )
    lines += ["", "residual correlation, f_c(x) - y (Teff, [Fe/H], logg, [a/Fe]):"]
    for row in order[:6]:
        c = resid[row["key"]]
        lines.append(f"  {row['key']} (loss share {100 * row['share']:.0f}%)")
        for i in range(4):
            lines.append("    " + " ".join(f"{c[i, j]:6.2f}" for j in range(4)))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "catalog_loss_weight.txt").write_text("\n".join(lines) + "\n")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 0.28 * len(order) + 1.8),
                             layout="constrained")
    ypos = np.arange(len(order))[::-1]
    axes[0].barh(ypos, [100 * r["share"] for r in order], color="C0")
    axes[0].set_yticks(ypos)
    axes[0].set_yticklabels([r["key"] for r in order], fontsize=7)
    axes[0].set_xlabel("share of the data term (%)")
    axes[1].barh(ypos, [r["chi2"] for r in order], color="C1")
    axes[1].set_yticks([])
    axes[1].set_xscale("log")
    axes[1].set_xlabel(r"mean $\chi^2$ per row (log)")
    names = [r["key"] for r in order]
    mat = np.array([np.r_[resid[k][np.triu_indices(4, 1)]] for k in names])
    im = axes[2].imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    labels = [f"{a}-{b}" for a, b in
              (("Teff", "[Fe/H]"), ("Teff", "logg"), ("Teff", "[a/Fe]"),
               ("[Fe/H]", "logg"), ("[Fe/H]", "[a/Fe]"), ("logg", "[a/Fe]"))]
    axes[2].set_xticks(range(len(labels)))
    axes[2].set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    axes[2].set_yticks(ypos)
    axes[2].set_yticklabels(names, fontsize=7)
    axes[2].set_title(r"correlation of $f_c(x) - y$")
    fig.colorbar(im, ax=axes[2], shrink=0.7)
    fig.savefig(args.out_dir / "catalog_loss_weight.png", dpi=140)

    print("\n".join(lines))
    print(f"\nwrote {args.out_dir}/catalog_loss_weight.{{txt,png}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
