"""Is a catalog's quoted uncertainty right, and what scale factor does the data support?

    astrotune-error-scale sdssv_clam --run-dir models/turnover50_64ep_16kbatch_19cat
    astrotune-error-scale lamost_mstar_li2025 --run-dir models/X --partners sdssv_clam sdssv_mdc

Several catalogs rescale their quoted uncertainties before the error floors are added
(``UNC_SCALES`` in the catalog module). This tool shows the evidence, from stars that are also
in other catalogs.

**Which space matters.** What we want to know is how much the *statistical* scatter is
under-estimated, not how large the systematic differences between catalogs are (the bijector
calibrates those away). Compared in the catalogs' own coordinates (5 ln Teff/5000 K, [Fe/H],
logg, [alpha/Fe]), the differences contain offsets that depend on the parameters, which inflate
the scatter and the fitted plateau. The default mode here is therefore the **harmonized space**
(``--run-dir RUN``): both estimates of a star go through the trained bijector, the quoted errors
are propagated with its Jacobian, ``C = J diag(sigma**2) J^T``, and the scatter of
``y_catalog - y_partner`` is compared with the propagated errors. Without ``--run-dir`` the
comparison is in the input coordinates (kept for reference; do not use it to choose floors).

For a partner catalog and a parameter, the difference ``d = x_catalog - x_partner`` has the
variance ``sigma_obs**2 = sigma_c**2 + sigma_p**2`` (independent errors). Binning the stars by
the catalog's *quoted* uncertainty ``q`` (recovered from the prepared table by undoing the scale
and the floor), the robust variance of ``d`` minus the partner's variance is the variance
attributed to the catalog, ``v``. Then ``v = s**2 q**2 + sigma_sys**2``, fitted for the scale
``s`` and a systematic term ``sigma_sys`` (the error that does not shrink with ``q``: the
"plateau"). The partner's own error, a constant offset within a bin (``d`` is centered on its
median) and non-Gaussian tails (robust scatter) are the main caveats; see the figure notes.

Writes to ``diagnostic_plots/<key>/``:

    error_scale.png        observed vs quoted uncertainty per parameter and partner
    error_scale_fit.png    the fitted scale ``s`` and plateau ``sigma_sys`` with 68 % intervals
    error_scale_z.png      the standardized differences with the quoted, the scaled and the
                           final (scaled + floor) uncertainty, against a unit Gaussian
                           (first partner only)
    error_scale_zcore.png  the width of those distributions for every partner
    error_scale.json       the numbers
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy import optimize

if not os.environ.get("DISPLAY"):  # headless by default
    import matplotlib

    matplotlib.use("Agg")

import matplotlib.pyplot as plt

from .. import config, registry
from .io import load_catalog
from .transforms import default_floors

log = logging.getLogger(__name__)

PARAMS = ("5 ln(Teff/5000 K)", "[Fe/H]", "logg", "[alpha/Fe]")
SHORT = ("Teff", "[Fe/H]", "logg", "[a/Fe]")
MISSING = 1e6

#: Partners for the catalogs that rescale their errors (the best-calibrated, independent catalogs)
DEFAULT_PARTNERS = {
    "sdssv_clam": ("apogee_dr17", "sdssv_aspcap", "galah_dr4"),
    "lamost_mstar_li2025": ("sdssv_clam", "sdssv_aspcap", "apogee_dr17", "desi_dr1", "sdssv_mdc"),
}
#: Partners whose labels (or spectra) the catalog was trained on or shares, so that the
#: scatter against them is not an independent test
NOT_INDEPENDENT = {"lamost_mstar_li2025": ("apogee_dr17", "sdssv_aspcap")}


def robust_scale(v: np.ndarray) -> float:
    """1.4826 x the median absolute deviation."""
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


def module_scales_and_floors(key: str, n_params: int):
    """``(UNC_SCALES, floors)`` a catalog's preparation used."""
    mod = registry.get_module(key)
    scales = np.asarray(getattr(mod, "UNC_SCALES", (1.0,) * n_params), dtype="f8")[:n_params]
    floors = np.asarray(getattr(mod, "ERROR_FLOORS", default_floors(n_params)), dtype="f8")[:n_params]
    return scales, floors


def quoted_sigma(key: str, sigma_x: np.ndarray) -> np.ndarray:
    """The catalog's own uncertainty, recovered from the prepared ``sigma_x``.

    ``sigma_x = sqrt((scale * quoted)**2 + floor**2)``, so ``quoted = sqrt(sigma_x**2 - floor**2) / scale``.
    """
    scales, floors = module_scales_and_floors(key, sigma_x.shape[1])
    return np.sqrt(np.maximum(sigma_x.astype("f8") ** 2 - floors**2, 0.0)) / scales


def first_rows(table):
    """One row per star: the first one, sorted by id."""
    ids = np.asarray(table["id"])
    _, first = np.unique(ids, return_index=True)
    return table[first]


def pair_sample(key: str, partner: str, param: int, cache: Optional[dict] = None) -> Optional[dict]:
    """Stars measured by both catalogs: ``d``, the catalog's quoted sigma, scaled/final sigma, partner sigma."""
    cache = {} if cache is None else cache
    for k in (key, partner):
        if k not in cache:
            cache[k] = first_rows(load_catalog(k))
    c, p = cache[key], cache[partner]
    if np.asarray(c["x"]).shape[1] <= param or np.asarray(p["x"]).shape[1] <= param:
        return None
    ids, ic, ip = np.intersect1d(np.asarray(c["id"]), np.asarray(p["id"]), return_indices=True)
    sc, sp = np.asarray(c["sigma_x"])[ic, param], np.asarray(p["sigma_x"])[ip, param]
    xc, xp = np.asarray(c["x"])[ic, param], np.asarray(p["x"])[ip, param]
    ok = (sc < MISSING) & (sp < MISSING) & np.isfinite(xc) & np.isfinite(xp)
    if ok.sum() < 500:
        return None
    q = quoted_sigma(key, np.asarray(c["sigma_x"])[ic][ok])[:, param]
    scales, floors = module_scales_and_floors(key, np.asarray(c["sigma_x"]).shape[1])
    return dict(d=(xc - xp)[ok], q=q, sp=sp[ok], sc_final=sc[ok], scale=float(scales[param]), floor=float(floors[param]))


class Harmonizer:
    """Maps catalog entries to the harmonized space with a trained bijector, with the Jacobian."""

    def __init__(self, run_dir):
        import tensorflow as tf

        from ..model import load_bijector
        from ..model.io import enable_memory_growth

        enable_memory_growth()
        self.run_dir = Path(run_dir)
        self.keys = tuple(registry.run_keys(self.run_dir))
        bij = load_bijector(self.run_dir)

        @tf.function(input_signature=[tf.TensorSpec([None, config.N_PARAMS]), tf.TensorSpec([None], dtype=tf.int32)])
        def forward(xx, cc):
            with tf.GradientTape(watch_accessed_variables=False) as g:
                g.watch(xx)
                yy = bij._forward(xx, c=cc)
            return yy, g.batch_jacobian(yy, xx, experimental_use_pfor=False)

        self._forward, self._tf = forward, tf

    def __call__(self, key: str, x: np.ndarray, batch_size: int = 4096):
        """``(y, J)`` for entries ``x`` (n, 4) of catalog ``key``: ``J[n, i, j] = dy_i / dx_j``."""
        tf = self._tf
        c = self.keys.index(key)
        ys, js = [], []
        for i in range(0, len(x), batch_size):
            xb = tf.constant(x[i : i + batch_size], tf.float32)
            y, j = self._forward(xb, tf.fill([xb.shape[0]], tf.constant(c, tf.int32)))
            ys.append(y.numpy())
            js.append(j.numpy())
        return np.concatenate(ys).astype("f8"), np.concatenate(js).astype("f8")


def propagate(J: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Diagonal of ``J diag(sigma**2) J^T``: the uncertainty of every output parameter, (n, 4)."""
    return np.sqrt(np.einsum("npk,nk->np", J**2, sigma.astype("f8") ** 2))


def harmonized_pair(key: str, partner: str, hz: Harmonizer, cache: dict) -> Optional[dict]:
    """Stars in both catalogs, in the harmonized space.

    Returns arrays of shape ``(n, 4)``: the difference ``d = y_catalog - y_partner``, the
    catalog's *quoted* uncertainty ``q`` and its final (scaled + floors) uncertainty ``fin``,
    and the partner's uncertainty ``sp`` (all propagated through the Jacobian), and ``ok``.
    """
    from ..apply import padded_inputs

    for k in (key, partner):
        if k not in cache:
            cache[k] = first_rows(load_catalog(k))
    c, p = cache[key], cache[partner]
    ids, ic, ip = np.intersect1d(np.asarray(c["id"]), np.asarray(p["id"]), return_indices=True)
    if len(ids) < 500:
        return None
    out = {}
    for tag, k, t in (("c", key, c[ic]), ("p", partner, p[ip])):
        x, sigma, measured = padded_inputs(t)
        y, J = hz(k, x)
        sig = np.where(measured, sigma, 0.0)  # an unmeasured parameter adds no uncertainty
        out[tag] = dict(y=y, J=J, sigma=sig, measured=measured, table=t, key=k)
    quoted = quoted_sigma(key, np.asarray(out["c"]["table"]["sigma_x"]))
    cq = np.zeros_like(out["c"]["sigma"])  # the quoted uncertainty, 0 where a parameter is not measured
    cq[:, : quoted.shape[1]] = np.where(out["c"]["measured"][:, : quoted.shape[1]], quoted, 0.0)
    ok = out["c"]["measured"] & out["p"]["measured"]
    y_d = out["c"]["y"] - out["p"]["y"]
    scales, floors = module_scales_and_floors(key, np.asarray(out["c"]["table"]["sigma_x"]).shape[1])
    good = np.isfinite(y_d).all(1) & np.isfinite(out["c"]["J"]).all((1, 2)) & np.isfinite(out["p"]["J"]).all((1, 2))
    return dict(
        d=y_d[good], ok=ok[good], q=propagate(out["c"]["J"], cq)[good],
        fin=propagate(out["c"]["J"], out["c"]["sigma"])[good], sp=propagate(out["p"]["J"], out["p"]["sigma"])[good],
        scales=scales, floors=floors, ids=ids[good],
    )


def harmonized_sample(hp: dict, param: int) -> Optional[dict]:
    """One parameter of :func:`harmonized_pair`, in the format of :func:`pair_sample`."""
    ok = hp["ok"][:, param] & (hp["q"][:, param] > 0)
    if ok.sum() < 500:
        return None
    return dict(
        d=hp["d"][ok, param], q=hp["q"][ok, param], sp=hp["sp"][ok, param], sc_final=hp["fin"][ok, param],
        scale=float(hp["scales"][param]), floor=float(hp["floors"][param]),
    )


def bin_stats(d, q, sp, edges, fin=None):
    """Per bin of the quoted sigma: centre, attributed variance, its standard error, n and
    (with ``fin``) the median final uncertainty."""
    which = np.clip(np.digitize(q, edges[1:-1]), 0, len(edges) - 2)
    out = []
    for b in range(len(edges) - 1):
        s = which == b
        n = int(s.sum())
        if n < 30:
            out.append((np.nan, np.nan, np.nan, n, np.nan))
            continue
        var_obs = robust_scale(d[s]) ** 2
        v = var_obs - float(np.mean(sp[s] ** 2))
        se = 3.0 * var_obs / np.sqrt(n)  # standard error of a MAD-based variance (normal case)
        out.append((float(np.median(q[s])), v, se, n, float(np.median(fin[s])) if fin is not None else np.nan))
    return np.array(out)


def fit_scale(stats: np.ndarray):
    """``v = s**2 q**2 + sigma_sys**2`` by weighted non-negative least squares: ``(s, sigma_sys)``."""
    ok = np.isfinite(stats[:, 0]) & np.isfinite(stats[:, 1])
    if ok.sum() < 3:
        return np.nan, np.nan
    q, v, se = stats[ok, 0], stats[ok, 1], stats[ok, 2]
    A = np.stack([q**2, np.ones_like(q)], axis=1) / se[:, None]
    (a, b), _ = optimize.nnls(A, v / se)
    return float(np.sqrt(a)), float(np.sqrt(b))


def analyze(key: str, partners: Sequence[str], n_boot: int = 200, seed: int = 0,
            hz: Optional[Harmonizer] = None) -> dict:
    """Binned statistics and fits for every parameter and partner.

    With ``hz`` (a :class:`Harmonizer`) the comparison is in the harmonized space; otherwise in
    the catalogs' own coordinates.
    """
    rng = np.random.default_rng(seed)
    cache: dict = {}
    hp_cache: dict = {}
    res: dict = {"key": key, "partners": list(partners), "params": {},
                 "space": "harmonized" if hz is not None else "input"}
    for p in range(config.N_PARAMS):
        res["params"][p] = {}
        for partner in partners:
            if hz is not None:
                if partner not in hp_cache:
                    hp_cache[partner] = harmonized_pair(key, partner, hz, cache)
                s = harmonized_sample(hp_cache[partner], p) if hp_cache[partner] is not None else None
            else:
                s = pair_sample(key, partner, p, cache)
            if s is None:
                continue
            n = len(s["d"])
            n_bins = int(np.clip(n // 400, 4, 10))
            edges = np.quantile(s["q"], np.linspace(0, 1, n_bins + 1))
            edges[-1] += 1e-9
            stats = bin_stats(s["d"], s["q"], s["sp"], edges, s["sc_final"])
            fit = fit_scale(stats)
            boots = []
            for _ in range(n_boot):
                i = rng.integers(0, n, n)
                boots.append(fit_scale(bin_stats(s["d"][i], s["q"][i], s["sp"][i], edges)))
            boots = np.array(boots)
            lo, hi = np.nanpercentile(boots, [16, 84], axis=0)
            res["params"][p][partner] = dict(
                n=n, stats=stats, s=fit[0], s_lo=float(lo[0]), s_hi=float(hi[0]),
                sys=fit[1], sys_lo=float(lo[1]), sys_hi=float(hi[1]),
                scale_used=s["scale"], floor_used=s["floor"], sample=s,
            )
    return res


def plot_calibration(res: dict, labels: Dict[str, str], out_path: Path) -> Path:
    key = res["key"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9.5), layout="constrained")
    for p, ax in enumerate(axes.flat):
        entries = res["params"].get(p, {})
        if not entries:
            ax.set_visible(False)
            continue
        lo = min(np.nanmin(e["stats"][:, 0]) for e in entries.values())
        hi = max(np.nanmax(e["stats"][:, 0]) for e in entries.values())
        grid = np.geomspace(lo / 1.5, hi * 1.5, 100)
        e0 = next(iter(entries.values()))
        s_used, floor = e0["scale_used"], e0["floor_used"]
        ax.plot(grid, grid, "k:", lw=1.2, label="quoted")
        ax.plot(grid, s_used * grid, "k--", lw=1.0, label="quoted x scale used")
        if res.get("space") == "harmonized":  # the floors are in input units: use the propagated final error
            st0 = e0["stats"]
            ok0 = np.isfinite(st0[:, 0]) & np.isfinite(st0[:, 4])
            ax.plot(st0[ok0, 0], st0[ok0, 4], "k-", lw=1.6, label="x scale, plus floor (final; median per bin)")
        else:
            ax.plot(grid, np.hypot(s_used * grid, floor), "k-", lw=1.6, label="x scale, plus floor (final)")
        for i, (partner, e) in enumerate(entries.items()):
            st = e["stats"]
            ok = np.isfinite(st[:, 1]) & (st[:, 1] > 0)
            color = f"C{i}"
            tag = " (not independent)" if partner in NOT_INDEPENDENT.get(key, ()) else ""
            ax.errorbar(st[ok, 0], np.sqrt(st[ok, 1]), yerr=0.5 * st[ok, 2] / np.sqrt(st[ok, 1]), fmt="o", ms=4,
                        color=color, label=f"{labels[partner]}{tag}")
            if np.isfinite(e["s"]):
                ax.plot(grid, np.sqrt((e["s"] * grid) ** 2 + e["sys"] ** 2), "-", color=color, lw=0.8, alpha=0.7)
        ax.set(xscale="log", yscale="log", xlabel="quoted uncertainty (binned)",
               ylabel="observed uncertainty attributed to the catalog",
               title=(f"{PARAMS[p]}   (used: x{s_used:g}" + (")" if res.get("space") == "harmonized" else f", floor {floor:g})")))
        ax.grid(True, alpha=0.2, which="both")
    handles, names = {}, []
    for ax in axes.flat:
        for h, n in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(n, h)
    fig.legend(list(handles.values()), list(handles), loc="outside lower center", ncol=2, fontsize=8)
    fig.suptitle(f"{labels[key]}, {res.get('space', 'input')} space: observed vs quoted uncertainty "
                 "(thin colored lines: fit of s and sigma_sys)", fontsize=10)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_fit(res: dict, labels: Dict[str, str], out_path: Path) -> Path:
    key = res["key"]
    fig, axes = plt.subplots(2, 4, figsize=(14, 6), layout="constrained")
    for p in range(config.N_PARAMS):
        entries = res["params"].get(p, {})
        for row, (name, lo_k, hi_k) in enumerate((("s", "s_lo", "s_hi"), ("sys", "sys_lo", "sys_hi"))):
            ax = axes[row, p]
            for i, (partner, e) in enumerate(entries.items()):
                ax.errorbar(i, e[name], yerr=[[e[name] - e[lo_k]], [e[hi_k] - e[name]]], fmt="o", color=f"C{i}")
            if entries:
                e0 = next(iter(entries.values()))
                ax.axhline(e0["scale_used"] if row == 0 else e0["floor_used"], color="k", ls="--", lw=1,
                           label="used" if p == 0 else None)
                ax.set_xticks(range(len(entries)), [labels[k] for k in entries], rotation=60, ha="right", fontsize=7)
            ax.set_ylim(bottom=0)
            ax.grid(True, alpha=0.2)
            ax.set_title(SHORT[p] + (": scale s" if row == 0 else ": plateau sigma_sys (floor used: dashed)"), fontsize=8)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"{labels[key]}, {res.get('space', 'input')} space: fitted scale and plateau per partner "
                 "(68 % bootstrap intervals)", fontsize=10)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_z(res: dict, labels: Dict[str, str], out_path: Path) -> Path:
    """Standardized differences against the first (preferred) partner, for three choices of uncertainty."""
    key = res["key"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    grid = np.linspace(-6, 6, 400)
    for p, ax in enumerate(axes.flat):
        entries = res["params"].get(p, {})
        if not entries:
            ax.set_visible(False)
            continue
        partner = next(iter(entries))
        e = entries[partner]
        s = e["sample"]
        d = s["d"] - np.median(s["d"])
        variants = (
            ("quoted", np.hypot(s["q"], s["sp"])),
            (f"x{s['scale']:g}", np.hypot(s["scale"] * s["q"], s["sp"])),
            (f"x{s['scale']:g} + floor {s['floor']:g} (final)", np.hypot(s["sc_final"], s["sp"])),
        )
        for (name, sig), color in zip(variants, ("C3", "C1", "C0")):
            z = d / sig
            ax.hist(np.clip(z, -6, 6), bins=80, range=(-6, 6), density=True, histtype="step", color=color,
                    label=f"{name}: robust sigma_z = {robust_scale(z):.2f}")
        ax.plot(grid, np.exp(-0.5 * grid**2) / np.sqrt(2 * np.pi), "k-", lw=1, label="unit Gaussian")
        ax.set(yscale="log", ylim=(1e-4, 1.5), xlabel="(x_catalog - x_partner - median) / sigma",
               title=f"{PARAMS[p]} vs {labels[partner]} (n={e['n']:,})")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)
    fig.suptitle(f"{labels[key]}, {res.get('space', 'input')} space: calibration of the standardized differences "
                 "(core width 1 = calibrated)", fontsize=10)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def z_core_widths(res: dict) -> dict:
    """Robust width of the standardized differences for three choices of uncertainty, per
    parameter and partner (1 = calibrated)."""
    out: dict = {}
    for p, entries in res["params"].items():
        for partner, e in entries.items():
            s = e["sample"]
            d = s["d"] - np.median(s["d"])
            out.setdefault(p, {})[partner] = tuple(
                robust_scale(d / sig)
                for sig in (np.hypot(s["q"], s["sp"]), np.hypot(s["scale"] * s["q"], s["sp"]),
                            np.hypot(s["sc_final"], s["sp"]))
            )
    return out


def plot_zcore(res: dict, labels: Dict[str, str], out_path: Path) -> Path:
    key = res["key"]
    widths = z_core_widths(res)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    for p, ax in enumerate(axes.flat):
        ws = widths.get(p, {})
        if not ws:
            ax.set_visible(False)
            continue
        x = np.arange(len(ws))
        for j, (name, color) in enumerate((("quoted", "C3"), ("scaled", "C1"), ("final (scaled + floor)", "C0"))):
            ax.bar(x + (j - 1) * 0.27, [w[j] for w in ws.values()], 0.27, color=color, label=name)
        ax.axhline(1, color="k", lw=1)
        ax.set_xticks(x, [labels[k] + ("*" if k in NOT_INDEPENDENT.get(key, ()) else "") for k in ws],
                      rotation=40, ha="right", fontsize=8)
        ax.set(title=PARAMS[p], ylabel="robust width of (difference / sigma)")
        ax.grid(True, alpha=0.2, axis="y")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"{labels[key]}, {res.get('space', 'input')} space: calibration of the standardized differences, "
                 "per partner (1 = calibrated; * not independent)", fontsize=10)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def format_summary(res: dict, labels: Dict[str, str]) -> str:
    lines = [f"{labels[res['key']]} ({res.get('space', 'input')} space): fitted scale s (68 % interval), "
             "plateau sigma_sys; used: scale / floor (input units)"]
    for p in range(config.N_PARAMS):
        for partner, e in res["params"].get(p, {}).items():
            lines.append(
                f"  {SHORT[p]:7s} vs {labels[partner]:26s} n={e['n']:7,}  s = {e['s']:5.2f} [{e['s_lo']:.2f}, {e['s_hi']:.2f}]"
                f"   sigma_sys = {e['sys']:.3f} [{e['sys_lo']:.3f}, {e['sys_hi']:.3f}]"
                f"   used: x{e['scale_used']:g} / {e['floor_used']:g}"
            )
    return "\n".join(lines)


def to_json(res: dict) -> dict:
    out = {"key": res["key"], "space": res.get("space", "input"), "partners": res["partners"], "params": {}}
    for p, entries in res["params"].items():
        out["params"][SHORT[p]] = {
            partner: {k: (np.asarray(v).tolist() if k == "stats" else v) for k, v in e.items() if k != "sample"}
            for partner, e in entries.items()
        }
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="astrotune-error-scale", description=__doc__.split("\n\n")[0])
    parser.add_argument("key", help="catalog key, e.g. sdssv_clam")
    parser.add_argument("--partners", nargs="+", default=None, help="partner catalog keys (default: a preset)")
    parser.add_argument("--out-dir", type=Path, default=None, help="default: diagnostic_plots/<key>")
    parser.add_argument("--n-boot", type=int, default=200)
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="trained run: compare in the HARMONIZED space (recommended). Without it the "
                        "comparison is in the catalogs' own coordinates, which includes systematic offsets")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    partners = args.partners or DEFAULT_PARTNERS.get(args.key)
    if not partners:
        partners = [k for k in registry.keys() if k != args.key and config.catalog_path(k).exists()]
    out_dir = args.out_dir or config.PROJECT_ROOT / "diagnostic_plots" / args.key
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = {k: registry.get_spec(k).label for k in registry.keys() + registry.candidate_keys()}

    hz = Harmonizer(args.run_dir) if args.run_dir else None
    if hz is not None and args.key not in hz.keys:
        parser.error(f"{args.key} is not one of the catalogs of {args.run_dir}")
    partners = [k for k in partners if k != args.key and (hz is None or k in hz.keys)]
    res = analyze(args.key, partners, args.n_boot, hz=hz)
    suffix = "_harmonized" if hz is not None else ""
    plot_calibration(res, labels, out_dir / f"error_scale{suffix}.png")
    plot_fit(res, labels, out_dir / f"error_scale_fit{suffix}.png")
    plot_z(res, labels, out_dir / f"error_scale_z{suffix}.png")
    plot_zcore(res, labels, out_dir / f"error_scale_zcore{suffix}.png")
    with open(out_dir / f"error_scale{suffix}.json", "w") as f:
        json.dump(to_json(res), f, indent=1)
    text = format_summary(res, labels)
    (out_dir / f"error_scale{suffix}.txt").write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
