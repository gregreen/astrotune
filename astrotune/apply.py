"""Apply a trained bijector to every catalog and merge the results by star.

    astrotune-apply --run-dir models/my_run

Steps (reference notebook, cells 461-488, with the bugs fixed):

1. Each catalog's prepared table (``x``, ``sigma_x``) is mapped to the harmonized
   space with the bijector of the catalog's category. The per-entry
   uncertainties are propagated through the map with its Jacobian (first order),
   ``C_y = J diag(sigma_x^2) J^T``. Parameters that a catalog does not measure
   (``sigma_x > 1e6``) are fed to the map as zero, contribute nothing to ``C_y``,
   and come out as ``y = 0, C_pp = 1e9`` with zero off-diagonal covariance.
2. :func:`astrotune.merge.combine` merges the entries of each Gaia DR3 source
   (iterating with outlier rejection).
3. Stars with non-finite values or values outside ``PARAM_LIMS`` are removed.

Writes to ``<run-dir>/harmonized/`` (``--out-dir``):

    merged_catalog.h5        one row per star (columns below), with metadata
    individual_estimates.h5  one table ``catalog_<i>`` per input catalog
    metadata.json            the same metadata as plain JSON
    summary.png              overview figure

Columns of ``merged_catalog.h5``:

    gdr3_source_id   Gaia DR3 source id (sorted)
    params_est       (4,) harmonized [5 ln(Teff/5000 K), [Fe/H], logg, [alpha/Fe]]
    params_err       (4,) sqrt(C_ii + 0.03^2)
    params_cov       (4,4) covariance of the combined estimate
    lndet_cov        ln det of the covariance over the measured parameters
    params_source    uint32 bit mask: bit i set if catalog i contributed
                     (``metadata["surveys"]`` lists which catalog is which)
    n_estimates      number of catalog entries combined
    chi2_max_used    largest chi2 of the entries used, against the merged estimate
                     (large: the entries of this star disagree)
    params_measured  (4,) bool: measured by at least one contributing catalog

For a parameter that no contributing catalog measured, ``params_est`` is 0,
``params_cov`` is 1e9 on the diagonal and 0 elsewhere (``params_measured`` is
False): do not use the value.

Columns of each table in ``individual_estimates.h5``: ``gdr3_source_id``,
``params_orig`` / ``params_err_orig`` (the prepared table's ``x`` / ``sigma_x``),
``params_harm``, ``params_cov_harm``, ``measured``, ``chi2_merged`` (chi2 of this
entry against the merged estimate) and ``used`` (it contributed to the merged
estimate, i.e. it was not rejected as an outlier).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from astropy.table import Table
from tqdm.auto import tqdm

from . import config, registry
from .merge import CHI2_MAX, EPS_ERR, MISSING_VAR, N_ITER, Estimates, Merged, catalogs_in, combine

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1

#: Stars with a combined estimate outside these limits are dropped (set by hand
#: in the notebook, cell 486)
PARAM_LIMS = ((-2.5, 11.5), (-4.5, 1.9), (-2.5, 6.0), (-2.0, 2.0))

PARAM_NAMES = ("5 ln(Teff / 5000 K)", "[Fe/H]", "logg", "[alpha/Fe]")

#: Values at or above this mark a parameter that a catalog does not measure
MISSING = 1e6


# --- Transforming one catalog --------------------------------------------------
def make_transform(bij):
    """``f(x, sigma_x, c) -> (y, cov_y)`` for a batch, as a traced function."""
    import tensorflow as tf

    n_params = config.N_PARAMS

    @tf.function(
        input_signature=[
            tf.TensorSpec([None, n_params]),
            tf.TensorSpec([None, n_params]),
            tf.TensorSpec([None], dtype=tf.int32),
        ]
    )
    def transform(xx, sigma_xx, cc):
        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as g:
            g.watch(xx)
            yy = bij._forward(xx, c=cc)
        dyy_dxx = g.batch_jacobian(yy, xx, experimental_use_pfor=False)
        cov_yy = tf.einsum("bik,bk,bjk->bij", dyy_dxx, sigma_xx**2, dyy_dxx)
        return yy, cov_yy

    return transform


def padded_inputs(table: Table):
    """``x``, ``sigma_x`` padded to 4 parameters, and the ``measured`` mask.

    A parameter is unmeasured if the catalog has no such column or its
    ``sigma_x`` is a placeholder (> 1e6). Its ``x`` is set to 0.
    """
    d = config.N_PARAMS
    x_in, s_in = np.asarray(table["x"], dtype="f4"), np.asarray(table["sigma_x"], dtype="f4")
    n = len(table)
    x = np.zeros((n, d), dtype="f4")
    sigma = np.full((n, d), MISSING_VAR, dtype="f4")
    x[:, : x_in.shape[1]] = x_in
    sigma[:, : s_in.shape[1]] = s_in
    measured = (sigma < MISSING) & np.isfinite(x)
    x[~measured] = 0.0
    return x, sigma, measured


def transform_catalog(
    transform, table: Table, index: int, batch_size: int = 4096, desc: str = ""
) -> Estimates:
    """Harmonize every entry of ``table`` (catalog number ``index``)."""
    import tensorflow as tf

    x, sigma, measured = padded_inputs(table)
    # Missing parameters must not feed uncertainty into the other parameters
    sigma_prop = np.where(measured, sigma, 0.0).astype("f4")
    c = np.full(len(x), index, dtype="i4")

    ds = tf.data.Dataset.from_tensor_slices((x, sigma_prop, c)).batch(batch_size)
    ys, covs = [], []
    for xx, ss, cc in tqdm(ds, total=-(-len(x) // batch_size), desc=desc):
        y, cov = transform(xx, ss, cc)
        ys.append(y.numpy())
        covs.append(cov.numpy())
    y = np.concatenate(ys)
    cov = np.concatenate(covs)
    return finalize_estimates(index, np.asarray(table["id"]), y, cov, measured)


def finalize_estimates(index, ids, y, cov, measured) -> Estimates:
    """Put the unmeasured parameters into the standard encoding."""
    y = y.copy()
    cov = cov.copy()
    for p in range(y.shape[1]):
        m = ~measured[:, p]
        y[m, p] = 0.0
        cov[m, p, :] = 0.0
        cov[m, :, p] = 0.0
        cov[m, p, p] = MISSING_VAR
    return Estimates(index=index, ids=ids, y=y, cov=cov, measured=measured)


# --- Cache of the (slow) transformation ---------------------------------------
def _cache_path(cache_dir: Path, index: int) -> Path:
    return cache_dir / f"catalog_{index:02d}.npz"


def save_cached(cache_dir: Path, est: Estimates) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = cache_dir / "tmp.npz"
    np.savez(tmp, ids=est.ids, y=est.y.astype("f4"), cov=est.cov.astype("f4"), measured=est.measured)
    os.replace(tmp, _cache_path(cache_dir, est.index))


def load_cached(cache_dir: Path, index: int, ids: np.ndarray) -> Optional[Estimates]:
    """The cached estimates of catalog ``index`` if they belong to the same rows."""
    path = _cache_path(cache_dir, index)
    if not path.exists():
        return None
    with np.load(path) as f:
        if len(f["ids"]) != len(ids) or not np.array_equal(f["ids"], ids):
            log.warning("Cache %s does not match the current table; recomputing", path)
            return None
        return Estimates(index, f["ids"], f["y"], f["cov"], f["measured"])


# --- Output ---------------------------------------------------------------------
def _jsonable(obj):
    return json.loads(json.dumps(obj, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def survey_info(keys: Sequence[str], indices: Sequence[int], tables, estimates, merged: Merged, final_mask) -> List[dict]:
    """One dictionary per catalog: identity plus what became of its entries."""
    from .prep.io import load_meta

    out = []
    last = merged.rejected_history[-1] if merged.rejected_history else {}
    for key, idx, t, est, used in zip(keys, indices, tables, estimates, merged.used):
        spec = registry.get_spec(key)
        try:
            mod = registry.get_module(key)
            reference = getattr(mod, "REFERENCE", "")
        except Exception:  # pragma: no cover - module import problems must not stop a run
            reference = ""
        try:
            prep = load_meta(key)
            prep = {k: prep.get(k) for k in ("created_utc", "git_commit", "n_rows", "cut_counts")}
        except OSError:
            prep = {}
        sel = catalogs_in(merged.params_source, idx)
        out.append(
            dict(
                index=int(idx),
                bit=int(1 << idx),
                key=key,
                label=spec.label,
                reference=reference,
                n_params_catalog=int(spec.n_params),
                parameters_measured=[
                    n for n, m in zip(PARAM_NAMES, est.measured.any(axis=0)) if m
                ],
                n_entries=int(len(est)),
                n_unique_stars=int(len(np.unique(est.ids))),
                n_entries_used=int(np.count_nonzero(used)),
                rejected_frac_all_entries=last.get(idx, {}).get("rejected_frac"),
                rejected_frac_multiply_observed=last.get(idx, {}).get("rejected_frac_multi"),
                n_stars_contributing=int(np.count_nonzero(sel)),
                n_stars_contributing_final_catalog=int(np.count_nonzero(sel & final_mask)),
                prepared_table=prep,
            )
        )
    return out


def clean_mask(merged: Merged, lims=PARAM_LIMS):
    """Finite estimates inside ``lims``; returns ``(mask, counts)``."""
    finite = np.all(np.isfinite(merged.params), axis=1) & np.all(np.isfinite(merged.err), axis=1)
    inside = np.ones_like(finite)
    with np.errstate(invalid="ignore"):
        for p, (lo, hi) in enumerate(lims):
            inside &= (merged.params[:, p] > lo) & (merged.params[:, p] < hi)
    counts = dict(
        n_sources=int(len(finite)),
        n_not_finite=int(np.count_nonzero(~finite)),
        n_out_of_bounds=int(np.count_nonzero(finite & ~inside)),
        n_kept=int(np.count_nonzero(finite & inside)),
    )
    return finite & inside, counts


COLUMN_DOC = {
    "gdr3_source_id": "Gaia DR3 source id",
    "params_est": "harmonized [5 ln(Teff/5000 K), [Fe/H], logg, [alpha/Fe]]",
    "params_err": "sqrt(C_ii + eps_err^2)",
    "params_cov": "covariance of the combined estimate (harmonized space)",
    "lndet_cov": "ln det(params_cov) over the measured parameters",
    "params_source": "uint32 bit mask; bit i set if catalog i contributed (see surveys)",
    "n_estimates": "number of catalog entries combined",
    "chi2_max_used": "largest chi2 of the entries used vs the merged estimate; large = entries disagree",
    "params_measured": "parameter measured by at least one contributing catalog",
}


def write_outputs(out_dir: Path, merged: Merged, estimates, tables, metadata: dict, mask) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = _jsonable(metadata)
    paths = {}

    sel = np.flatnonzero(mask)
    t = Table(
        {
            "gdr3_source_id": merged.ids[sel],
            "params_est": merged.params[sel].astype("f4"),
            "params_err": merged.err[sel].astype("f4"),
            "params_cov": merged.cov[sel].astype("f4"),
            "lndet_cov": merged.lndet_cov[sel].astype("f4"),
            "params_source": merged.params_source[sel].astype("u4"),
            "n_estimates": merged.n_used[sel],
            "chi2_max_used": merged.chi2_max_used[sel].astype("f4"),
            "params_measured": merged.measured[sel],
        },
        meta=meta,
    )
    for name, text in COLUMN_DOC.items():
        t[name].description = text
    paths["merged"] = out_dir / "merged_catalog.h5"
    t.write(paths["merged"], path="data", serialize_meta=True, overwrite=True, compression=True)

    paths["individual"] = out_dir / "individual_estimates.h5"
    if paths["individual"].exists():
        paths["individual"].unlink()
    for est, tab, used, chi2, info in zip(estimates, tables, merged.used, merged.chi2, meta["surveys"]):
        # The prepared table's own values (x = 0 and sigma_x = 1e9 where unmeasured)
        x, sigma, _ = padded_inputs(tab)
        ti = Table(
            {
                "gdr3_source_id": est.ids,
                "params_orig": x,
                "params_err_orig": sigma,
                "params_harm": est.y.astype("f4"),
                "params_cov_harm": est.cov.astype("f4"),
                "measured": est.measured,
                "chi2_merged": chi2.astype("f4"),
                "used": used,
            },
            meta=dict(info),
        )
        ti.write(
            paths["individual"], path=f"catalog_{est.index}", serialize_meta=True,
            append=True, compression=True,
        )

    paths["metadata"] = out_dir / "metadata.json"
    with open(paths["metadata"], "w") as f:
        json.dump(meta, f, indent=2)
    return paths


def read_merged(path):
    """Read ``merged_catalog.h5``; returns ``(table, metadata)``.

    ``metadata["surveys"]`` maps catalog index / bit / key / label; use
    :func:`astrotune.merge.catalogs_in` with a survey's ``index`` on
    ``table["params_source"]`` to select the stars it contributed to.
    """
    t = Table.read(path, path="data")
    return t, dict(t.meta)


def plot_summary(out_path: Path, table: Table, surveys: List[dict]) -> Path:
    import matplotlib

    if not os.environ.get("DISPLAY"):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    p = np.asarray(table["params_est"])
    meas = np.asarray(table["params_measured"])
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), layout="constrained")

    ax = axes[0, 0]
    k = meas[:, 0] & meas[:, 2]  # stars without a measured logg would sit at logg = 0
    ax.hist2d(p[k, 0], p[k, 2], bins=150, norm=LogNorm(), cmin=1, range=[[-3.0, 5.0], [-0.5, 6.0]])
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set(xlabel=r"$5\ln(T_{\rm eff}/5000\,{\rm K})$", ylabel=r"$\log g$",
           title=f"{np.count_nonzero(k):,} of {len(table):,} stars with Teff and logg")

    ax = axes[0, 1]
    a = meas[:, 3] & (np.asarray(table["params_err"])[:, 3] < 1.0)
    ax.hist2d(p[a, 1], p[a, 3], bins=150, norm=LogNorm(), cmin=1, range=[[-3.5, 1.0], [-0.5, 0.8]])
    ax.set(xlabel="[Fe/H]", ylabel=r"[$\alpha$/Fe]", title=f"{np.count_nonzero(a):,} stars with [alpha/Fe]")

    ax = axes[1, 0]
    labels = [s["label"] for s in surveys]
    y = np.arange(len(surveys))
    ax.barh(y, [s["n_stars_contributing_final_catalog"] for s in surveys])
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set(xscale="log", xlabel="stars contributing to the merged catalog")
    ax.invert_yaxis()

    ax = axes[1, 1]
    ax.barh(
        y, [100 * (s["rejected_frac_multiply_observed"] or 0) for s in surveys], color="C3"
    )
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.set(xlabel="% of entries rejected (stars with >= 2 entries)")
    ax.invert_yaxis()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


# --- Driver ---------------------------------------------------------------------
def apply_run(
    run_dir: Path,
    out_dir: Optional[Path] = None,
    *,
    dedupe: str = "none",
    eps_err: float = EPS_ERR,
    chi2_max: float = CHI2_MAX,
    n_iter: int = N_ITER,
    batch_size: int = 4096,
    only: Optional[Sequence[str]] = None,
    reuse_transform: bool = False,
) -> Dict[str, Path]:
    """Transform and merge all catalogs of a trained run; return the output paths."""
    from .model import load_bijector
    from .model.io import enable_memory_growth
    from .training_data import load_tables

    enable_memory_growth()

    run_dir = Path(run_dir)
    out_dir = Path(out_dir) if out_dir else run_dir / "harmonized"
    run_keys = registry.run_keys(run_dir)
    keys = list(run_keys) if only is None else list(only)
    unknown = [k for k in keys if k not in run_keys]
    if unknown:
        raise ValueError(f"Catalogs {unknown} were not part of the run (it has {list(run_keys)}).")
    indices = [registry.category_index(k) for k in keys]
    if tuple(registry.keys()[: len(run_keys)]) != tuple(run_keys):
        raise RuntimeError("The registry order no longer matches the run's catalog list.")

    log.info("Applying %s to %d catalogs", run_dir, len(keys))
    tables = load_tables(keys, dedupe=dedupe)
    cache_dir = out_dir / f"transform_cache_{dedupe}"
    transform = None

    estimates = []
    for key, idx, t in zip(keys, indices, tables):
        e = load_cached(cache_dir, idx, np.asarray(t["id"])) if reuse_transform else None
        if e is None:
            if transform is None:
                transform = make_transform(load_bijector(run_dir))
            e = transform_catalog(transform, t, idx, batch_size, desc=key)
            save_cached(cache_dir, e)
        else:
            log.info("%s: using the cached transformation", key)
        estimates.append(e)

    merged = combine(estimates, eps_err=eps_err, chi2_max=chi2_max, n_iter=n_iter)
    mask, counts = clean_mask(merged)
    log.info("Merged catalog: %s", counts)

    cfg = {}
    cfg_path = run_dir / "config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = json.load(f)
    metadata = dict(
        schema_version=SCHEMA_VERSION,
        created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        description="Harmonized stellar parameters, one estimate per Gaia DR3 source.",
        run_dir=str(run_dir),
        parameters=list(PARAM_NAMES),
        parameter_transform="Teff = 5000 K * exp(x[0] / 5)",
        missing_parameter_convention=(
            "value 0, covariance diagonal 1e9, zero off-diagonal; see params_measured"
        ),
        columns=COLUMN_DOC,
        merge=dict(
            eps_err=eps_err, chi2_max=chi2_max, n_iter=n_iter, dedupe=dedupe,
            param_lims=[list(l) for l in PARAM_LIMS],
            rejected_history=merged.rejected_history, **counts,
        ),
        training=cfg,
    )
    metadata["surveys"] = survey_info(keys, indices, tables, estimates, merged, mask)

    paths = write_outputs(out_dir, merged, estimates, tables, metadata, mask)
    t, _ = read_merged(paths["merged"])
    paths["summary"] = plot_summary(out_dir / "summary.png", t, metadata["surveys"])
    for p in paths.values():
        log.info("Wrote %s", p)
    return paths


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="astrotune-apply",
        description="Harmonize every catalog with a trained model and merge by Gaia source id.",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None, help="default: <run-dir>/harmonized")
    parser.add_argument("--dedupe", choices=("none", "first"), default="none",
                        help="'first' keeps one row per Gaia id in every catalog (default: keep all)")
    parser.add_argument("--eps-err", type=float, default=EPS_ERR)
    parser.add_argument("--chi2-max", type=float, default=CHI2_MAX)
    parser.add_argument("--n-iter", type=int, default=N_ITER)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--reuse-transform", action="store_true",
                        help="reuse the cached transformed estimates in <out-dir>/transform_cache_* "
                        "(written by every run), so merge settings can be changed quickly; "
                        "only valid for the same model and --dedupe")
    parser.add_argument("--only", nargs="+", metavar="KEY", default=None,
                        help="only these catalogs (for quick tests)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    apply_run(
        args.run_dir, args.out_dir, dedupe=args.dedupe, eps_err=args.eps_err,
        chi2_max=args.chi2_max, n_iter=args.n_iter, batch_size=args.batch_size, only=args.only,
        reuse_transform=args.reuse_transform,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
