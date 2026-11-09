"""Saving, loading, validating and cross-checking catalog tables."""

from __future__ import annotations

import json
import logging
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
from astropy.table import Table

from .. import config

log = logging.getLogger(__name__)

REQUIRED_COLUMNS = ("id", "x", "sigma_x", "x_err")


# --- Saving -----------------------------------------------------------------
def save_catalog(table: Table, key: str, *, meta: Optional[dict] = None) -> Path:
    """Write ``data/training/<key>/<key>.h5`` plus a JSON sidecar."""
    path = config.catalog_path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.write(path, serialize_metadata=True, overwrite=True)

    payload = {
        "catalog": key,
        "path": str(path),
        "n_rows": len(table),
        "columns": list(table.colnames),
        "n_params": int(table["x"].shape[1]),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": _git_commit(),
    }
    if meta:
        payload.update(meta)

    meta_file = config.meta_path(key)
    with open(meta_file, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return path


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=config.PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() or None
    except Exception:  # pragma: no cover - git may be unavailable
        return None


# --- Loading ----------------------------------------------------------------
def load_catalog(key: str, *, legacy: bool = False) -> Table:
    """Load a prepared catalog table.

    With ``legacy=True`` the table produced by the original notebook is read
    from ``training_data/`` instead; this is only used by the parity checks.
    """
    if legacy:
        path = config.LEGACY_TRAINING_DIR / f"{key}.h5"
    else:
        path = config.catalog_path(key)
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist; run astrotune-prepare first.")
    return Table.read(path)


def load_meta(key: str) -> dict:
    with open(config.meta_path(key)) as f:
        return json.load(f)


# --- Validation -------------------------------------------------------------
def validate_table(
    table: Table,
    *,
    key: Optional[str] = None,
    n_params: Optional[int] = None,
    strict_ids: bool = False,
) -> list[str]:
    """Check the on-disk contract; return a list of problems found."""
    problems: list[str] = []
    prefix = f"{key}: " if key else ""

    missing = [c for c in REQUIRED_COLUMNS if c not in table.colnames]
    if missing:
        problems.append(f"{prefix}missing columns {missing}")
        return problems

    ids = np.asarray(table["id"])
    x = np.asarray(table["x"])
    sigma_x = np.asarray(table["sigma_x"])
    x_err = np.asarray(table["x_err"])

    if ids.dtype.kind != "i":
        problems.append(f"{prefix}id dtype is {ids.dtype}, expected an integer type")
    if len(ids) == 0:
        problems.append(f"{prefix}table is empty")
        return problems
    if np.any(ids <= 0):
        n_bad = int(np.count_nonzero(ids <= 0))
        problems.append(f"{prefix}{n_bad} rows have id <= 0")

    if x.ndim != 2:
        problems.append(f"{prefix}x is not 2-dimensional")
        return problems
    if n_params is not None and x.shape[1] != n_params:
        problems.append(f"{prefix}x has {x.shape[1]} parameters, expected {n_params}")
    for name, array in (("sigma_x", sigma_x), ("x_err", x_err)):
        if array.shape != x.shape:
            problems.append(f"{prefix}{name} shape {array.shape} != x shape {x.shape}")

    if not np.all(np.isfinite(x)):
        problems.append(f"{prefix}x contains non-finite values")
    if not np.all(np.isfinite(sigma_x)):
        problems.append(f"{prefix}sigma_x contains non-finite values")
    if not np.all(np.isfinite(x_err)):
        problems.append(f"{prefix}x_err contains non-finite values")
    elif np.any(x_err <= 0):
        problems.append(f"{prefix}x_err has non-positive entries")

    n_dup = len(ids) - len(np.unique(ids))
    if n_dup:
        message = f"{prefix}{n_dup} duplicate identifiers"
        if strict_ids:
            problems.append(message)
        else:
            log.warning("%s (allowed if the catalog dedupes intentionally)", message)

    for col in table.colnames:
        if col in REQUIRED_COLUMNS:
            continue
        if not np.all(np.isfinite(np.asarray(table[col], dtype="f8"))):
            log.warning("%s[%s] contains non-finite values", prefix, col)

    return problems


# --- Parity against the legacy notebook output ------------------------------
def restore_legacy_alpha_floor(table: Table) -> Table:
    """A copy of ``table`` whose [alpha/Fe] floor is the legacy one (0.05, not the current one).

    The floor was lowered on purpose (:data:`astrotune.config.ERROR_FLOORS`); undoing it
    lets everything else in ``sigma_x`` still be compared with the legacy tables.
    """
    out = table.copy()
    if out["sigma_x"].shape[1] == config.N_PARAMS:
        s = np.asarray(out["sigma_x"][:, 3], dtype="f8")
        ok = s < 1e6  # not a "not measured" placeholder
        s[ok] = np.sqrt(np.maximum(s[ok] ** 2 - config.ERROR_FLOORS[3] ** 2 + config.LEGACY_ALPHA_FLOOR**2, 0.0))
        out["sigma_x"][:, 3] = s
    return out


def parity_report(table: Table, reference: Table, *, n_percentiles=(1.0, 50.0, 99.0)) -> dict:
    """Compare a freshly built table with the legacy notebook output."""
    report: dict[str, Any] = {
        "n_new": len(table),
        "n_ref": len(reference),
    }

    ids_new = np.asarray(table["id"])
    ids_ref = np.asarray(reference["id"])
    common = np.intersect1d(ids_new, ids_ref)

    report["n_common"] = int(len(common))
    report["n_only_new"] = int(len(np.setdiff1d(ids_new, ids_ref)))
    report["n_only_ref"] = int(len(np.setdiff1d(ids_ref, ids_new)))
    report["ids_match"] = bool(
        len(ids_new) == len(ids_ref) and np.array_equal(np.sort(ids_new), np.sort(ids_ref))
    )
    # Row order is expected to be preserved, which makes duplicates unambiguous.
    report["same_order"] = bool(
        len(ids_new) == len(ids_ref) and np.array_equal(ids_new, ids_ref)
    )

    if report["ids_match"] and (report["same_order"] or not n_dup_any(reference)):
        if report["same_order"]:
            new_aligned, ref_aligned = table, reference
        else:
            # Sort both tables the same way; stable sorting keeps duplicate
            # rows in their original relative order.
            new_aligned = table[np.argsort(ids_new, kind="stable")]
            ref_aligned = reference[np.argsort(ids_ref, kind="stable")]

        stats = {}
        for col in REQUIRED_COLUMNS[1:]:
            a = np.asarray(new_aligned[col], dtype="f8")
            b = np.asarray(ref_aligned[col], dtype="f8")
            stats[col] = {
                "max_abs_diff": float(np.max(np.abs(a - b))) if a.size else 0.0,
                "pct_new": np.percentile(a, n_percentiles, axis=0).tolist(),
                "pct_ref": np.percentile(b, n_percentiles, axis=0).tolist(),
            }
        report["columns"] = stats

    return report


def n_dup_any(table: Table) -> bool:
    ids = np.asarray(table["id"])
    return len(ids) != len(np.unique(ids))


def format_parity(report: dict) -> str:
    lines = [
        f"rows: {report['n_new']:,} new vs {report['n_ref']:,} legacy",
        f"ids: {report['n_common']:,} shared, "
        f"{report['n_only_new']:,} only-new, {report['n_only_ref']:,} only-legacy",
        f"identical id sets: {report['ids_match']}, same row order: {report.get('same_order')}",
    ]
    for col, stats in report.get("columns", {}).items():
        lines.append(f"  {col}: max |diff| = {stats['max_abs_diff']:.3g}")
    return "\n".join(lines)
