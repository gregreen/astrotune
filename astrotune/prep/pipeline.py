"""Orchestration of the catalog preparation step."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from astropy.table import Table

from .. import config, registry
from . import io, plots, transforms

log = logging.getLogger(__name__)


@dataclass
class Context:
    """Everything a catalog module needs to know about where things live."""

    key: str
    spec: registry.CatalogSpec
    raw_dir: Path
    interim_dir: Path
    training_dir: Path
    diagnostic_dir: Path
    make_plots: bool = True
    force: bool = False
    extra: dict = field(default_factory=dict)

    @property
    def n_params(self) -> int:
        return self.spec.n_params


def make_context(key: str, *, make_plots: bool = True, force: bool = False) -> Context:
    spec = registry.get_spec(key)
    return Context(
        key=key,
        spec=spec,
        raw_dir=config.raw_dir(key),
        interim_dir=config.interim_dir(key),
        training_dir=config.training_dir(key),
        diagnostic_dir=config.diagnostic_dir(key),
        make_plots=make_plots,
        force=force,
    )


def evaluate_cuts(t: Table, module, ctx: Context) -> tuple[np.ndarray, dict[str, int]]:
    """Combine the catalog's cuts and record how many stars survive each one."""
    if hasattr(module, "cuts"):
        named = module.cuts(t, ctx)
        mask = np.ones(len(t), dtype=bool)
        counts: dict[str, int] = {}
        for name, cut in named.items():
            mask &= np.asarray(cut, dtype=bool)
            counts[name] = int(np.count_nonzero(mask))
        return mask, counts

    mask = np.asarray(module.select(t, ctx), dtype=bool)
    return mask, {"selected": int(np.count_nonzero(mask))}


def build_table(t: Table, mask: np.ndarray, module, ctx: Context) -> Table:
    """Assemble the standardized table, applying transformations and floors."""
    d = len(module.PARAM_COLS)
    floors = getattr(module, "ERROR_FLOORS", transforms.default_floors(d))

    param_errs = None
    if hasattr(module, "param_errs"):
        param_errs = module.param_errs(t, ctx)

    return transforms.extract_table(
        t,
        module.ID_COL,
        module.PARAM_COLS,
        module.UNC_COLS,
        getattr(module, "EXTRA_COLS", ()),
        param_transf=transforms.PARAM_TRANSF[:d],
        param_unc_transf=transforms.apply_err_floors(
            transforms.PARAM_UNC_TRANSF[:d], floors[:d]
        ),
        param_errs=param_errs,
        mask=mask,
    )


def prepare(
    key: str,
    *,
    make_plots: bool = True,
    force: bool = False,
    save: bool = True,
) -> Table:
    """Run the full preparation chain for one catalog.

    Steps: download -> load -> derive columns -> quality cuts -> transform and
    assemble -> validate -> save -> diagnostics.
    """
    ctx = make_context(key, make_plots=make_plots, force=force)
    module = registry.get_module(key)
    t0 = time.monotonic()

    log.info("[%s] %s", key, ctx.spec.label)
    if hasattr(module, "download"):
        module.download(ctx)

    t = module.load(ctx)
    n_input = len(t)
    log.info("[%s] loaded %s stars", key, f"{n_input:,}")

    if hasattr(module, "prepare_columns"):
        t = module.prepare_columns(t, ctx)

    mask, cut_counts = evaluate_cuts(t, module, ctx)
    log.info(
        "[%s] %s stars survive cuts (%.2f%%)",
        key,
        f"{int(np.count_nonzero(mask)):,}",
        100 * np.mean(mask) if n_input else 0.0,
    )

    table = build_table(t, mask, module, ctx)

    problems = io.validate_table(table, key=key, n_params=module.N_PARAMS)
    if problems:
        raise ValueError("Invalid prepared table:\n  " + "\n  ".join(problems))
    log.info("[%s] table valid: %s stars, %d params", key, f"{len(table):,}", table["x"].shape[1])

    meta = {
        "label": ctx.spec.label,
        "n_params": module.N_PARAMS,
        "reference": getattr(module, "REFERENCE", ""),
        "sources": list(getattr(module, "SOURCES", ())),
        "n_input_rows": int(n_input),
        "n_selected_rows": int(len(table)),
        "cut_counts": cut_counts,
    }

    if save:
        path = io.save_catalog(table, key, meta=meta)
        log.info("[%s] wrote %s", key, path)

    if make_plots:
        _write_diagnostics(t, mask, table, module, ctx, cut_counts)

    _report_parity(key, table)

    log.info("[%s] done in %.1f s", key, time.monotonic() - t0)
    return table


def _write_diagnostics(t, mask, table, module, ctx: Context, cut_counts: dict) -> None:
    plots.setup_style()
    written = []

    counts = {"input": int(len(t))}
    counts.update(cut_counts)
    written.append(plots.plot_cut_flow(counts, ctx.diagnostic_dir))

    written.append(plots.plot_catalog_params(table, ctx.diagnostic_dir))
    written.append(plots.plot_catalog_uncertainties(table, ctx.diagnostic_dir))
    written.append(plots.plot_kiel(table, ctx.diagnostic_dir))
    written.append(plots.plot_feh_alpha(table, ctx.diagnostic_dir))
    written.append(plots.plot_sky(table, ctx.diagnostic_dir))

    if hasattr(module, "diagnostics"):
        extra = module.diagnostics(t, mask, table, ctx)
        written.extend([p for p in (extra or []) if p is not None])

    log.info(
        "[%s] wrote %d diagnostic figure(s) to %s",
        ctx.key,
        len([p for p in written if p is not None]),
        ctx.diagnostic_dir,
    )


def _report_parity(key: str, table: Table) -> None:
    """Compare with the legacy notebook output when it is available."""
    legacy = config.LEGACY_TRAINING_DIR / f"{key}.h5"
    if not legacy.exists():
        return
    try:
        reference = Table.read(legacy)
    except Exception as exc:  # pragma: no cover - unreadable legacy file
        log.warning("[%s] could not read legacy table %s: %s", key, legacy, exc)
        return
    report = io.parity_report(table, reference)
    for line in io.format_parity(report).splitlines():
        log.info("[%s] parity | %s", key, line)
