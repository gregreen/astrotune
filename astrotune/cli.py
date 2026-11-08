"""Command-line interface: ``astrotune-prepare``."""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Optional, Sequence

from . import registry
from .prep.pipeline import prepare

log = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="astrotune-prepare",
        description="Build per-catalog training tables for Astrotune.",
    )
    parser.add_argument(
        "catalogs",
        nargs="*",
        metavar="CATALOG",
        help="catalog keys to prepare (see --list)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="prepare every registered catalog, in registry order",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="list the registered catalogs and exit",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="skip diagnostic figures",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-download raw files and rebuild cached crossmatches",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="build the table but do not write it to data/training/",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="only report warnings and errors",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.list:
        print(f"{'idx':>3}  {'key':<24} {'n_par':>5}  label")
        for i, spec in enumerate(registry.CATALOGS):
            print(f"{i:>3}  {spec.key:<24} {spec.n_params:>5}  {spec.label}")
        for spec in registry.CANDIDATES:
            print(f"{'-':>3}  {spec.key:<24} {spec.n_params:>5}  {spec.label}  (candidate: not in the model)")
        return 0

    if args.all:
        selected = list(registry.keys())
    else:
        selected = list(args.catalogs)

    if not selected:
        build_parser().print_help()
        return 2

    unknown = [k for k in selected if k not in registry.INDEX and k not in registry.candidate_keys()]
    if unknown:
        print(f"Unknown catalog(s): {', '.join(unknown)}", file=sys.stderr)
        print(f"Known: {', '.join(registry.keys() + registry.candidate_keys())}", file=sys.stderr)
        return 2

    failures = {}
    for key in selected:
        try:
            prepare(
                key,
                make_plots=not args.no_plots,
                force=args.force,
                save=not args.no_save,
            )
        except Exception as exc:  # noqa: BLE001 - report and continue
            log.exception("[%s] failed", key)
            failures[key] = exc

    if failures:
        print(f"\n{len(failures)} catalog(s) failed: {', '.join(failures)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
