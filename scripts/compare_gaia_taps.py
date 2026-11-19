#!/usr/bin/env python
"""Compare the Gaia archives on the same crossmatch job.

    python scripts/compare_gaia_taps.py --repeats 5
    python scripts/compare_gaia_taps.py --services esa,gavo --catalog rave_dr6 --max-rows 200000

The official ESA archive has been answering with HTTP 500s and with aborted
jobs while it is being prepared for DR4, so this script runs the *same*
crossmatch against each service, repeatedly, and reports how often it worked,
how many stars it recovered and how long it took. The reference is the legacy
notebook table (``training_data/<key>.h5``), which was produced by the ESA
archive when it was healthy: any service that recovers fewer of those Gaia ids
is losing stars.

Each run goes through :func:`astrotune.prep.gaia.gaia_crossmatch_batched`, so it
exercises the same retry/validation/audit path that the catalog preparation
uses, and the per-batch accounting is reported alongside.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from astropy.table import Table

from astrotune import config
from astrotune.prep import gaia

#: Coordinates of catalogs that ship no Gaia ids, i.e. what actually needs a
#: crossmatch. The reader returns (table, ra column, dec column).
READERS = {
    "vmp_stars_li2022": lambda: (
        Table.read(
            config.raw_dir("vmp_stars_li2022") / "table2.dat",
            format="cds",
            readme=str(config.raw_dir("vmp_stars_li2022") / "readme.txt"),
        ),
        "RAdeg",
        "DEdeg",
    ),
    "lamost_csn_dr10": lambda: (
        Table.read(config.raw_dir("lamost_csn_dr10") / "DR10_Cycle-SN_M_dwarfs.csv"),
        "RA",
        "Dec",
    ),
    "lamost_ddp_dr9": lambda: (
        Table.read(config.raw_dir("lamost_ddp_dr9") / "LMDR9_DDPAYNE_recommend_202505.fits"),
        "RA",
        "DEC",
    ),
}


def load_coordinates(catalog: str, max_rows: Optional[int]):
    table, ra_col, dec_col = READERS[catalog]()
    if max_rows is not None:
        table = table[:max_rows]
    return (
        np.asarray(table[ra_col], dtype="f8"),
        np.asarray(table[dec_col], dtype="f8"),
    )


def legacy_ids(catalog: str) -> set:
    path = config.LEGACY_TRAINING_DIR / f"{catalog}.h5"
    if not path.exists():
        return set()
    return {int(x) for x in np.asarray(Table.read(path)["id"])}


def run_once(archive: gaia.Archive, ra, dec, *, separation, batch_size, timeout, method):
    started = time.monotonic()
    try:
        crossmatch = gaia.gaia_crossmatch_batched(
            ra,
            dec,
            max_separation=separation,
            max_batch_size=batch_size,
            archive=archive,
            method=method,
            timeout=timeout,
        )
    except gaia.CrossmatchError as exc:
        return {
            "ok": False,
            "seconds": time.monotonic() - started,
            "error": " ".join(str(exc).split())[:300],
        }

    meta = crossmatch.meta
    return {
        "ok": True,
        "seconds": time.monotonic() - started,
        "n_matched": meta["n_matched"],
        "n_uploaded": meta["n_uploaded"],
        "match_fraction": meta["match_fraction"],
        "attempts": [b["attempts"] for b in meta["batches"]],
        "suspect_batches": meta["suspect_batches"],
        "ids": {int(x) for x in np.asarray(crossmatch["gdr3_source_id"])},
    }


def summarise(name: str, runs: list[dict], reference: set) -> dict:
    ok = [r for r in runs if r["ok"]]
    n_matched = [r["n_matched"] for r in ok]
    recovered = [len(r["ids"] & reference) for r in ok]
    seconds = [r["seconds"] for r in runs]

    row = {
        "service": name,
        "runs": len(runs),
        "ok": len(ok),
        "matched": n_matched,
        "recovered": recovered,
        "seconds": [round(s, 1) for s in seconds],
        "errors": [r["error"] for r in runs if not r["ok"]],
    }
    print(f"\n{name}: {len(ok)}/{len(runs)} runs succeeded")
    if ok:
        counts = sorted(set(n_matched))
        print(
            f"  matched        {min(n_matched):,}-{max(n_matched):,}"
            f"  (median {statistics.median(n_matched):,.0f})  distinct values {counts}"
        )
        print(
            f"  legacy ids      {min(recovered):,}-{max(recovered):,} of {len(reference):,} recovered"
        )
        print(f"  wall time       {min(seconds):.1f}-{max(seconds):.1f} s  (all runs)")
    else:
        print(f"  wall time       {min(seconds):.1f}-{max(seconds):.1f} s")
    for message in row["errors"][:3]:
        print(f"  error: {message}")
    return row


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="compare_gaia_taps", description=__doc__.split("\n\n")[0]
    )
    parser.add_argument("--services", default="esa,gavo", help="comma-separated archive names")
    parser.add_argument("--catalog", default="vmp_stars_li2022", choices=sorted(READERS))
    parser.add_argument("--repeats", type=int, default=3, help="runs per service")
    parser.add_argument("--max-rows", type=int, default=None, help="use only the first N rows")
    parser.add_argument("--batch-size", type=int, default=gaia.DEFAULT_MAX_BATCH_SIZE)
    parser.add_argument(
        "--method",
        choices=("distance", "contains"),
        default="distance",
        help="ADQL geometry form: DISTANCE(...) < sep, or CONTAINS(circle) on the index",
    )
    parser.add_argument("--separation", type=float, default=0.5, help="arcsec")
    parser.add_argument("--timeout", type=float, default=gaia.JOB_TIMEOUT, help="seconds per batch")
    parser.add_argument("--out", type=Path, default=None, help="write the results as JSON")
    args = parser.parse_args(argv)

    ra, dec = load_coordinates(args.catalog, args.max_rows)
    reference = legacy_ids(args.catalog)
    print(
        f"{args.catalog}: {len(ra):,} coordinates, {len(reference):,} legacy Gaia ids, "
        f"{args.repeats} run(s) per service, batches of {args.batch_size:,}, "
        f"method={args.method}"
    )

    import astropy.units as units

    separation = args.separation * units.arcsec
    results = {}
    for name in [s.strip() for s in args.services.split(",") if s.strip()]:
        archive = gaia.archive_by_name(name)
        print(f"\n--- {archive.name}: {archive.table} on {archive.url or 'astroquery.gaia'}")
        if archive.note:
            print(f"    ({archive.note})")
        runs = []
        for i in range(args.repeats):
            run = run_once(
                archive,
                ra,
                dec,
                separation=separation,
                batch_size=args.batch_size,
                timeout=args.timeout,
                method=args.method,
            )
            runs.append(run)
            status = "ok" if run["ok"] else "FAILED"
            print(f"  run {i + 1}/{args.repeats}: {status} in {run['seconds']:.1f} s")
        results[archive.name] = summarise(archive.name, runs, reference)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(
                {"catalog": args.catalog, "n_coordinates": len(ra), "services": results},
                f,
                indent=2,
                default=str,
            )
        print(f"\nwrote {args.out}")

    reliable = [r for r in results.values() if r["ok"] == r["runs"]]
    best = max(results.values(), key=lambda r: (min(r["matched"], default=0), r["ok"]))
    print("\n" + ("=" * 60))
    if len(reliable) == 1:
        print(f"only {reliable[0]['service']} completed every run")
    elif not reliable:
        print("no service completed every run")
    else:
        print(f"every run succeeded on: {', '.join(r['service'] for r in reliable)}")
    if best["matched"]:
        print(f"most stars recovered: {best['service']} ({max(best['recovered']):,} legacy ids)")
    return 0 if all(r["ok"] for r in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
