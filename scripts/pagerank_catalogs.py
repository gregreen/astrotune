#!/usr/bin/env python
"""Print the PageRank ranking of the catalog overlap network.

    python scripts/pagerank_catalogs.py
    python scripts/pagerank_catalogs.py --recount   # rebuild the counts first

The nodes are the prepared catalogs (``astrotune.registry.CATALOGS`` plus any
prepared candidate). An undirected edge connects every pair of catalogs that
share at least one star, and its weight is the number of stars they share
(Gaia DR3 id) -- the full network, not the thinned subset drawn in
``catalog_graph_spring.png``.

The counts are the ones ``astrotune-catalog-graph`` caches in
``diagnostic_plots/catalog_graph/catalog_overlaps.json``, and the ranking is the
one that command writes to ``catalog_pagerank.ecsv``; this script only prints
it. Without the cache (or with ``--recount``) the counts are rebuilt from the
prepared tables.

Three rankings are printed in turn: all catalogs, only the catalogs that
measure [alpha/Fe], and only the catalogs that measure log g. A subset keeps
the same shared-star weights but is ranked on the subgraph it induces, with the
teleport distribution spread over the catalogs in that subset only.
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Optional, Sequence

import numpy as np

from astrotune import catalog_graph as cg
from astrotune import config, registry

log = logging.getLogger(__name__)

#: Cache written by ``astrotune-catalog-graph``, holding the shared-star counts.
OVERLAPS_JSON = config.PROJECT_ROOT / "diagnostic_plots" / "catalog_graph" / "catalog_overlaps.json"


def load_overlaps(recount: bool = False) -> dict:
    """The counts behind the figure, from its JSON cache when that is still current."""
    if OVERLAPS_JSON.exists() and not recount:
        data = json.load(open(OVERLAPS_JSON))
        if data.get("keys") == list(registry.keys()) and "n_measured" in data:
            log.info("using %s", OVERLAPS_JSON)
            return data
    data = cg.compute_overlaps()
    OVERLAPS_JSON.parent.mkdir(parents=True, exist_ok=True)
    json.dump(data, open(OVERLAPS_JSON, "w"))
    log.info("recomputed the counts into %s", OVERLAPS_JSON)
    return data


def print_ranking(keys: Sequence[str], labels: Sequence[str], scores: np.ndarray,
                  title: str, n_edges: int, n_total: int) -> None:
    n = len(keys)
    print()
    print("=" * 78)
    head = f"PageRank, {title}"
    if n < n_total:
        head += f"  ({n} of {n_total} catalogs)"
    print(head)
    print(f"weighted by shared Gaia DR3 stars; {n_edges} edges; score sums to 1 over these {n} nodes")
    print("=" * 78)
    width = max(len(label) for label in labels)
    print(f"{'#':>3}  {'catalog':<{width}}  {'PageRank':>10}")
    print("-" * (5 + width + 12))
    for rank, k in enumerate(np.argsort(-scores), start=1):
        print(f"{rank:>3}  {labels[k]:<{width}}  {scores[k]:>10.6f}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="pagerank-catalogs", description=__doc__.split("\n\n")[0])
    parser.add_argument("--damping", type=float, default=cg.PAGERANK_DAMPING, help="PageRank damping factor")
    parser.add_argument("--recount", action="store_true", help="recompute the overlap counts instead of reusing the JSON")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    data = load_overlaps(args.recount)
    ranks = cg.compute_pagerank(data, args.damping)
    label_of = dict(zip(data["keys"], data["labels"]))
    for param, title in cg.PAGERANK_SUBSETS:
        res = ranks[param or "all"]
        print_ranking(res["keys"], [label_of[k] for k in res["keys"]], res["score"], title,
                      res["n_edges"], len(data["keys"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
