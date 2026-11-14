"""Figure: how big is each catalog and how are the catalogs connected?

    astrotune-catalog-graph            # -> diagnostic_plots/catalog_graph/

*Nodes* are the catalogs in ``registry.CATALOGS``, plus any candidate that has been
prepared; candidates are outlined with a dashed ring, since they are not part of the
model yet. The circle's radius grows with
``log10(number of rows)`` and its color marks the family of spectra the catalog
is based on. The stroke around a node marks a missing parameter: white if the catalog
measures all four, one color if it has no [alpha/Fe] and another if it has no logg
(hatched, alternating with white, if only some of its stars lack the parameter; see
:func:`node_strokes`). Labels are placed next to the nodes, preferably inside their bounding
box so that they do not shrink the figure; the labels of the two largest nodes
(``INSIDE_LABELS``) are written inside them. *Edges* join catalogs that share stars (matched on Gaia DR3 id);
line width, opacity and color (``plasma``) grow with ``log10(number of shared stars)``. Edges below
``--min-shared`` are not drawn (the graph is nearly complete, so every pair
would otherwise be an edge), except that each catalog always keeps its ``--top-k``
strongest connections, so that no catalog is left without an edge.

Layout. With about 80 % of all pairs connected, no planar layout exists and
minimizing the number of crossings is NP-hard, so the layout is a heuristic: many
restarts of a weighted force-directed layout (strongly connected catalogs are
pulled together), keeping the one with the smallest *weighted* crossing number
(crossings between thick edges cost more). ``--layout circle`` instead puts the
nodes on a circle and minimizes the weighted chord crossings over the node order
by simulated annealing. The companion matrix figure shows every pair, including
those that are too faint for the graph.

The counts (rows, unique stars, shared stars) are written to
``catalog_overlaps.json`` and are independent of any trained model.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

if not os.environ.get("DISPLAY"):  # headless by default
    import matplotlib

    matplotlib.use("Agg")

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from . import config, registry
from .prep.io import load_catalog

log = logging.getLogger(__name__)

#: Which spectra a catalog is based on (nodes of one family share a color)
FAMILIES: Dict[str, Tuple[str, ...]] = {
    "APOGEE": ("apogee_dr17", "sdssv_aspcap", "sdssv_apogeenet"),
    "BOSS": ("sdssv_bossnet", "sdssv_slam", "sdssv_mdc", "sdssv_clam"),
    "LAMOST": (
        "lamost_mrs", "lamost_hotpayne", "lamost_ddp_dr9", "lamost_mstar_ding2022",
        "vmp_stars_li2022", "lamost_csn_dr10", "lamost_mstar_li2025",
    ),
    "DESI": ("desi_dr1", "desi_ddp", "desi_nagaraj2026"),
    "GALAH": ("galah_dr4",),
    "RAVE": ("rave_dr6",),
    "PIGS (AAT)": ("pristine_pigs",),
}

#: The family colors come from distinctipy, optimized to stay distinguishable under simulated
#: deuteranomaly (the most common color-vision deficiency, red-green). The seed was chosen from
#: 60 by the largest minimum pairwise distance (in the simulated space) among palettes that are
#: not too light for a white background; a test checks the distance. The mild pastel factor trades
#: some separation (min distance 0.52 against 0.76 without it) for less garish colors.
COLORBLIND_TYPE = "Deuteranomaly"
PALETTE_PASTEL, PALETTE_SEED = 0.2, 34

#: Strokes of the nodes mark catalogs that lack a parameter. The two stroke colors are the next two
#: colors of the same distinctipy palette (the palette is generated one color at a time, so adding
#: colors does not change the family colors).
STROKE_PARAMS = ("alpha", "logg")
STROKE_LABELS = {"alpha": "no [α/Fe]", "logg": "no log g"}
#: Index of each parameter in the 4-parameter vector (Teff, [Fe/H], logg, [alpha/Fe])
PARAM_INDEX = {"logg": 2, "alpha": 3}


def palette(n: int) -> List[str]:
    from distinctipy import distinctipy

    cols = distinctipy.get_colors(
        n,
        exclude_colors=[(1, 1, 1), (0, 0, 0), (0.5, 0.5, 0.5)],  # not the page, the text or the edges
        pastel_factor=PALETTE_PASTEL, colorblind_type=COLORBLIND_TYPE, rng=PALETTE_SEED,
    )
    return [distinctipy.get_hex(c) for c in cols]


def family_colors(families: Sequence[str]) -> Dict[str, str]:
    return dict(zip(families, palette(len(families))))


_PALETTE = palette(len(FAMILIES) + len(STROKE_PARAMS))
FAMILY_COLORS = dict(zip(FAMILIES, _PALETTE))
STROKE_COLORS = dict(zip(STROKE_PARAMS, _PALETTE[len(FAMILIES):]))


def family_of(key: str) -> str:
    for fam, keys in FAMILIES.items():
        if key in keys:
            return fam
    return "other"


# --- Counts ---------------------------------------------------------------------
def measured_counts(table) -> List[int]:
    """Number of rows of a prepared table that measure each of the 4 parameters (0 for a
    parameter the catalog has no column for)."""
    x, s = np.asarray(table["x"]), np.asarray(table["sigma_x"])
    out = [0] * config.N_PARAMS
    for j in range(min(x.shape[1], config.N_PARAMS)):
        out[j] = int(((s[:, j] < 1e6) & np.isfinite(x[:, j])).sum())
    return out


def missing_kind(n_measured: int, n_rows: int) -> str:
    """``"none"`` (all rows measure the parameter), ``"all"`` (no row does) or ``"some"``."""
    if n_measured >= n_rows:
        return "none"
    return "all" if n_measured == 0 else "some"


def compute_overlaps(keys: Optional[Sequence[str]] = None) -> dict:
    """Rows, unique stars, the matrix of shared stars and the parameters measured by the
    prepared catalogs."""
    if keys is None:
        # Prepared candidates are included so that the figure shows the catalog set as it
        # actually stands; the plot marks them, so they cannot be mistaken for model inputs.
        keys = [
            k
            for k in registry.keys() + registry.candidate_keys()
            if config.catalog_path(k).exists()
        ]
    keys = list(keys)
    rows, ids, measured = [], [], []
    for k in keys:
        table = load_catalog(k)
        col = np.asarray(table["id"])
        rows.append(int(len(col)))
        ids.append(np.unique(col))
        measured.append(measured_counts(table))
    n = len(keys)
    shared = np.zeros((n, n), dtype="i8")
    for i, j in itertools.combinations(range(n), 2):
        shared[i, j] = shared[j, i] = len(np.intersect1d(ids[i], ids[j], assume_unique=True))
    return dict(
        keys=keys, labels=[registry.get_spec(k).label for k in keys], rows=rows,
        unique=[int(len(a)) for a in ids], shared=shared.tolist(), n_measured=measured,
    )


# --- Layout ---------------------------------------------------------------------
def edge_list(shared: np.ndarray, min_shared: int, top_k: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Edges to draw: every pair with at least ``min_shared`` shared stars, plus each
    catalog's ``top_k`` strongest connections (so that no catalog is left unconnected,
    however small). Pairs sharing no star are never drawn."""
    n = len(shared)
    keep = shared >= min_shared
    for a in range(n):
        order = np.argsort(-shared[a])
        order = order[(order != a) & (shared[a, order] > 0)][:top_k]
        keep[a, order] = keep[order, a] = True
    keep &= shared > 0
    i, j = np.triu_indices(n, 1)
    m = keep[i, j]
    return np.stack([i[m], j[m]], axis=1), shared[i, j][m]


def edge_strength(counts: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """0..1, linear in log10 of the number of shared stars."""
    return np.clip((np.log10(np.maximum(counts, 1)) - np.log10(lo)) / max(np.log10(hi) - np.log10(lo), 1e-9), 0, 1)


def segments_cross(p: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """``(m, m)`` boolean: do the straight edges i and j properly cross?"""
    a, b = p[edges[:, 0]], p[edges[:, 1]]

    def orient(u, v, w):
        return np.sign((v[..., 0] - u[..., 0]) * (w[..., 1] - u[..., 1]) - (v[..., 1] - u[..., 1]) * (w[..., 0] - u[..., 0]))

    A, B = a[:, None], b[:, None]
    C, D = a[None], b[None]
    cross = (orient(A, B, C) * orient(A, B, D) < 0) & (orient(C, D, A) * orient(C, D, B) < 0)
    share = (
        (edges[:, None, 0] == edges[None, :, 0]) | (edges[:, None, 0] == edges[None, :, 1])
        | (edges[:, None, 1] == edges[None, :, 0]) | (edges[:, None, 1] == edges[None, :, 1])
    )
    return cross & ~share


def crossing_score(p: np.ndarray, edges: np.ndarray, strength: np.ndarray) -> Tuple[int, float]:
    """``(number of crossings, strength-weighted number of crossings)``."""
    c = np.triu(segments_cross(p, edges), 1)
    return int(c.sum()), float((c * strength[:, None] * strength[None, :]).sum())


#: Half-width of the spring layout in data units (the circles have radii of 0.01-0.15, so the
#: larger this is, the more room between the nodes; the figure scales to fit)
SPREAD = 1.7


def dissimilarity(n: int, edges: np.ndarray, strength: np.ndarray) -> np.ndarray:
    """Target distance between two catalogs: short for strong edges, longer for weak
    ones, longest for pairs that are not connected."""
    d = np.full((n, n), 1.4)
    d[edges[:, 0], edges[:, 1]] = d[edges[:, 1], edges[:, 0]] = 0.55 + 0.65 * (1 - strength)
    np.fill_diagonal(d, 0)
    return d


def smacof(delta: np.ndarray, X: np.ndarray, iters: int = 300) -> np.ndarray:
    """Weighted stress majorization (weights 1/delta^2): distances approach ``delta``."""
    n = len(delta)
    W = np.where(delta > 0, 1.0 / np.maximum(delta, 1e-9) ** 2, 0.0)
    V = -W.copy()
    V[np.diag_indices(n)] = W.sum(1)
    Vp = np.linalg.pinv(V)
    for _ in range(iters):
        D = np.linalg.norm(X[:, None] - X[None], axis=-1) + np.eye(n)
        B = -W * delta / D
        B[np.diag_indices(n)] = 0
        B[np.diag_indices(n)] = -B.sum(1)
        X = Vp @ B @ X
    return X


def remove_overlaps(pos: np.ndarray, radii: np.ndarray, gap: float = 0.04, iters: int = 300) -> np.ndarray:
    """Push circles apart until none overlap (plus ``gap``)."""
    pos = pos.copy()
    n = len(pos)
    for _ in range(iters):
        delta = pos[:, None] - pos[None]
        dist = np.linalg.norm(delta, axis=-1) + np.eye(n)
        need = radii[:, None] + radii[None] + gap - dist
        np.fill_diagonal(need, 0)
        if need.max() <= 1e-4:
            break
        push = (np.maximum(need, 0)[..., None] * delta / dist[..., None]).sum(1) * 0.5
        pos += push
    return pos


def best_spring_layout(shared, radii, edges, strength, n_restarts, seed):
    """Stress-majorization layout; the restart with the smallest weighted crossing number wins."""
    rng = np.random.default_rng(seed)
    delta = dissimilarity(len(shared), edges, strength)
    best, best_score = None, (np.inf, np.inf)
    for _ in range(n_restarts):
        p = smacof(delta, rng.normal(size=(len(shared), 2)))
        p = p - p.mean(0)
        p = SPREAD * p / np.abs(p).max()
        p = remove_overlaps(p, radii)
        s = crossing_score(p, edges, strength)
        if s[1] < best_score[1]:
            best, best_score = p, s
    return best, best_score


def circle_layout(shared, edges, strength, seed, n_iter=40000):
    """Nodes on a circle; the order minimizes the weighted chord crossings (annealing)."""
    n = len(shared)
    rng = np.random.default_rng(seed)
    w = np.zeros((n, n))
    w[edges[:, 0], edges[:, 1]] = w[edges[:, 1], edges[:, 0]] = strength

    def cost(order):
        pos = np.empty(n, dtype=int)
        pos[order] = np.arange(n)
        a = np.minimum(pos[edges[:, 0]], pos[edges[:, 1]])
        b = np.maximum(pos[edges[:, 0]], pos[edges[:, 1]])
        cr = (a[:, None] < a[None]) & (a[None] < b[:, None]) & (b[:, None] < b[None])  # interleaved chords
        return float((cr * strength[:, None] * strength[None, :]).sum())

    order = rng.permutation(n)
    cur = cost(order)
    best, best_c = order.copy(), cur
    for it in range(n_iter):
        T = 2.0 * (1 - it / n_iter) + 1e-3
        i, j = rng.integers(0, n, 2)
        cand = order.copy()
        cand[i], cand[j] = cand[j], cand[i]
        c = cost(cand)
        if c < cur or rng.random() < np.exp((cur - c) / T):
            order, cur = cand, c
            if cur < best_c:
                best, best_c = order.copy(), cur
    ang = 2 * np.pi * np.arange(n) / n
    pos = np.empty((n, 2))
    pos[best] = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    return pos


# --- Drawing --------------------------------------------------------------------
#: Node radius = R_REF * (rows / ROWS_REF) ** NODE_EXPONENT: a power law, so the area grows as
#: rows**(2 * NODE_EXPONENT). 0.25 keeps 300 rows (VMP) visible next to 4 M (DD-Payne): 11x in radius.
ROWS_REF, R_REF, NODE_EXPONENT = 4.2e6, 0.22, 0.25


def node_radii(rows: np.ndarray) -> np.ndarray:
    return R_REF * (np.asarray(rows, float) / ROWS_REF) ** NODE_EXPONENT


def edge_width(s):
    """Line width (pt) for an edge strength ``s`` in 0..1 (steeper than linear)."""
    return 0.7 + 8.3 * np.asarray(s) ** 1.5


def edge_alpha(s):
    """Opacity: falls off quickly for edges with few shared stars (0.15 .. 0.95)."""
    return 0.15 + 0.80 * np.asarray(s) ** 2.2


#: Edge color: the ``plasma`` color map over the (log) number of shared stars. Its last, bright
#: yellow part is nearly invisible on white, so the map stops at this fraction.
EDGE_CMAP, EDGE_CMAP_MAX = "plasma", 0.88


def edge_color(s):
    """RGB of an edge with strength ``s`` (0..1): dark blue/purple (few stars) to orange (many)."""
    return plt.get_cmap(EDGE_CMAP)(EDGE_CMAP_MAX * float(s))[:3]


#: Width (pt) of the stroke around every node, and the length (pt) of the dashes of a hatched one
STROKE_PT, HATCH_PT = 2.0, 3.0

#: Catalogs whose label is written inside the node (large enough), with the text to show
#: (``None``: the catalog's label)
INSIDE_LABELS: Dict[str, Optional[str]] = {"desi_dr1": None, "lamost_ddp_dr9": "LAMOST\nDD-Payne\n(DR9)"}

#: Height (data units) that the legend needs, and its width from its left edge
LEGEND_H, LEGEND_W = 2.9, 1.0


def points_per_unit(ax) -> float:
    """Points (1/72 inch) per data unit along x, with the axes as they are now."""
    (x0, _), (x1, _) = ax.transData.transform([[0.0, 0.0], [1.0, 0.0]])
    return float(abs(x1 - x0) * 72.0 / ax.figure.dpi)


def text_color_on(color: str) -> str:
    """Black or white, whichever reads better on ``color`` (a hex string)."""
    r, g, b = (int(color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return "black" if 0.299 * r + 0.587 * g + 0.114 * b > 0.6 else "white"


def node_strokes(data: dict) -> List[Tuple[Optional[str], str]]:
    """``(parameter, kind)`` for every catalog: the first of ``STROKE_PARAMS`` that it lacks
    (``kind`` ``"all"``: no star measures it, ``"some"``: only some do), or ``(None, "none")``.
    Without ``n_measured`` in the data (counts from before it was recorded) nothing is marked."""
    n_rows, n_meas = data["rows"], data.get("n_measured")
    out = []
    for k in range(len(data["keys"])):
        found = (None, "none")
        for param in STROKE_PARAMS if n_meas is not None else ():
            kind = missing_kind(n_meas[k][PARAM_INDEX[param]], n_rows[k])
            if kind != "none":
                found = (param, kind)
                break
        out.append(found)
    return out


def draw_stroke(ax, center, radius: float, width_pt: float, ppu: float, param: Optional[str], kind: str) -> None:
    """A stroke of ``width_pt`` *outside* the circle (so that it does not eat into small nodes):
    white, and on top of it the color of ``param``, solid for ``kind == "all"`` and as alternating
    colored and white dashes (a whole number of them around the circle) for ``"some"``."""
    r = radius + 0.5 * width_pt / ppu
    ax.add_patch(Circle(center, r, fill=False, ec="white", lw=width_pt, zorder=3.1))
    if param is None:
        return
    kw = dict(fill=False, ec=STROKE_COLORS[param], lw=width_pt, zorder=3.2)
    if kind == "all":
        ax.add_patch(Circle(center, r, **kw))
        return
    circumference = 2 * np.pi * r * ppu
    n = max(4, int(round(circumference / (2 * HATCH_PT))))
    seg = circumference / (2 * n) / width_pt  # matplotlib scales dash lengths by the line width
    ax.add_patch(Circle(center, r, ls=(0, (seg, seg)), **kw))


def place_labels(fig, ax, texts, pos: np.ndarray, radii: np.ndarray, skip: Sequence[int] = (),
                 preferred: Optional[Tuple[float, float, float, float]] = None,
                 limit: Optional[Tuple[float, float, float, float]] = None) -> List[Tuple[float, float, float, float]]:
    """Put every label next to its circle without covering any circle or other label.

    Larger circles first; for each label, candidate positions on rings around the
    circle (starting radially outward) are tried until one is free. The labels of the nodes in
    ``skip`` are left where they are (they sit inside their circles).

    ``preferred`` is a box ``(x0, y0, x1, y1)`` in data units, normally the bounding box of the
    nodes: a free position inside it is always taken over one that sticks out, and of the free
    positions that do stick out, the one that sticks out least is taken (so that labels do not
    make the figure, which is scaled to fit them, any bigger than the nodes need). ``limit`` is a
    hard box (default: the axis limits) that no label may leave.

    Returns the boxes ``(x0, y0, x1, y1)`` of the placed labels.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    if limit is None:
        limit = (*ax.get_xlim(), *ax.get_ylim())
        limit = (limit[0], limit[2], limit[1], limit[3])
    if preferred is None:
        preferred = limit
    boxes = [(p[0] - r, p[1] - r, p[0] + r, p[1] + r) for p, r in zip(pos, radii)]
    center = pos.mean(0)
    skip = set(skip)
    placed = []

    def overlaps(b, others):
        return any(b[0] < o[2] and b[2] > o[0] and b[1] < o[3] and b[3] > o[1] for o in others)

    def protrusion(b):
        return (max(0.0, preferred[0] - b[0]) + max(0.0, preferred[1] - b[1])
                + max(0.0, b[2] - preferred[2]) + max(0.0, b[3] - preferred[3]))

    for k in np.argsort(-radii):
        if k in skip:
            continue
        t = texts[k]
        out = pos[k] - center
        base = np.arctan2(out[1], out[0])
        others = [b for i, b in enumerate(boxes) if i != k]
        best = None  # (protrusion, setting, box) of the best free candidate so far
        first = None
        for ring in (0.015, 0.05, 0.1, 0.17, 0.26):
            for da in np.deg2rad([0, 25, -25, 50, -50, 80, -80, 110, -110, 150, -150, 180]):
                u = np.array([np.cos(base + da), np.sin(base + da)])
                ha = "left" if u[0] > 0.35 else ("right" if u[0] < -0.35 else "center")
                va = "bottom" if u[1] > 0.35 else ("top" if u[1] < -0.35 else "center")
                setting = (pos[k] + (radii[k] + ring) * u, ha, va)
                t.set_position(setting[0])
                t.set_ha(ha)
                t.set_va(va)
                bb = t.get_window_extent(renderer)
                (bx0, by0), (bx1, by1) = inv.transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
                box = (bx0, by0, bx1, by1)
                if first is None:
                    first = (setting, box)
                inside = bx0 > limit[0] and bx1 < limit[2] and by0 > limit[1] and by1 < limit[3]
                if inside and not overlaps(box, others) and not overlaps(box, [boxes[k]]):
                    pr = protrusion(box)
                    if best is None or pr < best[0] - 1e-9:
                        best = (pr, setting, box)
                    if pr <= 0:
                        break
            if best is not None and best[0] <= 0:
                break
        if best is None:  # nothing free: fall back to radially outward
            log.warning("no free label position for %s", t.get_text())
            best = (0.0, *first)
        _, (xy, ha, va), box = best
        t.set_position(xy)
        t.set_ha(ha)
        t.set_va(va)
        boxes.append(box)
        placed.append(box)
    return placed


def route_edges(pos: np.ndarray, radii: np.ndarray, edges: np.ndarray, n_points: int = 40, iters: int = 500,
                tension: float = 0.3, push: float = 0.02, reach: float = 0.02, seed: int = 0) -> np.ndarray:
    """Curve the edges around the nodes they are not attached to.

    Every edge is a chain of ``n_points`` points whose ends sit on its two nodes. Two forces
    move the interior points:

    * **tension**: each point is pulled toward the midpoint of its neighbors (``tension`` is
      the fraction of the offset removed per iteration), so the curve wants to be straight;
    * **node repulsion**: every node except the edge's own two pushes the points away from
      its center. The force is strong inside the circle (``1 + depth / reach``, in units of
      ``push``) and falls off exponentially outside the boundary, ``exp(-distance / reach)``,
      so it is negligible a few ``reach`` units from the circle.

    Returns the points, shape ``(len(edges), n_points, 2)``. With no obstacle an edge stays a
    straight line. A tiny sideways nudge (``seed``) decides which way an edge that runs exactly
    through a node's center goes around it.
    """
    rng = np.random.default_rng(seed)
    n_edges, n_nodes = len(edges), len(pos)
    a, b = pos[edges[:, 0]], pos[edges[:, 1]]
    t = np.linspace(0.0, 1.0, n_points)[None, :, None]
    P = a[:, None] + (b - a)[:, None] * t
    d = b - a
    normal = np.stack([-d[:, 1], d[:, 0]], axis=1)
    normal /= np.linalg.norm(normal, axis=1, keepdims=True) + 1e-12
    P = P + rng.normal(size=(n_edges, 1, 1)) * 1e-3 * np.sin(np.pi * t) * normal[:, None]
    own = np.zeros((n_edges, n_nodes), dtype=bool)
    own[np.arange(n_edges), edges[:, 0]] = own[np.arange(n_edges), edges[:, 1]] = True
    for _ in range(iters):
        inner = P[:, 1:-1]
        offset = 0.5 * (P[:, :-2] + P[:, 2:]) - inner  # toward the neighbors' midpoint
        delta = inner[:, :, None, :] - pos[None, None]
        dist = np.linalg.norm(delta, axis=-1) + 1e-9
        depth = dist - radii[None, None]  # < 0 inside the node
        f = np.where(depth < 0, 1.0 - depth / reach, np.exp(-np.maximum(depth, 0) / reach))
        f = np.where(own[:, None, :] | (depth > 8 * reach), 0.0, f)
        P[:, 1:-1] = inner + tension * offset + push * (f[..., None] * delta / dist[..., None]).sum(2)
    return P


def edge_clearance(P: np.ndarray, pos: np.ndarray, radii: np.ndarray, edges: np.ndarray) -> float:
    """Smallest distance from any routed edge to the *boundary* of a node it is not attached
    to (negative: the edge runs under that node)."""
    dist = np.linalg.norm(P[:, :, None, :] - pos[None, None], axis=-1) - radii[None, None]
    own = np.zeros(dist.shape[::2], dtype=bool)
    own[np.arange(len(edges)), edges[:, 0]] = own[np.arange(len(edges)), edges[:, 1]] = True
    dist = np.where(own[:, None, :], np.inf, dist)
    return float(dist.min())


def plot_graph(data: dict, out_base: Path, min_shared: int = 1000, layout: str = "spring", top_k: int = 3,
               n_restarts: int = 300, seed: int = 0, curved: bool = True) -> dict:
    keys, rows = data["keys"], np.asarray(data["rows"], float)
    shared = np.asarray(data["shared"])
    labels = list(data["labels"])
    # Prepared but not in the model yet: drawn with a dashed ring (see the legend).
    candidates = {k for k, key in enumerate(keys) if key in registry.candidate_keys()}
    radii = node_radii(rows)
    edges, counts = edge_list(shared, min_shared, top_k)
    s_lo, s_hi = float(max(min(min_shared, counts.min()), 1)), float(shared.max())
    strength = edge_strength(counts, s_lo, s_hi)

    if layout == "circle":
        pos = circle_layout(shared, edges, strength, seed)
        score = crossing_score(pos, edges, strength)
        radii = np.minimum(radii, 0.12)
    else:
        pos, score = best_spring_layout(shared, radii, edges, strength, n_restarts, seed)
    log.info("%d edges, %d crossings (weighted %.1f)", len(edges), score[0], score[1])

    fig, ax = plt.subplots(figsize=(12, 8.5))
    lw, al = edge_width, edge_alpha
    if curved:
        curves = route_edges(pos, radii, edges, seed=seed)
        log.info("closest approach of an edge to a node it is not attached to: %.3f", edge_clearance(curves, pos, radii, edges))
    else:
        curves = np.stack([pos[edges[:, 0]], pos[edges[:, 1]]], axis=1)
    for e in np.argsort(strength):  # weak edges first, so the strong ones are on top
        ax.plot(*curves[e].T, color=edge_color(strength[e]), lw=lw(strength[e]), alpha=al(strength[e]),
                solid_capstyle="round", zorder=1)
    inside = {k for k, key in enumerate(keys) if key in INSIDE_LABELS}
    texts = []
    for k, key in enumerate(keys):
        fill = FAMILY_COLORS.get(family_of(key), "k")
        ax.add_patch(Circle(pos[k], radii[k], fc=fill, ec="none", zorder=3))
        if k in inside:  # the label goes inside the circle, in a color that contrasts with it
            texts.append(ax.text(*pos[k], INSIDE_LABELS[key] or labels[k], fontsize=8, ha="center", va="center",
                                 color=text_color_on(fill), linespacing=0.95, zorder=5))
            continue
        out = pos[k] - pos.mean(0)
        out = out / (np.linalg.norm(out) + 1e-9)
        texts.append(ax.text(*(pos[k] + (radii[k] + 0.03) * out), labels[k], fontsize=8, ha="center", va="center",
                             zorder=5, path_effects=[pe.withStroke(linewidth=2.5, foreground="white")]))
    strokes = node_strokes(data)
    ax.set_aspect("equal")
    ax.axis("off")

    # The figure is scaled to fit the nodes, the legend and the labels. The labels are placed
    # inside the bounding box of the nodes where possible, so they should not make the nodes
    # smaller; if a few stick out, the limits grow to include them, and the labels are placed
    # again (their size in data units depends on the scale).
    node_lo, node_hi = pos.min(0), pos.max(0)
    lo, hi = node_lo.copy(), node_hi.copy()
    margin, w, ring = 0.04, 0.0, radii
    for _ in range(5):
        used_lo, used_hi = lo.copy(), hi.copy()
        ax.set_xlim(used_lo[0] - margin, used_hi[0] + margin + 0.15 + LEGEND_W)
        ax.set_ylim(min(used_lo[1] - margin, used_hi[1] + margin - LEGEND_H), used_hi[1] + margin)
        fig.tight_layout()
        w = STROKE_PT / points_per_unit(ax)
        ring = radii + w  # the strokes go outside the circles
        bbox_lo, bbox_hi = (pos - ring[:, None]).min(0), (pos + ring[:, None]).max(0)
        preferred = (bbox_lo[0], bbox_lo[1], bbox_hi[0], bbox_hi[1])
        limit = (bbox_lo[0] - 0.6, bbox_lo[1] - 0.6, bbox_hi[0] + 0.6, bbox_hi[1] + 0.6)
        boxes = place_labels(fig, ax, texts, pos, ring, skip=inside, preferred=preferred, limit=limit)
        lo = np.minimum(bbox_lo, np.min([b[:2] for b in boxes], axis=0)) if boxes else bbox_lo
        hi = np.maximum(bbox_hi, np.max([b[2:] for b in boxes], axis=0)) if boxes else bbox_hi
        if np.allclose(lo, used_lo, atol=1e-3) and np.allclose(hi, used_hi, atol=1e-3):
            break
    n_out = sum(b[0] < bbox_lo[0] or b[1] < bbox_lo[1] or b[2] > bbox_hi[0] or b[3] > bbox_hi[1] for b in boxes)
    log.info("%d of %d labels stick out of the bounding box of the nodes", n_out, len(boxes))
    ppu = points_per_unit(ax)
    renderer = fig.canvas.get_renderer()
    for k in inside:  # does the label fit in its circle?
        bb = texts[k].get_window_extent(renderer)
        if 0.5 * np.hypot(bb.width, bb.height) > radii[k] * ppu * fig.dpi / 72:
            log.warning("the label of %s does not fit in its circle", keys[k])
    for k in range(len(keys)):
        draw_stroke(ax, pos[k], radii[k], STROKE_PT, ppu, *strokes[k])
    # Ring just outside the parameter strokes, so it never hides them.
    ring = 1.9 * STROKE_PT / ppu
    for k in sorted(candidates):
        ax.add_patch(Circle(pos[k], radii[k] + ring, fill=False, ec="0.2", lw=0.9,
                            ls=(0, (2.5, 1.5)), zorder=4))

    # Legends, drawn in data units to the right of the graph
    x0, y = hi[0] + margin + 0.15, hi[1] + margin - 0.1
    head = dict(fontsize=9, weight="bold", va="center")
    ax.text(x0, y, "rows in catalog", **head)
    for ref in (1e3, 1e5, 1e6):
        r = float(node_radii(ref))
        y -= max(r + 0.04, 0.1)
        ax.add_patch(Circle((x0 + 0.2, y), r, fc="0.6", ec="none", zorder=3))
        ax.text(x0 + 0.2 + r + 0.06, y, f"{ref:,.0f}", fontsize=8, va="center")
        y -= r
    y -= 0.12
    ax.text(x0, y, "stars shared by two catalogs", **head)
    for ref in [10.0**k for k in range(int(np.ceil(np.log10(s_lo))), int(np.floor(np.log10(s_hi))) + 1)]:
        y -= 0.1
        s = float(edge_strength(np.array([ref]), s_lo, s_hi)[0])
        ax.plot([x0, x0 + 0.22], [y, y], color=edge_color(s), lw=lw(s), alpha=al(s), solid_capstyle="round")
        ax.text(x0 + 0.3, y, f"{ref:,.0f}", fontsize=8, va="center")
    y -= 0.14
    ax.text(x0, y, "spectra", **head)
    for fam in [f for f in FAMILY_COLORS if any(family_of(k) == f for k in keys)]:
        y -= 0.09
        ax.add_patch(Circle((x0 + 0.05, y), 0.03, fc=FAMILY_COLORS[fam], ec="none", zorder=3))
        ax.text(x0 + 0.13, y, fam, fontsize=8, va="center")
    y -= 0.14
    ax.text(x0, y, "parameters in catalog", **head)
    for param, text in ((None, "all parameters"), *((p, STROKE_LABELS[p]) for p in STROKE_PARAMS)):
        y -= 0.09
        ax.add_patch(Circle((x0 + 0.05, y), 0.03, fc="0.6", ec="none", zorder=3))
        draw_stroke(ax, (x0 + 0.05, y), 0.03, STROKE_PT, ppu, param, "all")
        ax.text(x0 + 0.13, y, text, fontsize=8, va="center")
    if candidates:
        y -= 0.09
        ax.add_patch(Circle((x0 + 0.05, y), 0.03, fill=False, ec="0.2", lw=0.9,
                            ls=(0, (2.5, 1.5)), zorder=3))
        ax.text(x0 + 0.13, y, "not in the model", fontsize=8, va="center")
    ax.text(x0, y - 0.12, f"edges: >= {min_shared:,} shared stars, or a catalog's top {top_k}", fontsize=7, color="0.3", va="center")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_base}.{ext}", dpi=170)
    plt.close(fig)
    return dict(n_edges=len(edges), crossings=score[0], weighted_crossings=score[1], pos=pos.tolist(),
                labels_outside=int(n_out))


def short(n: int) -> str:
    return f"{n / 1e6:.1f}M" if n >= 1e6 else (f"{n / 1e3:.0f}k" if n >= 1e4 else (f"{n / 1e3:.1f}k" if n >= 1e3 else str(n)))


def plot_matrix(data: dict, out_base: Path) -> None:
    """Shared stars for every pair; catalogs grouped by family, then by size."""
    from matplotlib.colors import LogNorm

    keys, rows = data["keys"], np.asarray(data["rows"])
    shared = np.asarray(data["shared"], float)
    order = sorted(range(len(keys)), key=lambda i: (list(FAMILY_COLORS).index(family_of(keys[i])), -rows[i]))
    S = shared[np.ix_(order, order)]
    n = len(order)
    fig, ax = plt.subplots(figsize=(10.5, 9), layout="constrained")
    M = np.ma.masked_where(S <= 0, S)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("0.93")
    im = ax.imshow(M, cmap=cmap, norm=LogNorm(vmin=10, vmax=S.max()))
    for i, j in itertools.product(range(n), range(n)):
        if i != j and S[i, j] >= 1000:
            ax.text(j, i, short(int(S[i, j])), ha="center", va="center", fontsize=5.5,
                    color="w" if S[i, j] < 3e4 else "k")
    labs = [data["labels"][i] for i in order]
    ax.set_xticks(range(n), labs, rotation=90, fontsize=7)
    ax.set_yticks(range(n), [f"{l}  ({short(rows[i])})" for l, i in zip(labs, order)], fontsize=7)
    from matplotlib.patches import Rectangle

    for pos_, i in enumerate(order):  # color chips along the left and top edges (readable for any color)
        c = FAMILY_COLORS.get(family_of(keys[i]), "k")
        ax.add_patch(Rectangle((-1.05, pos_ - 0.4), 0.7, 0.8, fc=c, ec="none", clip_on=False))
        ax.add_patch(Rectangle((pos_ - 0.4, -1.05), 0.8, 0.7, fc=c, ec="none", clip_on=False))
    ax.set_xlim(-1.2, n - 0.5)
    ax.set_ylim(n - 0.5, -1.2)
    handles = [Rectangle((0, 0), 1, 1, fc=c) for f, c in FAMILY_COLORS.items() if any(family_of(k) == f for k in keys)]
    ax.legend(handles, [f for f in FAMILY_COLORS if any(family_of(k) == f for k in keys)], title="spectra",
              fontsize=7, title_fontsize=8, loc="upper left", bbox_to_anchor=(1.12, 1.0), frameon=False)
    fig.colorbar(im, ax=ax, shrink=0.6, label="stars shared by both catalogs")
    ax.set_title("Shared stars (Gaia DR3 id); catalog sizes (rows) in parentheses", fontsize=10)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_base}.{ext}", dpi=150)
    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="astrotune-catalog-graph", description=__doc__.split("\n\n")[0])
    parser.add_argument("--out-dir", type=Path, default=config.PROJECT_ROOT / "diagnostic_plots" / "catalog_graph")
    parser.add_argument("--min-shared", type=int, default=1000, help="draw edges with at least this many shared stars")
    parser.add_argument("--top-k", type=int, default=3,
                        help="also draw each catalog's strongest K connections, however weak")
    parser.add_argument("--layout", choices=("spring", "circle"), default="spring")
    parser.add_argument("--restarts", type=int, default=300)
    parser.add_argument("--straight-edges", action="store_true",
                        help="draw straight edges instead of curving them around unrelated nodes")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--recount", action="store_true", help="recompute the counts instead of reusing the JSON")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / "catalog_overlaps.json"
    if path.exists() and not args.recount:
        data = json.load(open(path))
        if data["keys"] != list(registry.keys()) or "n_measured" not in data:
            data = None
    else:
        data = None
    if data is None:
        data = compute_overlaps()
        json.dump(data, open(path, "w"))
    info = plot_graph(data, args.out_dir / f"catalog_graph_{args.layout}", args.min_shared, args.layout,
                      args.top_k, args.restarts, args.seed, curved=not args.straight_edges)
    plot_matrix(data, args.out_dir / "catalog_matrix")
    print(f"{info['n_edges']} edges drawn, {info['crossings']} crossings (weighted {info['weighted_crossings']:.1f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
