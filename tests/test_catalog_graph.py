"""Tests for the catalog connectivity graph (layout geometry and counts)."""

import numpy as np

from astrotune import catalog_graph as cg


def test_crossing_count_of_a_square_and_its_diagonals():
    pos = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    edges = np.array([[0, 2], [1, 3], [0, 1], [1, 2]])  # two diagonals cross; sides share endpoints
    n, w = cg.crossing_score(pos, edges, np.ones(4))
    assert n == 1 and w == 1.0
    pos2 = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], float)  # now the sides 0-1 and 2-3 are parallel
    assert cg.crossing_score(pos2, np.array([[0, 1], [2, 3]]), np.ones(2))[0] == 0


def test_edges_below_the_threshold_are_dropped():
    S = np.array([[0, 5000, 10], [5000, 0, 999], [10, 999, 0]])
    edges, counts = cg.edge_list(S, 1000)
    assert edges.tolist() == [[0, 1]] and counts.tolist() == [5000]


def test_stress_layout_reproduces_distances_and_overlaps_are_removed():
    truth = np.array([[0, 0], [1, 0], [0, 2], [3, 1]], float)
    delta = np.linalg.norm(truth[:, None] - truth[None], axis=-1)
    X = cg.smacof(delta, np.random.default_rng(0).normal(size=(4, 2)), iters=500)
    D = np.linalg.norm(X[:, None] - X[None], axis=-1)
    assert np.allclose(D, delta, atol=1e-3)
    pos = cg.remove_overlaps(np.array([[0, 0], [0.05, 0], [0.5, 0.5]], float), np.array([0.1, 0.1, 0.1]))
    d = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
    assert d[0, 1] >= 0.2 - 1e-3


def test_node_radii_grow_with_catalog_size():
    r = cg.node_radii(np.array([300.0, 1e4, 4e6]))
    assert r[0] < r[1] < r[2]
    assert 8 < r[2] / r[0] < 14  # contrast between the smallest and the largest catalog
    w = cg.edge_width(np.array([0.0, 0.5, 1.0]))
    assert w[0] < w[1] < w[2] and w[2] / w[0] > 10 and cg.edge_alpha(0.0) < cg.edge_alpha(1.0) <= 1.0


def test_every_registered_catalog_has_a_family():
    from astrotune import registry

    assert all(cg.family_of(k) != "other" for k in registry.keys())


def test_circle_layout_puts_strongly_linked_nodes_next_to_each_other():
    n = 8
    S = np.zeros((n, n), dtype="i8")
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 0)]:  # a 4-cycle of strong edges, the rest unconnected
        S[a, b] = S[b, a] = 100000
    edges, counts = cg.edge_list(S, 1000)
    strength = cg.edge_strength(counts, 1000, 100000)
    pos = cg.circle_layout(S, edges, strength, seed=0, n_iter=3000)
    assert cg.crossing_score(pos, edges, strength)[0] == 0


def test_every_catalog_keeps_its_strongest_edges():
    S = np.array([
        [0, 90000, 50, 40, 0],
        [90000, 0, 20, 10, 0],
        [50, 20, 0, 30, 5],
        [40, 10, 30, 0, 0],
        [0, 0, 5, 0, 0],
    ])
    edges, _ = cg.edge_list(S, 1000, top_k=0)
    assert edges.tolist() == [[0, 1]]
    edges, counts = cg.edge_list(S, 1000, top_k=1)
    pairs = {tuple(e) for e in edges.tolist()}
    assert (0, 1) in pairs and (2, 4) in pairs  # the isolated catalog 4 keeps its only link (5 stars)
    assert all(any(i in e for e in pairs) for i in range(4))  # nobody with a link is left without an edge
    assert (0, 4) not in pairs  # pairs without shared stars are never drawn
    assert counts.min() >= 5


def test_family_colors_are_distinguishable_under_deuteranomaly():
    import itertools

    from distinctipy import colorblind, distinctipy

    def rgb(h):
        return tuple(int(h[i:i + 2], 16) / 255 for i in (1, 3, 5))

    cols = [rgb(c) for c in cg.FAMILY_COLORS.values()]
    sim = [colorblind.colorblind_filter(c, cg.COLORBLIND_TYPE) for c in cols]
    assert min(distinctipy.color_distance(a, b) for a, b in itertools.combinations(sim, 2)) > 0.4
    # the previous hand-picked palette would not have passed
    old = [rgb(h) for h in ("#d62728", "#ff7f0e", "#1f77b4", "#2ca02c", "#9467bd", "#8c564b", "#7f7f7f")]
    old_sim = [colorblind.colorblind_filter(c, cg.COLORBLIND_TYPE) for c in old]
    assert min(distinctipy.color_distance(a, b) for a, b in itertools.combinations(old_sim, 2)) < 0.2


def test_edge_color_and_opacity_follow_the_shared_star_count():
    import numpy as np

    c0, c1 = np.array(cg.edge_color(0.0)), np.array(cg.edge_color(1.0))
    assert c0.shape == (3,) and np.all((c0 >= 0) & (c0 <= 1)) and not np.allclose(c0, c1)
    assert c1.sum() < 2.6  # the bright yellow end of plasma is not used (invisible on white)
    a = cg.edge_alpha(np.array([0.0, 0.2, 0.45, 0.7, 1.0]))
    assert np.all(np.diff(a) > 0) and a[0] >= 0.15 and a[-1] <= 1.0
    assert a[2] < 0.4  # a 1,000-star edge (strength ~0.45) is already fairly faint


def test_edges_curve_around_unrelated_nodes_and_stay_straight_otherwise():
    import numpy as np

    pos = np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 1.5]])
    radii = np.array([0.05, 0.05, 0.2, 0.05])
    edges = np.array([[0, 1], [0, 3]])
    P = cg.route_edges(pos, radii, edges)
    assert np.allclose(P[:, 0], pos[edges[:, 0]]) and np.allclose(P[:, -1], pos[edges[:, 1]])  # ends fixed
    assert cg.edge_clearance(P, pos, radii, edges) > 0  # nothing runs under the node at the origin
    # the edge through the origin bends around it, but not absurdly: it stays short
    straight = 2.0
    length = np.linalg.norm(np.diff(P[0], axis=0), axis=1).sum()
    assert straight < length < 1.25 * straight
    # an edge with no obstacle in the way stays straight
    free = cg.route_edges(pos, radii, np.array([[0, 3]]))
    line = pos[0] + (pos[3] - pos[0]) * np.linspace(0, 1, free.shape[1])[:, None]
    assert np.allclose(free[0], line, atol=2e-3)
