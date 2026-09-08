"""Benchmarks for Tracks' node/edge attribute accessors -- both the single-item and
batch getters, and the setters. Separate from bench_actions.py, which benchmarks the
user-facing action layer (UserUpdateNodeAttrs, UserDeleteEdge, etc.) built on top of
these.
"""

import platform

import numpy as np
import pytest

from ._graph_builders import make_tracks

NUM_FRAMES = 20
FRAME_SHAPE = (500, 500)
CELLS_PER_FRAME = 50

ROUNDS = 3
N_OPS = 50
if platform.system() == "Darwin":
    # macOS runners are noisier than Linux; double for more stable measurements (same
    # adjustment as bench_actions.py). Unlike there, NUM_FRAMES does not need to scale
    # with this: every fixture here is module-scoped and reused across rounds rather
    # than drawing a fresh disjoint node batch per round, so there is no node-supply
    # constraint tied to ROUNDS * N_OPS.
    N_OPS = N_OPS * 2
    ROUNDS = ROUNDS * 2


@pytest.fixture(scope="module")
def tracks(_warm_jit):
    """One Tracks object shared by every benchmark in this module."""
    return make_tracks(
        n_frames=NUM_FRAMES,
        cells_per_frame=CELLS_PER_FRAME,
        frame_shape=FRAME_SHAPE,
    )


@pytest.fixture(scope="module")
def all_nodes(tracks):
    """Every node id in the solution graph."""
    return sorted(int(n) for n in tracks.graph_solution.node_ids())


@pytest.fixture(scope="module")
def small_subset(all_nodes):
    """~1% of the graph's nodes, to exercise get_nodes_attr's filter-first branch.

    get_nodes_attr picks between filtering first and fetching the whole graph based
    on what fraction of the graph `nodes` covers (see Tracks.get_nodes_attr) -- this
    fixture and `all_nodes` are sized to land on opposite sides of that threshold, so
    the two benchmarks below cover both branches instead of only the one a single
    fixture would happen to hit.
    """
    n = max(1, len(all_nodes) // 100)
    return all_nodes[:n]


@pytest.fixture(scope="module")
def all_edges(tracks):
    """Every (source, target) edge in the solution graph, in node id order."""
    edges = []
    for node in sorted(int(n) for n in tracks.graph_solution.node_ids()):
        for succ in tracks.successors(node):
            edges.append((node, int(succ)))
    return edges


@pytest.fixture(scope="module")
def one_track_id(tracks, all_nodes):
    """A track id with more than one node, to exercise the real multi-node
    per-tracklet query rather than a degenerate single-node one."""
    tid = tracks.get_track_id(all_nodes[0])
    for node in all_nodes:
        cand = tracks.get_track_id(node)
        if len(tracks.track_id_to_node[cand]) > 1:
            return cand
    return tid


@pytest.fixture(scope="module")
def one_node(all_nodes):
    """A node in the middle of the movie, for single-item read/write benchmarks."""
    return all_nodes[len(all_nodes) // 2]


@pytest.fixture(scope="module")
def one_edge(all_edges):
    """An edge in the middle of the movie, for single-item read/write benchmarks."""
    return all_edges[len(all_edges) // 2]


@pytest.fixture(scope="module")
def small_batch(all_nodes):
    """A sub-batch of N_OPS distinct nodes (evenly dividing it), for the
    batched-setter benchmarks. Sized so N_OPS // len(small_batch) batched calls
    cover the same total node-writes as N_OPS single-node calls -- an
    apples-to-apples batched-vs-looped comparison, not just fewer calls doing
    more total work.
    """
    return all_nodes[: N_OPS // 5]


@pytest.fixture(scope="module")
def small_batch_edges(all_edges):
    """Same sizing rationale as small_batch, for the edge setter comparison."""
    return all_edges[: N_OPS // 5]


def test_get_nodes_attr_all_nodes(benchmark, tracks, all_nodes):
    """get_nodes_attr called with (~all of) the graph's nodes.

    This is the shape TrackColormap.set_tracks (motile-tracker) calls it with --
    every node, every time the colormap is rebuilt -- which is why
    get_nodes_attr fetches the whole graph unfiltered rather than filtering to
    `nodes` first: filtering has its own setup cost that only pays off once `nodes`
    is a small slice of the graph (see test_get_nodes_attr_small_subset).
    """

    def run():
        tracks.get_nodes_attr(all_nodes, "area")

    benchmark.pedantic(run, rounds=ROUNDS, iterations=1)


def test_get_nodes_attr_small_subset(benchmark, tracks, small_subset):
    """get_nodes_attr called with a small slice of the graph's nodes.

    Exercises the filter-first branch: filtering to `nodes` before fetching
    attributes only walks the requested nodes, which wins once `nodes` is a small
    enough fraction of the graph (see test_get_nodes_attr_all_nodes for the
    opposite end).
    """

    def run():
        tracks.get_nodes_attr(small_subset, "area")

    benchmark.pedantic(run, rounds=ROUNDS, iterations=1)


def test_get_times(benchmark, tracks, all_nodes):
    """Batch time lookup for every node -- the tree/table view's hot path."""

    def run():
        tracks.get_times(all_nodes)

    benchmark.pedantic(run, rounds=ROUNDS, iterations=1)


def test_get_track_ids(benchmark, tracks, all_nodes):
    """Batch track-id lookup for every node.

    A thin wrapper over get_nodes_attr, documented as optimised for bulk (all-node)
    calls, so this benchmark should stay on `all_nodes` rather than a subset -- the
    filter-first branch is already covered by test_get_nodes_attr_small_subset.
    """

    def run():
        tracks.get_track_ids(all_nodes)

    benchmark.pedantic(run, rounds=ROUNDS, iterations=1)


def test_get_positions(benchmark, tracks, all_nodes):
    """Batch position lookup for every node -- always fetches the whole graph
    unfiltered (documented, unlike get_nodes_attr's adaptive filter/fetch-all)."""

    def run():
        tracks.get_positions(all_nodes)

    benchmark.pedantic(run, rounds=ROUNDS, iterations=1)


def test_get_positions_incl_time(benchmark, tracks, all_nodes):
    """get_positions with incl_time=True: an extra attr key and an extra
    positional reorder (see the `if incl_time` branch in Tracks.get_positions)."""

    def run():
        tracks.get_positions(all_nodes, incl_time=True)

    benchmark.pedantic(run, rounds=ROUNDS, iterations=1)


def test_get_edges_attr(benchmark, tracks, all_edges):
    """Batch edge-attribute lookup, over every edge in the graph.

    get_edges_attr still loops get_edge_attr per edge (see Tracks.get_edges_attr) --
    tracksdata's filter() only accepts node_ids, not edge_ids, so there is no
    equivalent one-query batching for edges today. This benchmark is a baseline to
    compare against if/when that becomes available upstream.
    """

    def run():
        tracks.get_edges_attr(all_edges, "iou")

    benchmark.pedantic(run, rounds=ROUNDS, iterations=1)


def test_get_track_node_times(benchmark, tracks, one_track_id):
    """Every (time, node) pair for one tracklet, sorted by time.

    Backs get_track_neighbors and has_track_id_at_time, which are not benchmarked
    separately since both are a thin loop over this same query.
    """

    def run():
        tracks.get_track_node_times(one_track_id)

    benchmark.pedantic(run, rounds=ROUNDS, iterations=1)


# ========== single-item getters ==========
# One graph query each, resolved directly on graph_full/graph_solution with no
# batching to compare against -- baselines for the corresponding bulk methods above.


def test_get_position(benchmark, tracks, one_node):
    def run():
        tracks.get_position(one_node)

    benchmark.pedantic(run, rounds=ROUNDS, iterations=1)


def test_get_time(benchmark, tracks, one_node):
    def run():
        tracks.get_time(one_node)

    benchmark.pedantic(run, rounds=ROUNDS, iterations=1)


def test_get_track_id(benchmark, tracks, one_node):
    def run():
        tracks.get_track_id(one_node)

    benchmark.pedantic(run, rounds=ROUNDS, iterations=1)


def test_get_lineage_id(benchmark, tracks, one_node):
    def run():
        tracks.get_lineage_id(one_node)

    benchmark.pedantic(run, rounds=ROUNDS, iterations=1)


def test_get_mask(benchmark, tracks, one_node):
    def run():
        tracks.get_mask(one_node)

    benchmark.pedantic(run, rounds=ROUNDS, iterations=1)


def test_get_node_attr(benchmark, tracks, one_node):
    def run():
        tracks.get_node_attr(one_node, "area")

    benchmark.pedantic(run, rounds=ROUNDS, iterations=1)


def test_get_edge_attr(benchmark, tracks, one_edge):
    def run():
        tracks.get_edge_attr(one_edge, "iou")

    benchmark.pedantic(run, rounds=ROUNDS, iterations=1)


def test_predecessors(benchmark, tracks, one_node):
    def run():
        tracks.predecessors(one_node)

    benchmark.pedantic(run, rounds=ROUNDS, iterations=1)


def test_successors(benchmark, tracks, one_node):
    def run():
        tracks.successors(one_node)

    benchmark.pedantic(run, rounds=ROUNDS, iterations=1)


# ========== setters ==========
# All alternate between two values so the graph is unchanged after each pedantic
# round (same self-inverting convention as bench_actions.py), since these mutate
# the shared module-scope `tracks` fixture in place.


def test_set_node_attr_single(benchmark, tracks, one_node):
    """N single-node attribute writes, each its own graph_full write."""

    def run():
        for i in range(N_OPS):
            tracks._set_node_attr(one_node, "score", float(i % 2))

    benchmark.pedantic(run, rounds=ROUNDS, iterations=1)


def test_set_nodes_attr_bulk(benchmark, tracks, small_batch):
    """N node-writes as len(small_batch)-sized batched update_node_attrs calls.

    Same total node-writes as test_set_node_attr_single (N_OPS), split into
    N_OPS // len(small_batch) batched calls instead of N_OPS single-node calls --
    an apples-to-apples comparison of what _set_nodes_attr's single-query batching
    is worth over looping the single-node setter, same comparison bench_actions.py
    draws at the action layer, one level lower.
    """
    n_calls = max(1, N_OPS // len(small_batch))

    def run():
        for i in range(n_calls):
            tracks._set_nodes_attr(
                small_batch, "score", [float(i % 2)] * len(small_batch)
            )

    benchmark.pedantic(run, rounds=ROUNDS, iterations=1)


def test_set_position(benchmark, tracks, one_node):
    """N single-node position writes, alternating between two positions."""
    pos_a = tracks.get_position(one_node)
    pos_b = [p + 1.0 for p in pos_a]

    def run():
        for i in range(N_OPS):
            tracks.set_position(one_node, pos_a if i % 2 == 0 else pos_b)

    benchmark.pedantic(run, rounds=ROUNDS, iterations=1)


def test_set_positions_bulk(benchmark, tracks, small_batch):
    """Same total node-writes as test_set_position (N_OPS), as batched
    set_positions calls instead of N_OPS single-node calls."""
    positions_a = np.array([tracks.get_position(node) for node in small_batch])
    positions_b = positions_a + 1.0
    n_calls = max(1, N_OPS // len(small_batch))

    def run():
        for i in range(n_calls):
            tracks.set_positions(small_batch, positions_a if i % 2 == 0 else positions_b)

    benchmark.pedantic(run, rounds=ROUNDS, iterations=1)


def test_set_edge_attr_single(benchmark, tracks, one_edge):
    """N single-edge attribute writes, each its own edge_id lookup + graph_full write."""

    def run():
        for i in range(N_OPS):
            tracks._set_edge_attr(one_edge, "iou", float(i % 2))

    benchmark.pedantic(run, rounds=ROUNDS, iterations=1)


def test_set_edges_attr_bulk(benchmark, tracks, small_batch_edges):
    """Same total edge-writes as test_set_edge_attr_single (N_OPS), as
    len(small_batch_edges)-sized _set_edges_attr calls instead of N_OPS
    single-edge calls.

    _set_edges_attr still loops update_edge_attrs per edge (unlike
    _set_nodes_attr's single batched call) -- see the comment on
    test_get_edges_attr. This benchmark is a baseline for whether that loop
    is worth collapsing, and by how much, if/when a bulk edge-id write path
    is available.
    """
    n_calls = max(1, N_OPS // len(small_batch_edges))

    def run():
        for i in range(n_calls):
            values = [float(i % 2)] * len(small_batch_edges)
            tracks._set_edges_attr(small_batch_edges, "iou", values)

    benchmark.pedantic(run, rounds=ROUNDS, iterations=1)


def test_update_mask(benchmark, tracks, one_node):
    """Alternating mask update on one node, to isolate update_mask itself from the
    full UserUpdateSegmentation action path
    """
    mask = tracks.get_mask(one_node)

    def run():
        for _ in range(N_OPS):
            tracks.update_mask(one_node, mask)

    benchmark.pedantic(run, rounds=ROUNDS, iterations=1)
