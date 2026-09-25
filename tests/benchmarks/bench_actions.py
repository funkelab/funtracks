import platform

import numpy as np
import pytest

from funtracks.user_actions import (
    UserAddEdge,
    UserAddNode,
    UserDeleteEdge,
    UserDeleteNode,
    UserDeleteNodes,
    UserSetDivision,
    UserSwapPredecessors,
    UserUpdateNodeAttrs,
    UserUpdateNodesAttrs,
    UserUpdateSegmentation,
)
from funtracks.utils.tracksdata_utils import td_mask_to_pixels

from ._graph_builders import make_tracks

NUM_FRAMES = 20
FRAME_SHAPE = (500, 500)
CELLS_PER_FRAME = 50

# Operations per timed call. Even, so the self-inverting benchmarks return the graph to
# its starting state. Raise this (rather than ROUNDS) if a benchmark reads as noisy.
N_OPS = 50
ROUNDS = 3
if platform.system() == "Darwin":
    N_OPS = N_OPS * 2
    ROUNDS = ROUNDS * 2
    # test_update_node_attrs_single/_bulk and test_delete_nodes each draw a disjoint
    # ROUNDS*N_OPS-node batch from solution_edges (3 * ROUNDS * N_OPS total); more
    # frames keeps that supply ahead of the doubled ROUNDS/N_OPS above.
    NUM_FRAMES = NUM_FRAMES * 4

# Fraction of a node's mask painted in the segmentation benchmark. Must stay < 1: a
# patch that fully covers the mask makes UserUpdateSegmentation delete the node instead
# of updating its mask, which is a different (and much more expensive) code path.
PATCH_FRACTION = 3


@pytest.fixture(scope="module")
def tracks(_warm_jit):
    """One Tracks object shared by every benchmark in this module."""
    return make_tracks(
        n_frames=NUM_FRAMES,
        cells_per_frame=CELLS_PER_FRAME,
        frame_shape=FRAME_SHAPE,
    )


@pytest.fixture(scope="module")
def solution_edges(tracks):
    """Every existing (source, target) edge, in node id order.

    Node ids are assigned frame by frame, so this is ordered by time: consecutive entries
    are cells in the same frame, and the benchmarks' per-round slices therefore walk
    forward through the movie rather than repeatedly hitting one frame.
    """
    edges = []
    for node in sorted(int(n) for n in tracks.graph_solution.node_ids()):
        successors = tracks.successors(node)
        if successors:
            edges.append((node, int(successors[0])))
    return edges


@pytest.fixture(scope="module")
def frame_nodes(tracks):
    """Node ids grouped by frame, in id order within each frame.

    solution_edges/_node_batches walk forward through the movie in track-chain
    order, which suits the single-node benchmarks above but not test_set_division
    or test_swap_predecessors: both need several nodes at the same (or an adjacent)
    time point from distinct tracks, which this fixture gives directly. make_tracks
    links 1:1 between frames, so frame t's node at index i and frame t+1's node at
    index i are already connected -- callers pick nodes by differing index to land
    on distinct tracks instead of an existing edge.
    """
    nodes = sorted(int(n) for n in tracks.graph_solution.node_ids())
    return [
        nodes[t * CELLS_PER_FRAME : (t + 1) * CELLS_PER_FRAME] for t in range(NUM_FRAMES)
    ]


def _node_batches(solution_edges, n_rounds=ROUNDS, per_round=N_OPS):
    """Split source nodes into one distinct batch per round.

    Each round gets its own nodes, and callers offset their slices so no two benchmarks
    share nodes. That keeps test_delete_nodes from removing nodes the attribute
    benchmarks reported on, and keeps every round of a destructive benchmark doing the
    same amount of work.

    Note this is no longer needed to avoid a measurement artifact: "score" is populated
    on every node at build time (see _graph_builders.make_tracks), so a node
    that has never been written costs the same to update as one that has.
    """
    nodes = [source for source, _ in solution_edges]
    batches = [nodes[r * per_round : (r + 1) * per_round] for r in range(n_rounds)]
    assert all(len(batch) == per_round for batch in batches), (
        f"need {n_rounds * per_round} source nodes to give each round a distinct batch, "
        f"only have {len(nodes)}"
    )
    return batches


def test_update_node_attrs_single(benchmark, tracks, solution_edges):
    """N single-node attribute updates, each its own history entry."""
    batches = iter(_node_batches(solution_edges))

    def run():
        for i, node in enumerate(next(batches)):
            UserUpdateNodeAttrs(tracks, node, {"score": float(i)})

    benchmark.pedantic(run, rounds=ROUNDS, iterations=1)


def test_update_node_attrs_bulk(benchmark, tracks, solution_edges):
    """The same N updates as one batched action.

    UserUpdateNodesAttrs exists to collapse history and refresh into a single entry for
    the whole batch; this pairs with test_update_node_attrs_single to show what that
    batching is worth.
    """
    # Offset past the batches test_update_node_attrs_single consumed, so these updates
    # also land on nodes not yet touched in this session.
    batches = iter(_node_batches(solution_edges[ROUNDS * N_OPS :]))

    def run():
        nodes = next(batches)
        UserUpdateNodesAttrs(
            tracks, nodes, {"score": [float(i) for i in range(len(nodes))]}
        )

    benchmark.pedantic(run, rounds=ROUNDS, iterations=1)


def test_add_delete_edges(benchmark, tracks, solution_edges):
    """Alternating edge delete and re-add on one edge.

    Both directions trigger UpdateTrackIDs: deleting orphans the downstream segment onto
    a fresh track/lineage id, and re-adding merges it back. Self-inverting, so the graph
    is unchanged afterwards.
    """
    edge = solution_edges[len(solution_edges) // 2]

    def run():
        for _ in range(N_OPS // 2):
            UserDeleteEdge(tracks, edge)
            UserAddEdge(tracks, edge)

    benchmark.pedantic(run, rounds=ROUNDS, iterations=1)


def test_swap_predecessors(benchmark, tracks, frame_nodes):
    """Alternating predecessor swap between two nodes at the same time point.

    Swapping twice is its own inverse, so the graph is unchanged afterwards. Uses a
    frame past what solution_edges-based benchmarks touch, and a dedicated pair
    per round so a mid-run failure in one round can't leave a later round's nodes
    in an unexpected state.
    """
    # frame_nodes[t][i] and frame_nodes[t-1][i] are already connected (1:1 linking in
    # make_tracks), so pairing index 0 at time t with index 1 at time t-1 gives each
    # node a predecessor from a different track -- the shape the swap is for.
    base_frame = NUM_FRAMES - ROUNDS - 1
    pairs = [
        (frame_nodes[base_frame + r][0], frame_nodes[base_frame + r][1])
        for r in range(ROUNDS)
    ]
    pairs = iter(pairs)

    def run():
        nodes = next(pairs)
        for _ in range(N_OPS // 2):
            UserSwapPredecessors(tracks, nodes)
            UserSwapPredecessors(tracks, nodes)

    benchmark.pedantic(run, rounds=ROUNDS, iterations=1)


def test_set_division(benchmark, tracks, frame_nodes):
    """Alternating make/break of a division between a parent and two children.

    Making then breaking is self-inverting, so the graph is unchanged afterwards.
    The two children are two distinct-track nodes at the frame after the parent;
    making the division deletes their existing (different-track) predecessor edges
    and adds the parent -> child edges instead, which is the expensive case
    (UserSetDivision's own conflicting-edge cleanup, not just a bare 2-edge add).
    """
    base_frame = NUM_FRAMES - ROUNDS - 2
    trios = [
        (
            frame_nodes[base_frame + r][0],
            frame_nodes[base_frame + r + 1][0],
            frame_nodes[base_frame + r + 1][1],
        )
        for r in range(ROUNDS)
    ]
    trios = iter(trios)

    def run():
        nodes = next(trios)
        for _ in range(N_OPS // 2):
            UserSetDivision(tracks, nodes)
            UserSetDivision(tracks, nodes)

    benchmark.pedantic(run, rounds=ROUNDS, iterations=1)


def test_add_delete_node(benchmark, tracks):
    """Alternating add and delete of a brand-new, unconnected node.

    Self-inverting, so the graph is unchanged afterwards. The new node starts its
    own track (no predecessor/successor edges), which is UserAddNode's cheapest
    graph-structural path; this isolates UserAddNode/UserDeleteNode's own
    bookkeeping (track/lineage id assignment, history) from
    test_update_segmentation's paint-existing-node case above.

    Every node in this graph carries mask/bbox (see _graph_builders.make_tracks),
    which registers a spatial index that requires bbox on every node added to it --
    a bare position dict raises inside tracksdata's spatial filter -- so pixels are
    passed here rather than a position, same as a real "paint a new cell" call.

    UserDeleteNode soft-deletes: the node id keeps living in graph_full afterwards,
    so a deleted id can never be reused by a later UserAddNode (it raises "already
    exists"). Node ids therefore come from a counter that only increases across
    rounds, rather than being reused per round like the other benchmarks' node
    batches.
    """
    time_key = tracks.features.time_key
    track_key = tracks.features.tracklet_key
    next_node_id = max(int(n) for n in tracks.graph_full.node_ids()) + 1

    # A small 3x3 pixel patch at time 0, away from any real cell.
    ys, xs = np.meshgrid(np.arange(3), np.arange(3), indexing="ij")
    ts = np.zeros_like(ys)
    pixels = (ts.ravel(), ys.ravel(), xs.ravel())

    counter = [0]

    def run():
        for _ in range(N_OPS // 2):
            node = next_node_id + counter[0]
            counter[0] += 1
            attrs = {
                time_key: 0,
                track_key: tracks.get_next_track_id(),
            }
            UserAddNode(tracks, node, attrs, pixels=pixels)
            UserDeleteNode(tracks, node)

    benchmark.pedantic(run, rounds=ROUNDS, iterations=1)


def test_update_segmentation(benchmark, tracks, solution_edges):
    """Alternating paint-out and paint-back of one sub-mask patch.

    Simulates the paint stroke path: UpdateNodeSeg plus the regionprops recompute
    (position, area) and the IoU refresh on adjacent edges. Removing then restoring the
    same patch keeps every operation on the UpdateNodeSeg branch and leaves the mask
    exactly as it started -- the patch is a strict subset of the mask so the node is
    never fully erased, and the paint-back targets an existing node id so it never falls
    through to UserAddNode.
    """
    node = solution_edges[len(solution_edges) // 2][0]
    mask_pixels = td_mask_to_pixels(
        tracks.get_mask(node), tracks.get_time(node), ndim=tracks.ndim
    )
    n_patch = len(mask_pixels[0]) // PATCH_FRACTION
    patch = tuple(dim_pixels[:n_patch] for dim_pixels in mask_pixels)

    def run():
        for i in range(N_OPS):
            if i % 2 == 0:
                # Erase the patch: old value is the node, new value is background.
                UserUpdateSegmentation(
                    tracks,
                    new_value=0,
                    updated_pixels=[(patch, node)],
                    current_track_id=1,
                )
            else:
                # Paint it back: old value is background, new value is the node.
                UserUpdateSegmentation(
                    tracks,
                    new_value=node,
                    updated_pixels=[(patch, 0)],
                    current_track_id=1,
                )

    benchmark.pedantic(run, rounds=ROUNDS, iterations=1)


def test_delete_nodes(benchmark, tracks, solution_edges):
    """N node deletions, each cascading to its adjacent edges and track ids.

    Destructive and not self-inverting, so this runs last in the module and takes a
    fresh slice of nodes per round. Deleting a mid-track node makes UserDeleteNode
    delete both adjacent edges and reconnect the neighbours with a skip edge, which is
    the expensive case.
    """
    # Start past the nodes the attribute benchmarks consumed so deletions do not remove
    # nodes those benchmarks already reported on.
    batches = iter(_node_batches(solution_edges[2 * ROUNDS * N_OPS :]))

    def run():
        for node in next(batches):
            UserDeleteNode(tracks, node)

    benchmark.pedantic(run, rounds=ROUNDS, iterations=1)


def test_delete_nodes_bulk(benchmark, tracks, solution_edges):
    """The same N deletions as test_delete_nodes, as one batched action.

    UserDeleteNodes exists to collapse history and refresh into a single entry for
    the whole batch; this pairs with test_delete_nodes to show what that batching is
    worth, the same comparison test_update_node_attrs_bulk makes for attribute
    updates. Destructive and not self-inverting, so this also runs late and takes a
    fresh slice of nodes past test_delete_nodes'.
    """
    batches = iter(_node_batches(solution_edges[3 * ROUNDS * N_OPS :]))

    def run():
        UserDeleteNodes(tracks, next(batches))

    benchmark.pedantic(run, rounds=ROUNDS, iterations=1)
