import platform

import pytest

from funtracks.user_actions import (
    UserAddEdge,
    UserDeleteEdge,
    UserDeleteNode,
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
