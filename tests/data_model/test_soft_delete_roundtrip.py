"""Step 6: soft-delete round-trip + repeated delete<->undo stability.

Soft-delete keeps the node/edge in ``graph_full`` (flag ``solution=False``) and only drops
it from ``graph_solution``. These tests pin the core Phase-1 guarantees:
- the full graph's topology is preserved across a leaf-node soft-delete (only flags flip);
- delete -> undo restores the solution view exactly (ids + attributes);
- repeated undo/redo through the real ``ActionHistory`` never drifts the view (invariant
  #4 in the persistence plan) — the R2 in-place revive must be a true inverse of remove.

A leaf node (5, edge 4->5) is used so the delete introduces no reconnection skip-edge; the
skip-edge case is exercised in ``test_mid_track_delete_*`` to document that graph_full
accumulates the skip edge as a candidate.
"""

import pytest

from funtracks.user_actions import UserDeleteNode


def _solution_state(tracks):
    """A hashable snapshot of the solution view's topology."""
    return (
        tuple(sorted(tracks.graph_solution.node_ids())),
        tuple(sorted(tracks.graph_solution.edge_list())),
    )


def _full_state(tracks):
    return (
        tuple(sorted(tracks.graph_full.node_ids())),
        tuple(sorted(tracks.graph_full.edge_list())),
    )


def _positions(tracks):
    """pos per solution node as plain float tuples (array-safe for == comparison)."""
    return {
        n: tuple(float(x) for x in tracks.get_node_attr(n, "pos"))
        for n in tracks.graph_solution.node_ids()
    }


@pytest.mark.parametrize("ndim", [3, 4])
@pytest.mark.parametrize("with_seg", [True, False])
def test_soft_delete_keeps_leaf_node_in_full_graph(get_tracks, ndim, with_seg):
    tracks = get_tracks(ndim=ndim, with_seg=with_seg, is_solution=True)
    full_before = _full_state(tracks)
    sol_nodes_before = set(tracks.graph_solution.node_ids())

    UserDeleteNode(tracks, 5)

    # Gone from the solution view ...
    assert 5 not in tracks.graph_solution.node_ids()
    assert set(tracks.graph_solution.node_ids()) == sol_nodes_before - {5}
    assert not tracks.graph_solution.has_edge(4, 5)

    # ... but preserved in the full graph as a soft-deleted candidate. Topology of the
    # full graph is unchanged: only the solution flag flipped.
    assert 5 in tracks.graph_full.node_ids()
    assert tracks.graph_full.has_edge(4, 5)
    assert tracks.graph_full.nodes[5]["solution"] is False
    assert _full_state(tracks) == full_before


@pytest.mark.parametrize("ndim", [3, 4])
@pytest.mark.parametrize("with_seg", [True, False])
def test_delete_undo_redo_roundtrip_identity(get_tracks, ndim, with_seg):
    tracks = get_tracks(ndim=ndim, with_seg=with_seg, is_solution=True)
    sol_ref = _solution_state(tracks)
    full_ref = _full_state(tracks)
    pos_ref = _positions(tracks)

    UserDeleteNode(tracks, 5)
    assert _solution_state(tracks) != sol_ref  # actually changed

    # Undo restores the solution view exactly, and the full graph is untouched.
    assert tracks.undo()
    assert _solution_state(tracks) == sol_ref
    assert _full_state(tracks) == full_ref
    assert _positions(tracks) == pos_ref

    # Redo re-deletes; undo restores again — same states.
    assert tracks.redo()
    assert 5 not in tracks.graph_solution.node_ids()
    assert tracks.undo()
    assert _solution_state(tracks) == sol_ref


@pytest.mark.parametrize("ndim", [3, 4])
def test_repeated_delete_undo_is_stable(get_tracks, ndim):
    """Invariant #4: N undo/redo cycles must not drift the solution view or full graph."""
    tracks = get_tracks(ndim=ndim, with_seg=True, is_solution=True)
    sol_ref = _solution_state(tracks)
    iou_ref = tracks.get_edge_attr((4, 5), "iou")

    UserDeleteNode(tracks, 5)
    deleted_sol = _solution_state(tracks)
    full_after_delete = _full_state(tracks)

    for _ in range(5):
        assert tracks.undo()
        assert _solution_state(tracks) == sol_ref
        # Revived edge keeps its computed attribute.
        assert tracks.get_edge_attr((4, 5), "iou") == iou_ref

        assert tracks.redo()
        assert _solution_state(tracks) == deleted_sol
        # Full graph topology is identical every redo — no candidate accumulation.
        assert _full_state(tracks) == full_after_delete

    # Leave it restored and confirm a clean final identity.
    assert tracks.undo()
    assert _solution_state(tracks) == sol_ref


@pytest.mark.parametrize("ndim", [3, 4])
def test_mid_track_delete_leaves_skip_edge_candidate_in_full(get_tracks, ndim):
    """Deleting a mid-track node adds a reconnection skip-edge (3->5). On undo it is
    soft-deleted, so it persists in graph_full as a solution=False candidate while the
    solution view returns to its original topology."""
    tracks = get_tracks(ndim=ndim, with_seg=True, is_solution=True)
    sol_ref = _solution_state(tracks)
    assert not tracks.graph_full.has_edge(3, 5)

    UserDeleteNode(tracks, 4)
    assert tracks.graph_solution.has_edge(3, 5)
    assert not tracks.graph_solution.has_edge(3, 4)

    assert tracks.undo()
    # Solution view is back to the original topology (skip edge removed from the view) ...
    assert _solution_state(tracks) == sol_ref
    assert not tracks.graph_solution.has_edge(3, 5)
    # ... but the skip edge now lives in graph_full as a candidate; node 4 is retained.
    assert tracks.graph_full.has_edge(3, 5)
    assert tracks.graph_full.edge_id(3, 5) is not None
    assert 4 in tracks.graph_full.node_ids()


@pytest.mark.parametrize("ndim", [3, 4])
def test_attr_reads_resolve_for_soft_deleted_node(get_tracks, ndim):
    """Regression: attribute reads must resolve for soft-deleted (solution=False) nodes,
    and the bulk `get_positions` must agree with the single-node `get_position`. The bulk
    path previously queried `graph_solution` and KeyError'd on a soft-deleted node while
    `get_position` (graph_full) succeeded — a latent inconsistency invisible to tests that
    only ever query in-solution nodes.
    """
    tracks = get_tracks(ndim=ndim, with_seg=True, is_solution=True)
    pos_single_before = tracks.get_position(5)
    pos_bulk_before = tracks.get_positions([5])[0].tolist()
    assert pos_bulk_before == pytest.approx(pos_single_before)

    UserDeleteNode(tracks, 5)
    assert 5 not in tracks.graph_solution.node_ids()  # soft-deleted

    # Both single and bulk position reads still resolve (graph_full) and agree.
    pos_single_after = tracks.get_position(5)
    pos_bulk_after = tracks.get_positions([5])[0].tolist()
    assert pos_single_after == pytest.approx(pos_single_before)
    assert pos_bulk_after == pytest.approx(pos_single_before)

    # Other intrinsic attrs resolve too.
    assert tracks.get_node_attr(5, "area") is not None
    assert tracks.get_time(5) is not None
