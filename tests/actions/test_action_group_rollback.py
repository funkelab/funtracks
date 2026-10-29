"""A user action that raises partway is rolled back by ActionGroup.

The user actions check their input before applying anything, but an unexpected
error after the first basic action must not leave the tracks half-changed either.
"""

import polars as pl
import pytest

from funtracks.actions import (
    ActionGroup,
    AddEdge,
    AddNode,
    DeleteEdge,
    DeleteNode,
    UpdateNodeAttrs,
    UpdateNodeSeg,
    UpdateTrackIDs,
)
from funtracks.exceptions import InvalidActionError
from funtracks.user_actions import (
    UserAddNode,
    UserConnectNodes,
    UserDeleteNode,
    UserMergeNodes,
    UserSetDivision,
    UserSwapPredecessors,
    UserUpdateNodesAttrs,
)

# Fixture graph: nodes 1 (t=0), 2 (t=1), 3 (t=1), 4 (t=2), 5 (t=4), 6 (t=4)
# Edges: 1 -> 2, 1 -> 3, 3 -> 4, 4 -> 5

BASIC_ACTIONS = [
    AddEdge,
    AddNode,
    DeleteEdge,
    DeleteNode,
    UpdateNodeAttrs,
    UpdateNodeSeg,
    UpdateTrackIDs,
]


def count_basic_actions(tracks, run) -> int:
    """Count the basic actions that run(tracks) applies."""
    applied: list[ActionGroup] = []
    tracks.action_applied.connect(applied.append)
    run(tracks)
    tracks.action_applied.disconnect(applied.append)
    return len(applied)


def fail_at_basic_action(monkeypatch, n: int) -> None:
    """Make the n-th basic action constructed from now on raise before applying."""
    calls = 0
    for cls in BASIC_ACTIONS:

        def init(self, *args, _original=cls.__init__, **kwargs):
            nonlocal calls
            calls += 1
            if calls == n:
                raise RuntimeError("injected failure")
            _original(self, *args, **kwargs)

        monkeypatch.setattr(cls, "__init__", init)


def add_confidence(tracks):
    tracks.graph_solution.add_node_attr_key(
        "confidence", default_value=0.0, dtype=pl.Float64
    )


def make_tracks(get_tracks, with_seg, setup):
    tracks = get_tracks(ndim=3, with_seg=with_seg, prefill_track_ids=True)
    # The fixture's prefilled iou values are rounded; recompute them so that a
    # rollback, which recomputes them too, compares equal
    tracks.annotators.compute()
    if setup is not None:
        setup(tracks)
    return tracks


# (with_seg, setup, run): run must apply at least two basic actions
USER_ACTIONS = [
    pytest.param(True, None, lambda t: UserConnectNodes(t, [2, 6]), id="connect"),
    pytest.param(
        True,
        None,
        lambda t: UserMergeNodes(t, [2, 3], track_ids=3),
        id="merge",
        marks=pytest.mark.xfail(
            strict=True,
            reason="Inverting UpdateNodeSeg(added=True) restores the pixels but not a "
            "tight bbox, so plain undo of a merge has the same mismatch",
        ),
    ),
    pytest.param(True, None, lambda t: UserSwapPredecessors(t, (5, 6)), id="swap"),
    pytest.param(True, None, lambda t: UserSetDivision(t, (3, 5, 6)), id="set_division"),
    pytest.param(True, None, lambda t: UserDeleteNode(t, 4), id="delete_node"),
    pytest.param(
        False,
        None,
        # 1 divides upstream, so force deletes (1, 2) and (1, 3) first
        lambda t: UserAddNode(
            t, 7, attributes={"t": 2, "track_id": 1, "pos": [3, 4]}, force=True
        ),
        id="add_node_forced",
    ),
    pytest.param(
        True,
        add_confidence,
        lambda t: UserUpdateNodesAttrs(t, [1, 2, 3], {"confidence": [0.1, 0.2, 0.3]}),
        id="update_nodes_attrs",
    ),
]


@pytest.mark.parametrize(("with_seg", "setup", "run"), USER_ACTIONS)
def test_failure_at_last_basic_action_rolls_back(
    get_tracks, tracks_state, monkeypatch, with_seg, setup, run
):
    dry_run = make_tracks(get_tracks, with_seg, setup)
    n_actions = count_basic_actions(dry_run, run)
    assert n_actions >= 2

    tracks = make_tracks(get_tracks, with_seg, setup)
    before = tracks_state(tracks, full=False)
    fail_at_basic_action(monkeypatch, n_actions)
    with pytest.raises(RuntimeError, match="injected failure"):
        run(tracks)
    assert tracks_state(tracks, full=False) == before


def test_set_division_onto_triple_division_rolls_back(get_tracks, tracks_state):
    """Making the division deletes (3, 4) and (4, 5) and adds (3, 5) before it
    reaches (1, 6), which it cannot delete because 1 is a three-way division."""
    tracks = make_tracks(get_tracks, with_seg=True, setup=None)
    # Make 1 a three-way division with the basic action, which does no checks
    AddEdge(tracks, (1, 6))
    before = tracks_state(tracks, full=False)
    with (
        pytest.warns(UserWarning, match="Removing conflicting edge"),
        pytest.raises(InvalidActionError, match="Expected degree of 0 or 1"),
    ):
        UserSetDivision(tracks, (3, 5, 6))
    assert tracks_state(tracks, full=False) == before


class _ApplyThenRaise(ActionGroup):
    def __init__(self, tracks, edges, raise_before_super=False):
        if raise_before_super:
            raise RuntimeError("before super")
        super().__init__(tracks, actions=[])
        for edge in edges:
            self.actions.append(DeleteEdge(tracks, edge))
        raise RuntimeError("after applying")


class _Nested(ActionGroup):
    def __init__(self, tracks):
        super().__init__(tracks, actions=[])
        self.actions.append(DeleteEdge(tracks, (4, 5)))
        self.actions.append(_ApplyThenRaise(tracks, [(3, 4)]))


class TestRollbackMechanics:
    def test_applied_actions_are_inverted(self, get_tracks, tracks_state):
        tracks = get_tracks(ndim=3, with_seg=False, prefill_track_ids=True)
        before = tracks_state(tracks, full=False)
        with pytest.raises(RuntimeError, match="after applying"):
            _ApplyThenRaise(tracks, [(3, 4), (4, 5)])
        assert tracks_state(tracks, full=False) == before

    def test_raise_before_super_init(self, get_tracks, tracks_state):
        tracks = get_tracks(ndim=3, with_seg=False, prefill_track_ids=True)
        before = tracks_state(tracks)
        with pytest.raises(RuntimeError, match="before super"):
            _ApplyThenRaise(tracks, [], raise_before_super=True)
        assert tracks_state(tracks) == before

    def test_nested_groups_roll_back_once(self, get_tracks, tracks_state):
        tracks = get_tracks(ndim=3, with_seg=False, prefill_track_ids=True)
        before = tracks_state(tracks, full=False)
        with pytest.raises(RuntimeError, match="after applying"):
            _Nested(tracks)
        assert tracks_state(tracks, full=False) == before
