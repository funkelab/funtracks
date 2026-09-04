"""Tests for UserConnectNodes.

The fixture graph is::

    1 (t=0) --> 2 (t=1)
      \\
       --> 3 (t=1) --> 4 (t=2) --> 5 (t=4)

    6 (t=4) is isolated.
"""

import pytest

from funtracks.exceptions import InvalidActionError
from funtracks.user_actions import UserConnectNodes


@pytest.fixture
def tracks(get_tracks):
    return get_tracks(ndim=3, with_seg=False, prefill_track_ids=True)


class TestConnect:
    def test_connect_skip_edge(self, tracks):
        """Nodes do not need to be in consecutive time points."""

        assert not tracks.graph_solution.has_edge(2, 6)
        track_id = tracks.get_track_id(2)

        action = UserConnectNodes(tracks, [2, 6])
        assert tracks.graph_solution.has_edge(2, 6)
        # the tracklet id of the first node in time propagates to the whole chain
        assert tracks.get_track_id(6) == track_id

        action.inverse()
        assert not tracks.graph_solution.has_edge(2, 6)

    def test_connect_chain_propagates_first_track_id(self, tracks):
        """Connecting several nodes gives them all the first node's tracklet id."""

        # break 3 -> 4 -> 5 apart first so that we can reconnect it as a chain
        UserConnectNodes(tracks, [3, 4, 5])
        assert len({tracks.get_track_id(n) for n in (3, 4, 5)}) == 3

        UserConnectNodes(tracks, [3, 4, 5])
        assert tracks.graph_solution.has_edge(3, 4)
        assert tracks.graph_solution.has_edge(4, 5)
        track_id = tracks.get_track_id(3)
        assert tracks.get_track_id(4) == track_id
        assert tracks.get_track_id(5) == track_id

    def test_connect_selection_order_is_irrelevant(self, tracks):
        """The nodes are sorted by time, not by selection order."""

        UserConnectNodes(tracks, [6, 2])
        assert tracks.graph_solution.has_edge(2, 6)

    def test_connect_creates_division(self, tracks):
        """A source with a single existing successor becomes a division."""

        action = UserConnectNodes(tracks, [4, 6])
        assert tracks.graph_solution.has_edge(4, 6)
        assert sorted(tracks.successors(4)) == [5, 6]

        action.inverse()
        assert not tracks.graph_solution.has_edge(4, 6)

    def test_partially_connected_still_connects(self, tracks):
        """If one pair is not connected yet, nothing gets disconnected."""

        UserConnectNodes(tracks, [3, 4, 6])
        assert tracks.graph_solution.has_edge(3, 4)  # existing edge is kept
        assert tracks.graph_solution.has_edge(4, 6)  # missing edge is added


class TestDisconnect:
    def test_disconnect_fully_connected_selection(self, tracks):
        """Selecting an already connected chain breaks it apart again."""

        action = UserConnectNodes(tracks, [3, 4, 5])
        assert not tracks.graph_solution.has_edge(3, 4)
        assert not tracks.graph_solution.has_edge(4, 5)
        track_ids = [tracks.get_track_id(n) for n in (3, 4, 5)]
        assert len(set(track_ids)) == 3

        action.inverse()
        assert tracks.graph_solution.has_edge(3, 4)
        assert tracks.graph_solution.has_edge(4, 5)

    def test_disconnect_keeps_edges_outside_selection(self, tracks):
        """Only the edges between the selected nodes are removed."""

        UserConnectNodes(tracks, [4, 5])
        assert not tracks.graph_solution.has_edge(4, 5)
        assert tracks.graph_solution.has_edge(3, 4)  # incoming edge of the first node


class TestConflicts:
    def test_merge_conflict_is_forceable(self, tracks):
        """Node 4 already has 3 as a parent, connecting 2 -> 4 conflicts."""

        with pytest.raises(InvalidActionError, match="conflict") as excinfo:
            UserConnectNodes(tracks, [2, 4])
        assert excinfo.value.forceable is True
        assert tracks.graph_solution.has_edge(3, 4)  # nothing was applied

        with pytest.warns(UserWarning, match="Removing edge"):
            action = UserConnectNodes(tracks, [2, 4], force=True)
        assert tracks.graph_solution.has_edge(2, 4)
        assert not tracks.graph_solution.has_edge(3, 4)

        action.inverse()
        assert not tracks.graph_solution.has_edge(2, 4)
        assert tracks.graph_solution.has_edge(3, 4)

    def test_three_way_division_is_forceable(self, tracks):
        """Node 1 already has two children, so a third one conflicts."""

        with pytest.raises(InvalidActionError, match="conflict") as excinfo:
            UserConnectNodes(tracks, [1, 6])
        assert excinfo.value.forceable is True

        with pytest.warns(UserWarning, match="Removing edge"):
            UserConnectNodes(tracks, [1, 6], force=True)
        assert tracks.successors(1) == [6]

    def test_nothing_applied_when_a_later_pair_conflicts(self, tracks):
        """Conflicts are detected up front, so the graph is left untouched."""

        # 2 -> 6 would be fine, but 6 -> ... is not part of it: use 2 -> 4 in a chain
        with pytest.raises(InvalidActionError):
            UserConnectNodes(tracks, [1, 2, 4])
        assert not tracks.graph_solution.has_edge(1, 4)
        assert not tracks.graph_solution.has_edge(2, 4)
        assert tracks.graph_solution.has_edge(3, 4)
        assert tracks.graph_solution.has_edge(1, 2)

    def test_forced_chain_removes_all_conflicting_edges(self, tracks):
        """Forcing removes every conflicting edge in one action."""

        with pytest.warns(UserWarning, match="Removing edge"):
            action = UserConnectNodes(tracks, [1, 2, 4], force=True)
        assert tracks.graph_solution.has_edge(1, 2)
        assert tracks.graph_solution.has_edge(2, 4)
        assert not tracks.graph_solution.has_edge(3, 4)
        # edges that do not conflict with the chain are left alone
        assert tracks.graph_solution.has_edge(1, 3)

        action.inverse()
        assert tracks.graph_solution.has_edge(3, 4)
        assert not tracks.graph_solution.has_edge(2, 4)


class TestInvalid:
    def test_horizontal_nodes_not_forceable(self, tracks):
        """Nodes 2 and 3 are both in time point 1."""

        for force in (False, True):
            with pytest.raises(InvalidActionError, match="same time point") as excinfo:
                UserConnectNodes(tracks, [2, 3], force=force)
            assert excinfo.value.forceable is False

    def test_horizontal_nodes_in_longer_selection(self, tracks):
        with pytest.raises(InvalidActionError, match="same time point"):
            UserConnectNodes(tracks, [1, 2, 3, 4])

    def test_too_few_nodes(self, tracks):
        with pytest.raises(InvalidActionError, match="at least two nodes"):
            UserConnectNodes(tracks, [1])
        with pytest.raises(InvalidActionError, match="at least two nodes"):
            UserConnectNodes(tracks, [1, 1])

    def test_node_not_in_solution(self, tracks):
        with pytest.raises(InvalidActionError, match="not in solution"):
            UserConnectNodes(tracks, [1, 42])


def test_undo_redo_through_history(tracks):
    """The whole connect is undone and redone as a single history entry."""

    UserConnectNodes(tracks, [2, 6])
    assert tracks.graph_solution.has_edge(2, 6)

    tracks.undo()
    assert not tracks.graph_solution.has_edge(2, 6)

    tracks.redo()
    assert tracks.graph_solution.has_edge(2, 6)


class TestLinear:
    def test_linear_treats_existing_child_as_conflict(self, tracks):
        """Node 4 already has child 5, so connecting 4 -> 6 linearly conflicts."""

        # with divisions this is a plain, non-conflicting division
        assert UserConnectNodes.has_division_choice(tracks, [4, 6]) is True

        with pytest.raises(InvalidActionError, match="conflict") as excinfo:
            UserConnectNodes(tracks, [4, 6], linear=True)
        assert excinfo.value.forceable is True
        assert tracks.graph_solution.has_edge(4, 5)  # nothing was applied

        with pytest.warns(UserWarning, match="Removing edge"):
            action = UserConnectNodes(tracks, [4, 6], linear=True, force=True)
        assert tracks.graph_solution.has_edge(4, 6)
        assert not tracks.graph_solution.has_edge(4, 5)
        assert tracks.successors(4) == [6]

        action.inverse()
        assert tracks.graph_solution.has_edge(4, 5)
        assert not tracks.graph_solution.has_edge(4, 6)

    def test_linear_removes_division_on_an_already_connected_pair(self, tracks):
        """Sources keep no siblings anywhere in the chain, not just on new edges."""

        # 1 -> 2 already exists, 2 -> 4 is new, and 1 -> 3 makes 1 a division
        with pytest.warns(UserWarning, match="Removing edge"):
            UserConnectNodes(tracks, [1, 2, 4], linear=True, force=True)
        assert tracks.graph_solution.has_edge(1, 2)
        assert tracks.graph_solution.has_edge(2, 4)
        assert not tracks.graph_solution.has_edge(1, 3)  # division removed
        assert not tracks.graph_solution.has_edge(3, 4)  # merge removed
        assert tracks.successors(1) == [2]

    def test_linear_keeps_outgoing_edges_of_the_last_node(self, tracks):
        """The last node gets no new edge, so its children are left alone."""

        assert UserConnectNodes.has_division_choice(tracks, [2, 4]) is False

        with pytest.warns(UserWarning, match="Removing edge"):
            UserConnectNodes(tracks, [2, 4], linear=True, force=True)
        assert tracks.graph_solution.has_edge(2, 4)
        assert tracks.graph_solution.has_edge(4, 5)  # untouched

    def test_linear_and_divisions_agree_without_siblings(self, tracks):
        """Without existing outgoing edges both modes do exactly the same."""

        assert UserConnectNodes.has_division_choice(tracks, [2, 6]) is False

        UserConnectNodes(tracks, [2, 6], linear=True)
        assert tracks.graph_solution.has_edge(2, 6)
        assert tracks.get_track_id(6) == tracks.get_track_id(2)


class TestHasDivisionChoice:
    def test_false_for_fully_connected_selection(self, tracks):
        """A selection that would be disconnected has nothing to choose."""

        assert UserConnectNodes.has_division_choice(tracks, [3, 4, 5]) is False

    def test_false_for_invalid_selection(self, tracks):
        assert UserConnectNodes.has_division_choice(tracks, [2, 3]) is False  # same time
        assert UserConnectNodes.has_division_choice(tracks, [1]) is False
        assert UserConnectNodes.has_division_choice(tracks, [1, 42]) is False

    def test_false_when_both_modes_conflict_equally(self, tracks):
        """Node 1 already has two children, so both modes remove both of them."""

        assert UserConnectNodes.has_division_choice(tracks, [1, 6]) is False
