import pytest

from funtracks.exceptions import InvalidActionError
from funtracks.user_actions import UserAddEdge, UserSetDivision

# Fixture graph: nodes 1 (t=0), 2 (t=1), 3 (t=1), 4 (t=2), 5 (t=4), 6 (t=4)
# Edges: 1 -> 2, 1 -> 3, 3 -> 4, 4 -> 5


@pytest.mark.parametrize("ndim", [3, 4])
@pytest.mark.parametrize("with_seg", [True, False])
class TestUserSetDivision:
    @pytest.mark.parametrize("order", [(1, 2, 3), (2, 1, 3), (3, 2, 1)])
    def test_break_existing_division(self, get_tracks, ndim, with_seg, order):
        """Node 1 already divides into 2 and 3, so the division is broken."""
        tracks = get_tracks(ndim=ndim, with_seg=with_seg, prefill_track_ids=True)

        action = UserSetDivision(tracks, order)
        assert action.parent == 1
        assert action.breaking

        assert not tracks.graph_solution.has_edge(1, 2)
        assert not tracks.graph_solution.has_edge(1, 3)
        assert tracks.successors(1) == []
        assert tracks.predecessors(2) == []
        assert tracks.predecessors(3) == []
        # the children became separate lineages
        assert tracks.get_lineage_id(2) != tracks.get_lineage_id(1)
        assert tracks.get_lineage_id(3) != tracks.get_lineage_id(1)

        action.inverse()
        assert tracks.graph_solution.has_edge(1, 2)
        assert tracks.graph_solution.has_edge(1, 3)
        assert tracks.get_lineage_id(2) == tracks.get_lineage_id(1)
        assert tracks.get_lineage_id(3) == tracks.get_lineage_id(1)

    def test_make_division_no_existing_edges(self, get_tracks, ndim, with_seg):
        """Node 2 (t=1) has no children, 5 and 6 (t=4) get connected to it."""
        tracks = get_tracks(ndim=ndim, with_seg=with_seg, prefill_track_ids=True)

        old_track_id_2 = tracks.get_track_id(2)
        with pytest.warns(UserWarning, match="Removing conflicting edge"):
            action = UserSetDivision(tracks, (5, 2, 6))
        assert action.parent == 2
        assert not action.breaking

        assert tracks.graph_solution.has_edge(2, 5)
        assert tracks.graph_solution.has_edge(2, 6)
        # node 5's previous parent edge was in conflict and got removed
        assert not tracks.graph_solution.has_edge(4, 5)
        assert tracks.get_lineage_id(5) == tracks.get_lineage_id(2)
        assert tracks.get_lineage_id(6) == tracks.get_lineage_id(2)
        # both children get their own tracklet
        assert tracks.get_track_id(5) != old_track_id_2
        assert tracks.get_track_id(6) != old_track_id_2
        assert tracks.get_track_id(5) != tracks.get_track_id(6)

        action.inverse()
        assert not tracks.graph_solution.has_edge(2, 5)
        assert not tracks.graph_solution.has_edge(2, 6)
        assert tracks.graph_solution.has_edge(4, 5)
        assert tracks.get_track_id(2) == old_track_id_2

    def test_make_division_one_edge_exists(self, get_tracks, ndim, with_seg):
        """Node 4 -> 5 exists already, so only 4 -> 6 has to be added."""
        tracks = get_tracks(ndim=ndim, with_seg=with_seg, prefill_track_ids=True)

        old_track_id_4 = tracks.get_track_id(4)
        action = UserSetDivision(tracks, (4, 5, 6))
        assert action.parent == 4
        assert not action.breaking

        assert tracks.graph_solution.has_edge(4, 5)
        assert tracks.graph_solution.has_edge(4, 6)
        assert tracks.get_lineage_id(5) == tracks.get_lineage_id(4)
        assert tracks.get_lineage_id(6) == tracks.get_lineage_id(4)
        # node 5 shared the parent's tracklet before, now both children get new ones
        assert tracks.get_track_id(5) != old_track_id_4
        assert tracks.get_track_id(6) != old_track_id_4
        assert tracks.get_track_id(5) != tracks.get_track_id(6)

        action.inverse()
        assert tracks.graph_solution.has_edge(4, 5)
        assert not tracks.graph_solution.has_edge(4, 6)
        assert tracks.get_track_id(5) == old_track_id_4

    def test_make_division_toggles_back_and_forth(self, get_tracks, ndim, with_seg):
        """Running the action twice makes and then breaks the same division."""
        tracks = get_tracks(ndim=ndim, with_seg=with_seg, prefill_track_ids=True)

        UserSetDivision(tracks, (4, 5, 6))
        assert tracks.graph_solution.has_edge(4, 6)

        action = UserSetDivision(tracks, (4, 5, 6))
        assert action.breaking
        assert not tracks.graph_solution.has_edge(4, 5)
        assert not tracks.graph_solution.has_edge(4, 6)

    def test_make_division_removes_third_child(self, get_tracks, ndim, with_seg):
        """A parent that already has two children loses the non-selected one."""
        tracks = get_tracks(ndim=ndim, with_seg=with_seg, prefill_track_ids=True)

        # Node 1 (t=0) has children 2 and 3 (t=1); make it divide into 3 and 6 (t=4)
        with pytest.warns(UserWarning, match=r"Removing conflicting edge \(1, 2\)"):
            UserSetDivision(tracks, (1, 3, 6))

        assert tracks.graph_solution.has_edge(1, 3)
        assert tracks.graph_solution.has_edge(1, 6)
        assert not tracks.graph_solution.has_edge(1, 2)
        assert sorted(tracks.successors(1)) == [3, 6]

    def test_undo_redo_via_history(self, get_tracks, ndim, with_seg):
        """The whole trio is undone and redone as a single history entry."""
        tracks = get_tracks(ndim=ndim, with_seg=with_seg, prefill_track_ids=True)

        UserSetDivision(tracks, (4, 5, 6))
        assert tracks.graph_solution.has_edge(4, 6)

        assert tracks.undo()
        assert not tracks.graph_solution.has_edge(4, 6)
        assert tracks.graph_solution.has_edge(4, 5)

        assert tracks.redo()
        assert tracks.graph_solution.has_edge(4, 6)
        assert tracks.graph_solution.has_edge(4, 5)

    def test_wrong_number_of_nodes(self, get_tracks, ndim, with_seg):
        tracks = get_tracks(ndim=ndim, with_seg=with_seg, prefill_track_ids=True)

        with pytest.raises(InvalidActionError, match="exactly 3 distinct nodes"):
            UserSetDivision(tracks, (1, 2))
        with pytest.raises(InvalidActionError, match="exactly 3 distinct nodes"):
            UserSetDivision(tracks, (1, 2, 3, 4))
        with pytest.raises(InvalidActionError, match="exactly 3 distinct nodes"):
            UserSetDivision(tracks, (1, 2, 2))

    def test_no_unique_parent(self, get_tracks, ndim, with_seg):
        """Nodes 2 and 3 are both at t=1, so there is no single earliest node."""
        tracks = get_tracks(ndim=ndim, with_seg=with_seg, prefill_track_ids=True)

        with pytest.raises(InvalidActionError, match="exactly one node to be earlier"):
            UserSetDivision(tracks, (2, 3, 5))

    def test_daughter_between_parent_and_other_daughter(self, get_tracks, ndim, with_seg):
        """A chain 3 -> 4 -> 5 becomes a division of 3 into 4 and 5."""
        tracks = get_tracks(ndim=ndim, with_seg=with_seg, prefill_track_ids=True)

        with pytest.warns(UserWarning, match=r"Removing conflicting edge \(4, 5\)"):
            UserSetDivision(tracks, (3, 4, 5))

        assert tracks.graph_solution.has_edge(3, 4)
        assert tracks.graph_solution.has_edge(3, 5)
        assert not tracks.graph_solution.has_edge(4, 5)
        assert tracks.get_lineage_id(4) == tracks.get_lineage_id(3)
        assert tracks.get_lineage_id(5) == tracks.get_lineage_id(3)

    def test_break_only_when_both_edges_exist(self, get_tracks, ndim, with_seg):
        """A parent connected to only one child completes the division."""
        tracks = get_tracks(ndim=ndim, with_seg=with_seg, prefill_track_ids=True)

        # 1 -> 2 exists, 1 -> 6 does not
        UserAddEdge(tracks, (2, 6))  # give node 6 a parent that has to be removed
        with pytest.warns(UserWarning, match=r"Removing conflicting edge \(2, 6\)"):
            action = UserSetDivision(tracks, (1, 2, 6))

        assert not action.breaking
        assert tracks.graph_solution.has_edge(1, 2)
        assert tracks.graph_solution.has_edge(1, 6)
        assert not tracks.graph_solution.has_edge(2, 6)
