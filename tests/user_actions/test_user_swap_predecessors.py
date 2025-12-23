import pytest

from funtracks.exceptions import InvalidActionError
from funtracks.user_actions import UserSwapPredecessors


@pytest.mark.parametrize("ndim", [3, 4])
@pytest.mark.parametrize("with_seg", [True, False])
class TestUserSwapPredecessors:
    @pytest.mark.parametrize("order", [(5, 6), (6, 5)])
    def test_user_swap_nodes_one_predecessor(self, get_tracks, ndim, with_seg, order):
        """Test swapping when one node has a predecessor and one doesn't."""
        tracks = get_tracks(ndim=ndim, with_seg=with_seg, is_solution=True)

        # Nodes 5 and 6 are both at time 4
        # Node 5 has predecessor 4, node 6 has no predecessor
        assert (4, 5) in tracks.graph.edges
        assert list(tracks.graph.predecessors(6)) == []
        old_track_id_5 = tracks.get_track_id(5)
        old_track_id_6 = tracks.get_track_id(6)

        action = UserSwapPredecessors(tracks, order)

        # After swap: edge should go from 4 to 6 instead of 4 to 5
        assert (4, 6) in tracks.graph.edges
        assert (4, 5) not in tracks.graph.edges

        # Track IDs should be updated
        assert tracks.get_track_id(6) == old_track_id_5
        assert tracks.get_track_id(5) != old_track_id_5

        # Test inverse
        action.inverse()
        assert (4, 5) in tracks.graph.edges
        assert (4, 6) not in tracks.graph.edges
        assert tracks.get_track_id(5) == old_track_id_5
        assert tracks.get_track_id(6) == old_track_id_6

    def test_user_swap_nodes_same_predecessor(self, get_tracks, ndim, with_seg):
        """Test swapping when both nodes have the same predecessor is a no-op."""
        tracks = get_tracks(ndim=ndim, with_seg=with_seg, is_solution=True)

        # Nodes 2 and 3 are both at time 1, both have predecessor 1
        assert (1, 2) in tracks.graph.edges
        assert (1, 3) in tracks.graph.edges
        old_track_id_2 = tracks.get_track_id(2)
        old_track_id_3 = tracks.get_track_id(3)

        action = UserSwapPredecessors(tracks, (2, 3))

        # No actions should be created since swapping would result in identical graph
        assert len(action.actions) == 0

        # Everything should remain unchanged
        assert (1, 2) in tracks.graph.edges
        assert (1, 3) in tracks.graph.edges
        assert tracks.get_track_id(2) == old_track_id_2
        assert tracks.get_track_id(3) == old_track_id_3

    def test_user_swap_nodes_different_predecessors(self, get_tracks, ndim, with_seg):
        """Test swapping when both nodes have different predecessors."""
        tracks = get_tracks(ndim=ndim, with_seg=with_seg, is_solution=True)

        # Add an edge from 2 to 6 so node 6 has a predecessor different from node 5's
        from funtracks.user_actions import UserAddEdge

        UserAddEdge(tracks, (2, 6))

        # Now: node 5 has predecessor 4, node 6 has predecessor 2
        assert (4, 5) in tracks.graph.edges
        assert (2, 6) in tracks.graph.edges
        old_track_id_5 = tracks.get_track_id(5)
        old_track_id_6 = tracks.get_track_id(6)

        action = UserSwapPredecessors(tracks, (5, 6))

        # After swap: edges should be swapped
        assert (4, 6) in tracks.graph.edges
        assert (2, 5) in tracks.graph.edges
        assert (4, 5) not in tracks.graph.edges
        assert (2, 6) not in tracks.graph.edges

        # Test inverse
        action.inverse()
        assert (4, 5) in tracks.graph.edges
        assert (2, 6) in tracks.graph.edges
        assert tracks.get_track_id(5) == old_track_id_5
        assert tracks.get_track_id(6) == old_track_id_6

    def test_user_swap_different_time_valid(self, get_tracks, ndim, with_seg):
        """Test swapping nodes at different time points when predecessors are valid."""
        tracks = get_tracks(ndim=ndim, with_seg=with_seg, is_solution=True)

        # Graph structure: 1(t=0) -> 2(t=1), 1(t=0) -> 3(t=1) -> 4(t=2) -> 5(t=4), 6(t=4)
        # We want to swap predecessors between nodes at different times.
        # Use node 4 (time 2, pred 3 at time 1) and node 6 (time 4, no pred)
        # After adding edge 2 -> 6: node 6 has pred 2 at time 1
        # pred of 4 (time 1) < node 6 (time 4) ✓
        # pred of 6 (time 1) < node 4 (time 2) ✓
        from funtracks.user_actions import UserAddEdge

        UserAddEdge(tracks, (2, 6))

        assert (3, 4) in tracks.graph.edges
        assert (2, 6) in tracks.graph.edges

        action = UserSwapPredecessors(tracks, (4, 6))

        # After swap: 3 -> 6, 2 -> 4
        assert (3, 6) in tracks.graph.edges
        assert (2, 4) in tracks.graph.edges
        assert (3, 4) not in tracks.graph.edges
        assert (2, 6) not in tracks.graph.edges

        # Inverse should restore
        action.inverse()
        assert (3, 4) in tracks.graph.edges
        assert (2, 6) in tracks.graph.edges

    def test_user_swap_different_time_invalid_raises(self, get_tracks, ndim, with_seg):
        """Test that swapping raises error when predecessor would not be before node."""
        tracks = get_tracks(ndim=ndim, with_seg=with_seg, is_solution=True)

        # Node 3 at time 1, pred 1 at time 0
        # Node 4 at time 2, pred 3 at time 1
        # pred of 3 (time 0) < node 4 (time 2) ✓
        # pred of 4 (time 1) >= node 3 (time 1) ✗ (equal, not strictly before)
        with pytest.raises(InvalidActionError, match="Cannot swap: predecessor"):
            UserSwapPredecessors(tracks, (3, 4))

    def test_user_swap_nodes_wrong_count_raises(self, get_tracks, ndim, with_seg):
        """Test that swapping with wrong number of nodes raises an error."""
        tracks = get_tracks(ndim=ndim, with_seg=with_seg, is_solution=True)

        with pytest.raises(
            InvalidActionError, match="You can only swap a pair of two nodes"
        ):
            UserSwapPredecessors(tracks, (1,))  # type: ignore[arg-type]

        with pytest.raises(
            InvalidActionError, match="You can only swap a pair of two nodes"
        ):
            UserSwapPredecessors(tracks, (1, 2, 3))  # type: ignore[arg-type]

    def test_user_swap_nodes_no_predecessors(self, get_tracks, ndim, with_seg):
        """Test swapping two nodes that both have no predecessors does nothing."""
        tracks = get_tracks(ndim=ndim, with_seg=with_seg, is_solution=True)

        # Create a scenario with two nodes at the same time with no predecessors
        # Nodes 2 and 3 are both at time 1, but both have predecessor 1
        # Node 6 has no predecessor. We need to find/create another node at time 4
        # with no predecessor. For now, let's just verify the action completes
        # without error when there's nothing to swap.

        # Delete edge (4, 5) so node 5 has no predecessor like node 6
        from funtracks.user_actions import UserDeleteEdge

        UserDeleteEdge(tracks, (4, 5))

        # Now both 5 and 6 have no predecessors
        old_track_id_5 = tracks.get_track_id(5)
        old_track_id_6 = tracks.get_track_id(6)

        action = UserSwapPredecessors(tracks, (5, 6))

        # Nothing should change since neither has a predecessor
        assert tracks.get_track_id(5) == old_track_id_5
        assert tracks.get_track_id(6) == old_track_id_6
        assert len(action.actions) == 0
