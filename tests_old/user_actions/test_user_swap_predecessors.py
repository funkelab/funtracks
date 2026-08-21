import pytest

from funtracks.exceptions import InvalidActionError
from funtracks.user_actions import UserAddEdge, UserDeleteEdge, UserSwapPredecessors


@pytest.mark.parametrize("ndim", [3, 4])
@pytest.mark.parametrize("with_seg", [True, False])
class TestUserSwapPredecessors:
    @pytest.mark.parametrize("order", [(5, 6), (6, 5)])
    def test_one_predecessor(self, get_tracks, ndim, with_seg, order):
        """Test swapping when one node has a predecessor and one doesn't."""
        tracks = get_tracks(ndim=ndim, with_seg=with_seg, is_solution=True)

        # Node 5 (t=4) has pred 4, node 6 (t=4) has no pred
        assert tracks.graph.has_edge(4, 5)
        assert list(tracks.graph.predecessors(6)) == []
        old_track_id_5 = tracks.get_track_id(5)
        old_track_id_6 = tracks.get_track_id(6)

        action = UserSwapPredecessors(tracks, order)

        assert tracks.graph.has_edge(4, 6)
        assert not tracks.graph.has_edge(4, 5)
        assert tracks.get_track_id(6) == old_track_id_5
        assert tracks.get_track_id(5) != old_track_id_5

        action.inverse()
        assert tracks.graph.has_edge(4, 5)
        assert not tracks.graph.has_edge(4, 6)
        assert tracks.get_track_id(5) == old_track_id_5
        assert tracks.get_track_id(6) == old_track_id_6

    def test_same_predecessor_raises(self, get_tracks, ndim, with_seg):
        """Test error when both nodes have the same predecessor."""
        tracks = get_tracks(ndim=ndim, with_seg=with_seg, is_solution=True)

        # Nodes 2 and 3 both have predecessor 1
        with pytest.raises(InvalidActionError, match="same predecessor"):
            UserSwapPredecessors(tracks, (2, 3))

    def test_different_predecessors(self, get_tracks, ndim, with_seg):
        """Test swapping when both nodes have different predecessors."""
        tracks = get_tracks(ndim=ndim, with_seg=with_seg, is_solution=True)

        UserAddEdge(tracks, (2, 6))

        # Node 5 has pred 4, node 6 has pred 2
        old_track_id_5 = tracks.get_track_id(5)
        old_track_id_6 = tracks.get_track_id(6)

        action = UserSwapPredecessors(tracks, (5, 6))

        assert tracks.graph.has_edge(4, 6)
        assert tracks.graph.has_edge(2, 5)
        assert not tracks.graph.has_edge(4, 5)
        assert not tracks.graph.has_edge(2, 6)

        action.inverse()
        assert tracks.graph.has_edge(4, 5)
        assert tracks.graph.has_edge(2, 6)
        assert tracks.get_track_id(5) == old_track_id_5
        assert tracks.get_track_id(6) == old_track_id_6

    def test_different_times_valid(self, get_tracks, ndim, with_seg):
        """Test swapping nodes at different times when predecessors are valid."""
        tracks = get_tracks(ndim=ndim, with_seg=with_seg, is_solution=True)

        # Add edge 2->6 so node 6 (t=4) has pred 2 (t=1)
        # Node 4 (t=2) has pred 3 (t=1)
        # Both preds at t=1 are before both nodes (t=2 and t=4)
        UserAddEdge(tracks, (2, 6))

        action = UserSwapPredecessors(tracks, (4, 6))

        assert tracks.graph.has_edge(3, 6)
        assert tracks.graph.has_edge(2, 4)
        assert not tracks.graph.has_edge(3, 4)
        assert not tracks.graph.has_edge(2, 6)

        action.inverse()
        assert tracks.graph.has_edge(3, 4)
        assert tracks.graph.has_edge(2, 6)

    def test_different_times_invalid_raises(self, get_tracks, ndim, with_seg):
        """Test error when predecessor would not be before swapped node."""
        tracks = get_tracks(ndim=ndim, with_seg=with_seg, is_solution=True)

        # Node 3 (t=1) has pred 1 (t=0), node 4 (t=2) has pred 3 (t=1)
        # pred of 4 (t=1) is not before node 3 (t=1)
        with pytest.raises(InvalidActionError, match="Cannot swap: predecessor"):
            UserSwapPredecessors(tracks, (3, 4))

    def test_wrong_count_raises(self, get_tracks, ndim, with_seg):
        """Test error when not exactly two nodes provided."""
        tracks = get_tracks(ndim=ndim, with_seg=with_seg, is_solution=True)

        with pytest.raises(
            InvalidActionError, match="You can only swap a pair of two nodes"
        ):
            UserSwapPredecessors(tracks, (1,))  # type: ignore[arg-type]

        with pytest.raises(
            InvalidActionError, match="You can only swap a pair of two nodes"
        ):
            UserSwapPredecessors(tracks, (1, 2, 3))  # type: ignore[arg-type]

    def test_no_predecessors_raises(self, get_tracks, ndim, with_seg):
        """Test error when neither node has a predecessor."""
        tracks = get_tracks(ndim=ndim, with_seg=with_seg, is_solution=True)

        # Delete edge so node 5 has no predecessor like node 6
        UserDeleteEdge(tracks, (4, 5))

        with pytest.raises(InvalidActionError, match="neither node has a predecessor"):
            UserSwapPredecessors(tracks, (5, 6))
