import pytest

from funtracks.exceptions import InvalidActionError
from funtracks.user_actions import UserAddEdge, UserDeleteEdge


@pytest.mark.parametrize("ndim", [3, 4])
@pytest.mark.parametrize("with_seg", [True, False])
class TestUserAddDeleteEdge:
    def test_user_add_edge(self, get_tracks, ndim, with_seg):
        tracks = get_tracks(ndim=ndim, with_seg=with_seg, is_solution=True)
        # add an edge from 4 to 6 (will make 4 a division and 5 will need to relabel
        # track id)
        edge = (4, 6)
        old_child = 5
        old_track_id = tracks.get_track_id(old_child)
        assert not tracks.graph.has_edge(*edge)
        action = UserAddEdge(tracks, edge)
        assert tracks.graph.has_edge(*edge)
        assert tracks.get_track_id(old_child) != old_track_id

        inverse = action.inverse()
        assert not tracks.graph.has_edge(*edge)
        assert tracks.get_track_id(old_child) == old_track_id

        inverse.inverse()
        assert tracks.graph.has_edge(*edge)
        assert tracks.get_track_id(old_child) != old_track_id

    def test_user_add_merge_edge(self, get_tracks, ndim, with_seg):
        tracks = get_tracks(ndim=ndim, with_seg=with_seg, is_solution=True)
        # add an edge from 2 to 4 (there is already an edge from 3 to 4)
        edge = (2, 4)
        old_edge = (3, 4)
        assert not tracks.graph.has_edge(*edge)
        assert tracks.graph.has_edge(*old_edge)
        with pytest.raises(
            InvalidActionError, match="Cannot make a merge edge in a tracking solution"
        ):
            UserAddEdge(tracks, edge)
        with pytest.warns(
            UserWarning,
            match="Removing edge .* to add new edge without merging.",
        ):
            action = UserAddEdge(tracks, edge, force=True)
        assert tracks.graph.has_edge(*edge)
        assert not tracks.graph.has_edge(*old_edge)

        inverse = action.inverse()
        assert not tracks.graph.has_edge(*edge)
        assert tracks.graph.has_edge(*old_edge)

        inverse.inverse()
        assert tracks.graph.has_edge(*edge)
        assert not tracks.graph.has_edge(*old_edge)

    def test_user_delete_edge(self, get_tracks, ndim, with_seg):
        tracks = get_tracks(ndim=ndim, with_seg=with_seg, is_solution=True)
        # delete edge (1, 3). (1,2) is now not a division anymore
        edge = (1, 3)
        old_child = 2

        old_track_id = tracks.get_track_id(old_child)
        new_track_id = tracks.get_track_id(1)
        assert tracks.graph.has_edge(*edge)

        action = UserDeleteEdge(tracks, edge)
        assert not tracks.graph.has_edge(*edge)
        assert tracks.get_track_id(old_child) == new_track_id

        inverse = action.inverse()
        assert tracks.graph.has_edge(*edge)
        assert tracks.get_track_id(old_child) == old_track_id

        double_inv = inverse.inverse()
        assert not tracks.graph.has_edge(*edge)
        assert tracks.get_track_id(old_child) == new_track_id

        # TODO: error if edge doesn't exist?
        double_inv.inverse()

        # delete edge (3, 4). 4 and 5 should get new track id
        edge = (3, 4)
        old_child = 5

        old_track_id = tracks.get_track_id(old_child)
        assert tracks.graph.has_edge(*edge)

        action = UserDeleteEdge(tracks, edge)
        assert not tracks.graph.has_edge(*edge)
        assert tracks.get_track_id(old_child) != old_track_id

        inverse = action.inverse()
        assert tracks.graph.has_edge(*edge)
        assert tracks.get_track_id(old_child) == old_track_id

        inverse.inverse()
        assert not tracks.graph.has_edge(*edge)
        assert tracks.get_track_id(old_child) != old_track_id


def test_add_edge_missing_node(get_tracks):
    tracks = get_tracks(ndim=3, with_seg=True, is_solution=True)
    with pytest.raises(InvalidActionError, match="Source node .* not in solution yet"):
        UserAddEdge(tracks, (10, 11))
    with pytest.raises(InvalidActionError, match="Target node .* not in solution yet"):
        UserAddEdge(tracks, (1, 11))


def test_add_edge_triple_div(get_tracks):
    tracks = get_tracks(ndim=3, with_seg=True, is_solution=True)
    with pytest.raises(
        InvalidActionError, match="Expected degree of 0 or 1 before adding edge"
    ):
        UserAddEdge(tracks, (1, 6))


def test_delete_missing_edge(get_tracks):
    tracks = get_tracks(ndim=3, with_seg=True, is_solution=True)
    with pytest.raises(InvalidActionError, match="Edge .* not in solution"):
        UserDeleteEdge(tracks, (10, 11))


def test_delete_edge_triple_div(get_tracks):
    tracks = get_tracks(ndim=3, with_seg=True, is_solution=True)
    tracks.graph.add_edge(1, 6)
    with pytest.raises(
        InvalidActionError, match="Expected degree of 0 or 1 after removing edge"
    ):
        UserDeleteEdge(tracks, (1, 6))
