import pytest

from funtracks.user_actions import UserAddNode


def test_user_force_add_downstream(get_tracks):
    """Test force adding a node of which the track id has an upstream division event.
    Should break the edges of the division event to allow this new edge."""

    tracks = get_tracks(ndim=3, with_seg=False, is_solution=True)

    # upstream division, with force
    attrs = {"t": 2, "track_id": 1, "pos": [3, 4]}
    UserAddNode(tracks, node=7, attributes=attrs, force=True)
    assert tracks.get_track_id(7) == 1
    assert (1, 2) not in tracks.graph.edges
    assert (1, 3) not in tracks.graph.edges
    assert (1, 7) in tracks.graph.edges


def test_user_force_add_upstream(get_tracks):
    """Test force adding a node upstream, of which the track id co-exists with the parent
    track id. Should break the edge with the parent track to allow this new edge."""

    tracks = get_tracks(ndim=3, with_seg=False, is_solution=True)

    # downstream parent division, with force
    attrs = {"t": 0, "track_id": 3, "pos": [3, 4]}
    UserAddNode(tracks, node=7, attributes=attrs, force=True)
    assert tracks.get_track_id(7) == 3
    assert (1, 2) in tracks.graph.edges  # still there
    assert (1, 3) not in tracks.graph.edges  # should be removed
    assert (7, 3) in tracks.graph.edges  # new forced edge


def test_auto_assign_new_track_id(get_tracks):
    """Test that adding a node with a track id that already exists at the current time
    point raises a warning and auto-assigns a new track id instead."""

    tracks = get_tracks(ndim=3, with_seg=False, is_solution=True)

    # existing track id at current time --> allowed, with warning
    with pytest.warns(UserWarning, match="Starting a new track, because track id"):
        attrs = {"t": 1, "track_id": 2, "pos": [3, 4]}  # combination exists already
        UserAddNode(tracks, node=7, attributes=attrs)

        assert 7 in tracks.graph.nodes
        assert tracks.get_track_id(7) == 6  # new assigned track id
