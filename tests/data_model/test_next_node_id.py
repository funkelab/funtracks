"""Tests for handing out node ids that are free and stay free."""

import pytest

from funtracks.user_actions import UserAddNode, UserDeleteNode, UserUpdateSegmentation
from funtracks.utils.tracksdata_utils import td_mask_to_pixels


@pytest.mark.parametrize("ndim", [3])
class TestGetNextNodeId:
    def test_is_free_and_above_every_existing_id(self, get_tracks, ndim):
        tracks = get_tracks(ndim=ndim, with_seg=True, prefill_track_ids=True)

        next_id = tracks.get_next_node_id()

        assert not tracks.graph_full.has_node(next_id)
        assert next_id > max(tracks.graph_full.node_ids())

    def test_is_stable_until_something_is_added(self, get_tracks, ndim):
        tracks = get_tracks(ndim=ndim, with_seg=True, prefill_track_ids=True)

        assert tracks.get_next_node_id() == tracks.get_next_node_id()

    def test_moves_on_once_the_id_is_used(self, get_tracks, ndim):
        # no segmentation: these nodes are added with a position only
        tracks = get_tracks(ndim=ndim, with_seg=False, prefill_track_ids=True)
        taken = tracks.get_next_node_id()

        UserAddNode(
            tracks,
            taken,
            attributes={
                "pos": [5, 8],
                tracks.features.time_key: 0,
                tracks.features.tracklet_key: tracks.get_next_track_id(),
            },
        )

        assert tracks.get_next_node_id() > taken

    def test_does_not_reuse_a_deleted_id(self, get_tracks, ndim):
        """A deleted node keeps its id, so handing it out again would collide."""
        tracks = get_tracks(ndim=ndim, with_seg=True, prefill_track_ids=True)
        node_id = 3

        UserDeleteNode(tracks, node_id)

        assert not tracks.graph_solution.has_node(node_id)
        assert tracks.get_next_node_id() != node_id
        assert tracks.get_next_node_id() > node_id

    def test_does_not_reuse_an_id_whose_add_was_undone(self, get_tracks, ndim):
        """Undoing an add soft-deletes the node; its id stays taken."""
        tracks = get_tracks(ndim=ndim, with_seg=True, prefill_track_ids=True)
        painted = tracks.get_next_node_id()
        pixels = td_mask_to_pixels(
            tracks.get_mask(3), tracks.get_time(3), ndim=tracks.ndim
        )

        UserUpdateSegmentation(
            tracks,
            new_value=painted,
            updated_pixels=[(pixels, 3)],
            current_track_id=tracks.get_next_track_id(),
        )
        tracks.undo()

        # the node is gone from the solution but still holds its id
        assert not tracks.graph_solution.has_node(painted)
        assert tracks.graph_full.has_node(painted)
        assert tracks.get_next_node_id() > painted

    def test_agrees_with_scanning_the_graph(self, get_tracks, ndim):
        """The maintained value must match what listing every id would give."""
        tracks = get_tracks(ndim=ndim, with_seg=False, prefill_track_ids=True)
        tracks.get_next_node_id()  # look the highest id up before mutating

        for _ in range(3):
            UserAddNode(
                tracks,
                tracks.get_next_node_id(),
                attributes={
                    "pos": [5, 8],
                    tracks.features.time_key: 0,
                    tracks.features.tracklet_key: tracks.get_next_track_id(),
                },
            )

        assert tracks.get_next_node_id() == max(tracks.graph_full.node_ids()) + 1

    def test_does_not_list_every_node_id_once_looked_up(self, get_tracks, ndim):
        """Listing ids is a full table query on a database-backed graph."""
        tracks = get_tracks(ndim=ndim, with_seg=True, prefill_track_ids=True)
        tracks.get_next_node_id()  # first call is allowed to look it up

        calls = []
        original = tracks.graph_full.node_ids
        tracks.graph_full.node_ids = lambda *a, **k: calls.append(1) or original(*a, **k)
        try:
            for _ in range(5):
                tracks.get_next_node_id()
        finally:
            tracks.graph_full.node_ids = original

        assert calls == []


def test_next_node_id_on_an_empty_graph(get_tracks):
    """An empty graph still has to produce a usable id."""
    tracks = get_tracks(ndim=3, with_seg=False)
    for node in list(tracks.graph_solution.node_ids()):
        UserDeleteNode(tracks, node)

    assert tracks.get_next_node_id() >= 0
    assert not tracks.graph_solution.has_node(tracks.get_next_node_id())
