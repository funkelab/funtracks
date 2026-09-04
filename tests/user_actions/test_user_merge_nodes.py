import pytest

from funtracks.exceptions import InvalidActionError
from funtracks.user_actions import (
    UserMergeNodes,
    get_merge_groups,
    get_track_id_options,
)
from funtracks.utils.tracksdata_utils import td_mask_to_pixels

# Fixture graph: nodes 1 (t=0), 2 (t=1), 3 (t=1), 4 (t=2), 5 (t=4), 6 (t=4)
# Edges: 1 -> 2, 1 -> 3, 3 -> 4, 4 -> 5
# Track ids: 1 -> 1, 2 -> 2, 3 -> 3, 4 -> 3, 5 -> 3, 6 -> 5


def pixel_set(tracks, node):
    pixels = td_mask_to_pixels(
        tracks.get_mask(node), tracks.get_time(node), ndim=tracks.ndim
    )
    return set(zip(*pixels, strict=True))


@pytest.mark.parametrize("ndim", [3, 4])
class TestUserMergeNodes:
    def test_merge_pair(self, get_tracks, ndim):
        """Nodes 2 and 3 share t=1 and are merged into node 3."""
        tracks = get_tracks(ndim=ndim, with_seg=True, prefill_track_ids=True)

        pixels_2 = pixel_set(tracks, 2)
        pixels_3 = pixel_set(tracks, 3)

        action = UserMergeNodes(tracks, [2, 3], track_ids=3)
        assert action.kept_nodes == [3]

        assert not tracks.graph_solution.has_node(2)
        assert tracks.graph_solution.has_node(3)
        assert pixel_set(tracks, 3) == pixels_2 | pixels_3
        # node 1 no longer divides, so its tracklet continues through node 3
        assert tracks.get_track_id(3) == tracks.get_track_id(1)
        # node 3 keeps its own edges, node 2's edge is gone
        assert tracks.graph_solution.has_edge(1, 3)
        assert tracks.graph_solution.has_edge(3, 4)
        assert not tracks.graph_solution.has_edge(1, 2)

        action.inverse()
        assert tracks.graph_solution.has_node(2)
        assert tracks.graph_solution.has_edge(1, 2)
        assert pixel_set(tracks, 3) == pixels_3
        assert pixel_set(tracks, 2) == pixels_2

    def test_merge_keeps_other_track_id(self, get_tracks, ndim):
        """The same pair merged into node 2 instead, by picking its tracklet id."""
        tracks = get_tracks(ndim=ndim, with_seg=True, prefill_track_ids=True)

        pixels_2 = pixel_set(tracks, 2)
        pixels_3 = pixel_set(tracks, 3)

        action = UserMergeNodes(tracks, [2, 3], track_ids=2)
        assert action.kept_nodes == [2]

        assert tracks.graph_solution.has_node(2)
        assert not tracks.graph_solution.has_node(3)
        assert pixel_set(tracks, 2) == pixels_2 | pixels_3
        assert tracks.graph_solution.has_edge(1, 2)
        assert not tracks.graph_solution.has_edge(3, 4)
        # node 1 no longer divides, so its tracklet continues through node 2
        assert tracks.get_track_id(2) == tracks.get_track_id(1)

    def test_merge_three_nodes(self, get_tracks, ndim):
        """A group of any size is merged in one go."""
        tracks = get_tracks(ndim=ndim, with_seg=True, prefill_track_ids=True)
        # move node 6 to t=1 so that nodes 2, 3 and 6 all share a time point
        tracks._set_node_attr(6, tracks.features.time_key, 1)

        pixels = pixel_set(tracks, 2) | pixel_set(tracks, 3) | pixel_set(tracks, 6)
        UserMergeNodes(tracks, [2, 3, 6], track_ids=3)

        assert tracks.graph_solution.has_node(3)
        assert not tracks.graph_solution.has_node(2)
        assert not tracks.graph_solution.has_node(6)
        assert pixel_set(tracks, 3) == pixels

    def test_merge_multiple_groups(self, get_tracks, ndim):
        """Two horizontal sets in one action, each with its own tracklet id."""
        tracks = get_tracks(ndim=ndim, with_seg=True, prefill_track_ids=True)

        pixels_2, pixels_3 = pixel_set(tracks, 2), pixel_set(tracks, 3)
        pixels_5, pixels_6 = pixel_set(tracks, 5), pixel_set(tracks, 6)

        action = UserMergeNodes(tracks, [2, 3, 5, 6], track_ids={1: 3, 4: 5})
        assert action.kept_nodes == [3, 6]

        assert pixel_set(tracks, 3) == pixels_2 | pixels_3
        assert pixel_set(tracks, 6) == pixels_5 | pixels_6
        assert not tracks.graph_solution.has_node(2)
        assert not tracks.graph_solution.has_node(5)

        action.inverse()
        for node, pixels in ((2, pixels_2), (3, pixels_3), (5, pixels_5), (6, pixels_6)):
            assert tracks.graph_solution.has_node(node)
            assert pixel_set(tracks, node) == pixels

    def test_track_ids_resolved_before_merging(self, get_tracks, ndim):
        """Merging t=1 re-assigns track ids down the track, but the t=4 choice still
        refers to the track ids as they were when the user picked them."""
        tracks = get_tracks(ndim=ndim, with_seg=True, prefill_track_ids=True)
        # nodes 3, 4 and 5 all share track id 3; merging 2 and 3 collapses the
        # division at node 1, which renames that whole track to track id 1
        assert tracks.get_track_id(5) == 3

        action = UserMergeNodes(tracks, [2, 3, 5, 6], track_ids={1: 3, 4: 3})
        assert action.kept_nodes == [3, 5]
        assert not tracks.graph_solution.has_node(2)
        assert not tracks.graph_solution.has_node(6)
        assert tracks.get_track_id(5) == tracks.get_track_id(1)

    def test_lone_nodes_are_ignored(self, get_tracks, ndim):
        """Node 1 has no partner in t=0, so it is left untouched."""
        tracks = get_tracks(ndim=ndim, with_seg=True, prefill_track_ids=True)

        UserMergeNodes(tracks, [1, 2, 3], track_ids=3)
        assert tracks.graph_solution.has_node(1)
        assert tracks.graph_solution.has_node(3)
        assert not tracks.graph_solution.has_node(2)

    def test_default_track_id_is_lowest(self, get_tracks, ndim):
        tracks = get_tracks(ndim=ndim, with_seg=True, prefill_track_ids=True)

        action = UserMergeNodes(tracks, [2, 3])
        assert action.kept_nodes == [2]  # track id 2 < track id 3

    def test_too_few_nodes(self, get_tracks, ndim):
        tracks = get_tracks(ndim=ndim, with_seg=True, prefill_track_ids=True)
        with pytest.raises(InvalidActionError, match="at least two nodes"):
            UserMergeNodes(tracks, [2, 2])

    def test_no_shared_time_point(self, get_tracks, ndim):
        tracks = get_tracks(ndim=ndim, with_seg=True, prefill_track_ids=True)
        with pytest.raises(InvalidActionError, match="share a time point"):
            UserMergeNodes(tracks, [1, 2, 4])

    def test_node_not_in_solution(self, get_tracks, ndim):
        tracks = get_tracks(ndim=ndim, with_seg=True, prefill_track_ids=True)
        with pytest.raises(InvalidActionError, match="not in solution"):
            UserMergeNodes(tracks, [2, 99])

    def test_unknown_track_id(self, get_tracks, ndim):
        tracks = get_tracks(ndim=ndim, with_seg=True, prefill_track_ids=True)
        with pytest.raises(InvalidActionError, match="names 0 of the nodes"):
            UserMergeNodes(tracks, [2, 3], track_ids=99)

    def test_missing_track_id_for_group(self, get_tracks, ndim):
        tracks = get_tracks(ndim=ndim, with_seg=True, prefill_track_ids=True)
        with pytest.raises(InvalidActionError, match="No tracklet id given"):
            UserMergeNodes(tracks, [2, 3, 5, 6], track_ids={1: 3})

    def test_without_segmentation(self, get_tracks, ndim):
        tracks = get_tracks(ndim=ndim, with_seg=False, prefill_track_ids=True)
        with pytest.raises(InvalidActionError, match="without a segmentation"):
            UserMergeNodes(tracks, [2, 3])

    def test_get_merge_groups(self, get_tracks, ndim):
        tracks = get_tracks(ndim=ndim, with_seg=True, prefill_track_ids=True)
        assert get_merge_groups(tracks, [1, 2, 3, 5, 6]) == {1: [2, 3], 4: [5, 6]}

    def test_get_track_id_options(self, get_tracks, ndim):
        tracks = get_tracks(ndim=ndim, with_seg=True, prefill_track_ids=True)
        assert get_track_id_options(tracks, [2, 3, 5, 6]) == {1: [2, 3], 4: [3, 5]}
