from collections import Counter

import numpy as np
import pytest

from funtracks.exceptions import InvalidActionError
from funtracks.user_actions import UserUpdateSegmentation
from funtracks.utils.tracksdata_utils import td_mask_to_pixels

iou_key = "iou"
area_key = "area"


# TODO: add area to the 4d testing graph
@pytest.mark.parametrize(
    "ndim",
    [3],
)
class TestUpdateNodeSeg:
    def pixels_equal_mask(self, pixels, tracks, node_id):
        mask_pixels = td_mask_to_pixels(
            tracks.get_mask(node_id), tracks.get_time(node_id), ndim=tracks.ndim
        )
        return Counter(zip(*pixels, strict=True)) == Counter(
            zip(*mask_pixels, strict=True)
        )

    def test_user_update_seg_smaller(self, get_tracks, ndim):
        tracks = get_tracks(ndim=ndim, with_seg=True, prefill_track_ids=True)
        node_id = 3
        edge = (1, 3)

        orig_pixels = td_mask_to_pixels(
            tracks.get_mask(node_id), tracks.get_time(node_id), ndim=tracks.ndim
        )
        orig_position = tracks.get_position(node_id)
        orig_area = tracks.get_node_attr(node_id, area_key)
        orig_iou = tracks.get_edge_attr(edge, iou_key)

        # remove all but one pixel
        pixels_to_remove = tuple(orig_pixels[d][1:] for d in range(len(orig_pixels)))
        remaining_loc = tuple(orig_pixels[d][0] for d in range(len(orig_pixels)))
        new_position = [remaining_loc[1].item(), remaining_loc[2].item()]
        remaining_pixels = tuple(
            np.array([remaining_loc[d]]) for d in range(len(orig_pixels))
        )

        action = UserUpdateSegmentation(
            tracks,
            new_value=0,
            updated_pixels=[(pixels_to_remove, node_id)],
            current_track_id=1,
        )
        assert tracks.graph_solution.has_node(node_id)
        assert self.pixels_equal_mask(remaining_pixels, tracks, node_id)
        assert tracks.get_position(node_id) == new_position
        assert tracks.get_node_attr(node_id, "area") == 1
        assert tracks.get_edge_attr(edge, iou_key) == pytest.approx(0.0, abs=0.01)

        inverse = action.inverse()
        assert tracks.graph_solution.has_node(node_id)
        assert self.pixels_equal_mask(orig_pixels, tracks, node_id)
        assert tracks.get_position(node_id) == orig_position
        assert tracks.get_node_attr(node_id, "area") == orig_area
        assert tracks.get_edge_attr(edge, iou_key) == pytest.approx(orig_iou, abs=0.01)

        inverse.inverse()
        assert self.pixels_equal_mask(remaining_pixels, tracks, node_id)
        assert tracks.get_position(node_id) == new_position
        assert tracks.get_node_attr(node_id, "area") == 1
        assert tracks.get_edge_attr(edge, iou_key) == pytest.approx(0.0, abs=0.01)

    def test_user_update_seg_bigger(self, get_tracks, ndim):
        tracks = get_tracks(ndim=ndim, with_seg=True, prefill_track_ids=True)
        node_id = 3
        edge = (1, 3)

        orig_pixels = td_mask_to_pixels(
            tracks.get_mask(node_id), tracks.get_time(node_id), ndim=tracks.ndim
        )
        orig_position = tracks.get_position(node_id)
        orig_area = tracks.get_node_attr(node_id, "area")
        orig_iou = tracks.get_edge_attr(edge, iou_key)

        # add one pixel
        pixels_to_add = tuple(
            np.array([orig_pixels[d][0]]) for d in range(len(orig_pixels))
        )
        new_x_val = 10
        pixels_to_add = (*pixels_to_add[:-1], np.array([new_x_val]))
        all_pixels = tuple(
            np.concat([orig_pixels[d], pixels_to_add[d]]) for d in range(len(orig_pixels))
        )

        action = UserUpdateSegmentation(
            tracks, new_value=3, updated_pixels=[(pixels_to_add, 0)], current_track_id=1
        )
        assert tracks.graph_solution.has_node(node_id)
        assert self.pixels_equal_mask(all_pixels, tracks, node_id)
        assert tracks.get_node_attr(node_id, "area") == orig_area + 1
        assert tracks.get_edge_attr(edge, iou_key) != orig_iou

        inverse = action.inverse()
        assert tracks.graph_solution.has_node(node_id)
        assert self.pixels_equal_mask(orig_pixels, tracks, node_id)
        assert tracks.get_position(node_id) == orig_position
        assert tracks.get_node_attr(node_id, "area") == orig_area
        assert tracks.get_edge_attr(edge, iou_key) == pytest.approx(orig_iou, abs=0.01)

        inverse.inverse()
        assert tracks.graph_solution.has_node(node_id)
        assert self.pixels_equal_mask(all_pixels, tracks, node_id)
        assert tracks.get_node_attr(node_id, "area") == orig_area + 1
        assert tracks.get_edge_attr(edge, iou_key) != orig_iou

    def test_invalid_action_with_segmentation(self, get_tracks, ndim):
        tracks = get_tracks(ndim=ndim, with_seg=True, prefill_track_ids=True)
        node_id = 1

        # Paint on top of node 1 with track id 3: because of the downstream division, this
        # should raise an invalid action error.
        orig_pixels = td_mask_to_pixels(
            tracks.get_mask(node_id), tracks.get_time(node_id), ndim=tracks.ndim
        )

        pixels_to_add = tuple(
            np.array([orig_pixels[d][0]]) for d in range(len(orig_pixels))
        )
        new_value = 7

        # assert InvalidActionError is raised
        with pytest.raises(
            InvalidActionError,
            match="Cannot add node here - downstream division of parent detected.",
        ):
            UserUpdateSegmentation(
                tracks,
                new_value=new_value,
                updated_pixels=[(pixels_to_add, node_id)],
                current_track_id=3,
            )
        # because the existing nodes are only updated after the UserAddNode action is
        # applied (which does not happen if caught by the error), the original
        # segmentation should be unchanged.
        t, y, x = (a.item() for a in pixels_to_add)
        assert np.asarray(tracks.segmentation[t, y, x]) == node_id

        # If the action is forced, the segmentation for node 1 should be updated, and the
        # new node should be added.
        update_seg_action = UserUpdateSegmentation(
            tracks,
            new_value=new_value,
            updated_pixels=[(pixels_to_add, node_id)],
            current_track_id=3,
            force=True,
        )

        # assert that the segmentation now has the new value
        assert np.asarray(tracks.segmentation[t, y, x]) == new_value
        assert tracks.graph_solution.has_node(new_value)
        assert len(update_seg_action.actions) == 2  # one for adding a node,
        # and one for updating existing node 1

    def test_user_erase_seg(self, get_tracks, ndim):
        tracks = get_tracks(ndim=ndim, with_seg=True, prefill_track_ids=True)
        node_id = 3
        edge = (1, 3)

        orig_pixels = td_mask_to_pixels(
            tracks.get_mask(node_id), tracks.get_time(node_id), ndim=tracks.ndim
        )
        orig_position = tracks.get_position(node_id)
        orig_area = tracks.get_node_attr(node_id, "area")
        orig_iou = tracks.get_edge_attr(edge, iou_key)

        # remove all pixels
        pixels_to_remove = orig_pixels
        # setting of pixels no longer necessary, done in UpdateNodeSeg
        action = UserUpdateSegmentation(
            tracks,
            new_value=0,
            updated_pixels=[(pixels_to_remove, node_id)],
            current_track_id=1,
        )
        assert not tracks.graph_solution.has_node(node_id)

        inverse = action.inverse()
        assert tracks.graph_solution.has_node(node_id)
        self.pixels_equal_mask(orig_pixels, tracks, node_id)
        assert tracks.get_position(node_id) == orig_position
        assert tracks.get_node_attr(node_id, "area") == orig_area
        assert tracks.get_edge_attr(edge, iou_key) == pytest.approx(orig_iou, abs=0.01)

        inverse.inverse()
        assert not tracks.graph_solution.has_node(node_id)

    def test_user_erase_seg_history_size(self, get_tracks, ndim):
        """An erase via UserUpdateSegmentation must add exactly one history
        entry. Regression test for a bug where the nested UserDeleteNode
        also registered itself, leaving two entries per fill and corrupting
        undo behavior."""
        tracks = get_tracks(ndim=ndim, with_seg=True, prefill_track_ids=True)
        node_id = 6
        pixels = td_mask_to_pixels(
            tracks.get_mask(node_id), tracks.get_time(node_id), ndim=tracks.ndim
        )
        UserUpdateSegmentation(
            tracks,
            new_value=0,
            updated_pixels=[(pixels, node_id)],
            current_track_id=1,
        )
        assert len(tracks.action_history.undo_stack) == 1

    def test_user_two_erases_then_two_undos(self, get_tracks, ndim):
        """Two consecutive erases must both be reversible via
        tracks.action_history.undo(). Reproduces bug_paint_undo: the second
        undo crashed because the buggy history had a duplicate UserDeleteNode
        entry that tried to re-add an already-restored node."""
        tracks = get_tracks(ndim=ndim, with_seg=True, prefill_track_ids=True)
        pixels_5 = td_mask_to_pixels(
            tracks.get_mask(5), tracks.get_time(5), ndim=tracks.ndim
        )
        pixels_6 = td_mask_to_pixels(
            tracks.get_mask(6), tracks.get_time(6), ndim=tracks.ndim
        )

        UserUpdateSegmentation(
            tracks, new_value=0, updated_pixels=[(pixels_5, 5)], current_track_id=1
        )
        assert not tracks.graph_solution.has_node(5)

        UserUpdateSegmentation(
            tracks, new_value=0, updated_pixels=[(pixels_6, 6)], current_track_id=1
        )
        assert not tracks.graph_solution.has_node(6)

        assert tracks.action_history.undo() is True
        assert tracks.graph_solution.has_node(6)
        assert not tracks.graph_solution.has_node(5)

        assert tracks.action_history.undo() is True
        assert tracks.graph_solution.has_node(5)
        assert tracks.graph_solution.has_node(6)

    def test_user_add_seg(self, get_tracks, ndim):
        tracks = get_tracks(ndim=ndim, with_seg=True, prefill_track_ids=True)
        # draw a new node just like node 6 but in time 3 (instead of 4)
        old_node_id = 6
        node_id = 7
        time = 3

        pixels_to_add = td_mask_to_pixels(
            tracks.get_mask(old_node_id), tracks.get_time(old_node_id), ndim=tracks.ndim
        )
        pixels_to_add = (
            np.ones(shape=(pixels_to_add[0].shape), dtype=np.uint32) * time,
            *pixels_to_add[1:],
        )
        position = tracks.get_position(old_node_id)
        area = tracks.get_node_attr(old_node_id, "area")

        assert not tracks.graph_solution.has_node(node_id)

        assert np.sum(tracks.segmentation == node_id) == 0
        action = UserUpdateSegmentation(
            tracks,
            new_value=node_id,
            updated_pixels=[(pixels_to_add, 0)],
            current_track_id=10,
        )
        assert np.sum(np.asarray(tracks.segmentation) == node_id) == len(pixels_to_add[0])
        assert tracks.graph_solution.has_node(node_id)
        assert tracks.get_position(node_id) == position
        assert tracks.get_node_attr(node_id, "area") == area
        assert tracks.get_track_id(node_id) == 10

        inverse = action.inverse()
        assert not tracks.graph_solution.has_node(node_id)

        inverse.inverse()
        assert tracks.graph_solution.has_node(node_id)
        assert tracks.get_position(node_id) == position
        assert tracks.get_node_attr(node_id, "area") == area
        assert tracks.get_track_id(node_id) == 10


def test_missing_seg(get_tracks):
    tracks = get_tracks(ndim=3, with_seg=False, prefill_track_ids=True)
    with pytest.raises(ValueError, match="Cannot update non-existing segmentation"):
        UserUpdateSegmentation(tracks, 0, [], 1)
