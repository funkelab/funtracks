from collections import Counter

import numpy as np
import pytest

from funtracks.data_model import EdgeAttr, NodeAttr, SolutionTracks
from funtracks.user_actions import UserUpdateSegmentation


# TODO: add area to the 4d testing graph
@pytest.mark.parametrize(
    "ndim",
    [3],
)
class TestUpdateNodeSeg:
    def get_tracks(self, request, ndim) -> SolutionTracks:
        seg_name = "segmentation_2d" if ndim == 3 else "segmentation_3d"
        seg = request.getfixturevalue(seg_name)

        gt_graph = self.get_gt_graph(request, ndim)
        tracks = SolutionTracks(gt_graph, segmentation=seg, ndim=ndim)
        return tracks

    def get_gt_graph(self, request, ndim):
        graph_name = "graph_2d" if ndim == 3 else "graph_3d"
        gt_graph = request.getfixturevalue(graph_name)
        return gt_graph

    def test_user_update_seg_smaller(self, request, ndim):
        tracks: SolutionTracks = self.get_tracks(request, ndim)
        node_id = 3
        edge = (1, 3)

        orig_pixels = tracks.get_pixels(node_id)
        orig_position = tracks.get_position(node_id)
        orig_area = tracks.get_node_attr(node_id, NodeAttr.AREA.value)
        orig_iou = tracks.get_edge_attr(edge, EdgeAttr.IOU.value)

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
        assert tracks.graph.has_node(node_id)
        assert self.pixel_equals(tracks.get_pixels(node_id), remaining_pixels)
        assert tracks.get_position(node_id) == new_position
        assert tracks.get_area(node_id) == 1
        assert tracks.get_edge_attr(edge, EdgeAttr.IOU.value) == pytest.approx(
            0.0, abs=0.01
        )

        inverse = action.inverse()
        assert tracks.graph.has_node(node_id)
        assert self.pixel_equals(tracks.get_pixels(node_id), orig_pixels)
        assert tracks.get_position(node_id) == orig_position
        assert tracks.get_area(node_id) == orig_area
        assert tracks.get_edge_attr(edge, EdgeAttr.IOU.value) == pytest.approx(
            orig_iou, abs=0.01
        )

        inverse.inverse()
        assert self.pixel_equals(tracks.get_pixels(node_id), remaining_pixels)
        assert tracks.get_position(node_id) == new_position
        assert tracks.get_area(node_id) == 1
        assert tracks.get_edge_attr(edge, EdgeAttr.IOU.value) == pytest.approx(
            0.0, abs=0.01
        )

    def pixel_equals(self, pixels1, pixels2):
        return Counter(zip(*pixels1, strict=True)) == Counter(zip(*pixels2, strict=True))

    def test_user_update_seg_bigger(self, request, ndim):
        tracks: SolutionTracks = self.get_tracks(request, ndim)
        node_id = 3
        edge = (1, 3)

        orig_pixels = tracks.get_pixels(node_id)
        orig_position = tracks.get_position(node_id)
        orig_area = tracks.get_area(node_id)
        orig_iou = tracks.get_edge_attr(edge, EdgeAttr.IOU.value)

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
        assert tracks.graph.has_node(node_id)
        assert self.pixel_equals(all_pixels, tracks.get_pixels(node_id))
        assert tracks.get_area(node_id) == orig_area + 1
        assert tracks.get_edge_attr(edge, EdgeAttr.IOU.value) != orig_iou

        inverse = action.inverse()
        assert tracks.graph.has_node(node_id)
        assert self.pixel_equals(orig_pixels, tracks.get_pixels(node_id))
        assert tracks.get_position(node_id) == orig_position
        assert tracks.get_area(node_id) == orig_area
        assert tracks.get_edge_attr(edge, EdgeAttr.IOU.value) == pytest.approx(
            orig_iou, abs=0.01
        )

        inverse.inverse()
        assert tracks.graph.has_node(node_id)
        assert self.pixel_equals(all_pixels, tracks.get_pixels(node_id))
        assert tracks.get_area(node_id) == orig_area + 1
        assert tracks.get_edge_attr(edge, EdgeAttr.IOU.value) != orig_iou

    def test_user_erase_seg(self, request, ndim):
        tracks: SolutionTracks = self.get_tracks(request, ndim)
        node_id = 3
        edge = (1, 3)

        orig_pixels = tracks.get_pixels(node_id)
        orig_position = tracks.get_position(node_id)
        orig_area = tracks.get_area(node_id)
        orig_iou = tracks.get_edge_attr(edge, EdgeAttr.IOU.value)

        # remove all pixels
        pixels_to_remove = orig_pixels
        # set the pixels in the array first
        # (to reflect that the user directly changes the segmentation array)
        tracks.set_pixels(pixels_to_remove, 0)
        action = UserUpdateSegmentation(
            tracks,
            new_value=0,
            updated_pixels=[(pixels_to_remove, node_id)],
            current_track_id=1,
        )
        assert not tracks.graph.has_node(node_id)

        tracks.set_pixels(pixels_to_remove, node_id)
        inverse = action.inverse()
        assert tracks.graph.has_node(node_id)
        self.pixel_equals(tracks.get_pixels(node_id), orig_pixels)
        assert tracks.get_position(node_id) == orig_position
        assert tracks.get_area(node_id) == orig_area
        assert tracks.get_edge_attr(edge, EdgeAttr.IOU.value) == pytest.approx(
            orig_iou, abs=0.01
        )

        tracks.set_pixels(pixels_to_remove, 0)
        inverse.inverse()
        assert not tracks.graph.has_node(node_id)

    def test_user_add_seg(self, request, ndim):
        tracks: SolutionTracks = self.get_tracks(request, ndim)
        # draw a new node just like node 6 but in time 3 (instead of 4)
        old_node_id = 6
        node_id = 7
        time = 3

        pixels_to_add = tracks.get_pixels(old_node_id)
        pixels_to_add = (
            np.ones(shape=(pixels_to_add[0].shape), dtype=np.uint32) * time,
            *pixels_to_add[1:],
        )
        position = tracks.get_position(old_node_id)
        area = tracks.get_area(old_node_id)

        assert not tracks.graph.has_node(node_id)

        assert np.sum(tracks.segmentation == node_id) == 0
        tracks.set_pixels(pixels_to_add, node_id)
        action = UserUpdateSegmentation(
            tracks,
            new_value=node_id,
            updated_pixels=[(pixels_to_add, 0)],
            current_track_id=10,
        )
        assert np.sum(tracks.segmentation == node_id) == len(pixels_to_add[0])
        assert tracks.graph.has_node(node_id)
        assert tracks.get_position(node_id) == position
        assert tracks.get_area(node_id) == area
        assert tracks.get_track_id(node_id) == 10

        inverse = action.inverse()
        assert not tracks.graph.has_node(node_id)

        inverse.inverse()
        assert tracks.graph.has_node(node_id)
        assert tracks.get_position(node_id) == position
        assert tracks.get_area(node_id) == area
        assert tracks.get_track_id(node_id) == 10


def test_missing_seg(graph_2d):
    tracks = SolutionTracks(graph_2d, ndim=3)
    with pytest.raises(ValueError, match="Cannot update non-existing segmentation"):
        UserUpdateSegmentation(tracks, 0, [], 1)
