from collections import Counter

import funlib.persistence as fp
import numpy as np
import pytest

from funtracks import NxGraph, Project, TrackingGraph
from funtracks.features import FeatureSet
from funtracks.features.edge_features import IoU
from funtracks.features.node_features import Area
from funtracks.params import ProjectParams
from funtracks.user_actions import UserUpdateSegmentation


@pytest.mark.parametrize(
    "ndim",
    [
        3,
    ],
)
class TestUpdateNodeSeg:
    def get_project(self, request, ndim):
        params = ProjectParams()
        seg_name = "segmentation_2d" if ndim == 3 else "segmentation_3d"
        segmentation = request.getfixturevalue(seg_name)
        axis_names = ["t", "z", "y", "x"] if ndim == 4 else ["t", "y", "x"]
        seg = fp.Array(segmentation, axis_names=axis_names, voxel_size=None)

        gt_graph = self.get_gt_graph(request, ndim)
        features = FeatureSet(ndim=ndim, seg=True)
        cand_graph = TrackingGraph(NxGraph, gt_graph, features)
        return Project("test", params, segmentation=seg, cand_graph=cand_graph)

    def get_gt_graph(self, request, ndim):
        graph_name = "graph_2d" if ndim == 3 else "graph_3d"
        gt_graph = request.getfixturevalue(graph_name)
        return gt_graph

    def test_user_update_seg_smaller(self, request, ndim):
        project = self.get_project(request, ndim)
        graph = project.cand_graph
        node_id = 3
        edge = (1, 3)

        orig_pixels = project.get_pixels(node_id)
        orig_position = project.cand_graph.get_position(node_id)
        orig_area = project.cand_graph.get_feature_value(node_id, Area())
        orig_iou = project.cand_graph.get_feature_value(edge, IoU())

        # remove all but one pixel
        pixels_to_remove = tuple(orig_pixels[d][1:] for d in range(len(orig_pixels)))
        remaining_loc = tuple(orig_pixels[d][0] for d in range(len(orig_pixels)))
        new_position = [remaining_loc[1].item(), remaining_loc[2].item()]
        remaining_pixels = tuple(
            np.array([remaining_loc[d]]) for d in range(len(orig_pixels))
        )

        action = UserUpdateSegmentation(
            project, new_value=0, updated_pixels=[(pixels_to_remove, node_id)]
        )
        assert graph.has_node(node_id)
        assert graph.get_feature_value(node_id, graph.features.node_selection_pin) is True
        assert self.pixel_equals(project.get_pixels(node_id), remaining_pixels)
        assert graph.get_feature_value(node_id, graph.features.position) == new_position
        assert graph.get_feature_value(node_id, Area()) == 1
        assert graph.get_feature_value(edge, IoU()) == pytest.approx(0.0, abs=0.001)

        inverse = action.inverse()
        assert graph.has_node(node_id)
        assert graph.get_feature_value(node_id, graph.features.node_selection_pin) is None
        assert self.pixel_equals(project.get_pixels(node_id), orig_pixels)
        assert graph.get_feature_value(node_id, graph.features.position) == orig_position
        assert graph.get_feature_value(node_id, Area()) == orig_area
        assert graph.get_feature_value(edge, IoU()) == pytest.approx(orig_iou, abs=0.01)

        inverse.inverse()
        assert graph.has_node(node_id)
        assert graph.get_feature_value(node_id, graph.features.node_selection_pin) is True
        assert self.pixel_equals(project.get_pixels(node_id), remaining_pixels)
        assert graph.get_feature_value(node_id, graph.features.position) == new_position
        assert graph.get_feature_value(node_id, Area()) == 1
        assert graph.get_feature_value(edge, IoU()) == pytest.approx(0.0, abs=0.001)

    def pixel_equals(self, pixels1, pixels2):
        return Counter(zip(*pixels1)) == Counter(zip(*pixels2))

    def test_user_update_seg_bigger(self, request, ndim):
        project = self.get_project(request, ndim)
        graph = project.cand_graph
        node_id = 3
        edge = (1, 3)

        orig_pixels = project.get_pixels(node_id)
        orig_position = project.cand_graph.get_position(node_id)
        orig_area = project.cand_graph.get_feature_value(node_id, Area())
        orig_iou = project.cand_graph.get_feature_value(edge, IoU())

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
            project, new_value=3, updated_pixels=[(pixels_to_add, 0)]
        )
        assert graph.has_node(node_id)
        assert self.pixel_equals(all_pixels, project.get_pixels(node_id))
        assert graph.get_feature_value(node_id, graph.features.node_selection_pin) is True
        assert graph.get_feature_value(node_id, Area()) == orig_area + 1
        assert graph.get_feature_value(edge, IoU()) != orig_iou

        inverse = action.inverse()
        assert graph.has_node(node_id)
        assert graph.get_feature_value(node_id, graph.features.node_selection_pin) is None
        assert self.pixel_equals(orig_pixels, project.get_pixels(node_id))
        assert graph.get_feature_value(node_id, graph.features.position) == orig_position
        assert graph.get_feature_value(node_id, Area()) == orig_area
        assert graph.get_feature_value(edge, IoU()) == pytest.approx(orig_iou, abs=0.01)

        inverse.inverse()
        assert graph.has_node(node_id)
        assert graph.get_feature_value(node_id, graph.features.node_selection_pin) is True
        assert self.pixel_equals(all_pixels, project.get_pixels(node_id))
        assert graph.get_feature_value(node_id, Area()) == orig_area + 1
        assert graph.get_feature_value(edge, IoU()) != orig_iou

    def test_user_erase_seg(self, request, ndim):
        project = self.get_project(request, ndim)
        graph = project.cand_graph
        node_id = 3
        edge = (1, 3)

        orig_pixels = project.get_pixels(node_id)
        orig_position = project.cand_graph.get_position(node_id)
        orig_area = project.cand_graph.get_feature_value(node_id, Area())
        orig_iou = project.cand_graph.get_feature_value(edge, IoU())

        # remove all pixels
        pixels_to_remove = orig_pixels
        # set the pixels in the array first
        # (to reflect that the user directly changes the segmentation array)
        project.set_pixels(pixels_to_remove, 0)
        action = UserUpdateSegmentation(
            project, new_value=0, updated_pixels=[(pixels_to_remove, node_id)]
        )
        assert not graph.has_node(node_id)

        project.set_pixels(pixels_to_remove, node_id)
        inverse = action.inverse()
        assert graph.has_node(node_id)
        assert graph.get_feature_value(node_id, graph.features.node_selection_pin) is None
        self.pixel_equals(project.get_pixels(node_id), orig_pixels)
        assert graph.get_feature_value(node_id, graph.features.position) == orig_position
        assert graph.get_feature_value(node_id, Area()) == orig_area
        assert graph.get_feature_value(edge, IoU()) == pytest.approx(orig_iou, abs=0.01)

        project.set_pixels(pixels_to_remove, 0)
        inverse.inverse()
        assert not graph.has_node(node_id)

    def test_user_add_seg(self, request, ndim):
        project = self.get_project(request, ndim)
        graph = project.cand_graph
        # draw a new node just like node 6 but in time 3 (instead of 4)
        old_node_id = 6
        node_id = 7
        time = 3

        # TODO: add candidate edges when you add nodes to candidate graph
        cand_edge = (7, 6)

        pixels_to_add = project.get_pixels(old_node_id)
        pixels_to_add = (
            np.ones(shape=(pixels_to_add[0].shape), dtype=np.uint32) * time,
            *pixels_to_add[1:],
        )
        position = project.cand_graph.get_position(old_node_id)
        area = project.cand_graph.get_feature_value(old_node_id, Area())
        expected_cand_iou = 1.0

        assert not graph.has_node(node_id)

        assert np.sum(project.segmentation.data == node_id).compute() == 0
        project.set_pixels(pixels_to_add, node_id)
        action = UserUpdateSegmentation(
            project, new_value=node_id, updated_pixels=[(pixels_to_add, 0)]
        )
        assert np.sum(project.segmentation.data == node_id) == len(pixels_to_add[0])
        assert graph.has_node(node_id)
        assert project.solution.has_node(node_id)
        assert graph.get_feature_value(node_id, graph.features.node_selection_pin) is True
        assert graph.get_feature_value(node_id, graph.features.position) == position
        assert graph.get_feature_value(node_id, Area()) == area
        assert graph.get_feature_value(cand_edge, IoU()) == pytest.approx(
            expected_cand_iou, abs=0.01
        )

        inverse = action.inverse()
        assert not graph.has_node(node_id)

        inverse.inverse()
        assert graph.has_node(node_id)
        assert project.solution.has_node(node_id)
        assert graph.get_feature_value(node_id, graph.features.node_selection_pin) is True
        assert graph.get_feature_value(node_id, graph.features.position) == position
        assert graph.get_feature_value(node_id, Area()) == area
        assert graph.get_feature_value(cand_edge, IoU()) == pytest.approx(
            expected_cand_iou, abs=0.01
        )
