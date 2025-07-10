import funlib.persistence as fp
import numpy as np
import pytest

from funtracks import NxGraph, Project, TrackingGraph
from funtracks.actions import UpdateNodeSeg
from funtracks.features import FeatureSet
from funtracks.features.edge_features import IoU
from funtracks.features.node_features import Area
from funtracks.params import ProjectParams


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
        return Project("test", params, segmentation=seg, graph=cand_graph)

    def get_gt_graph(self, request, ndim):
        graph_name = "graph_2d" if ndim == 3 else "graph_3d"
        gt_graph = request.getfixturevalue(graph_name)
        return gt_graph

    def test_update_seg(self, request, ndim):
        project = self.get_project(request, ndim)
        graph = project.graph
        node_id = 3
        edge = (1, 3)

        orig_pixels = project.get_pixels(node_id)
        orig_position = project.graph.get_position(node_id)
        orig_area = project.graph.get_feature_value(node_id, Area())
        orig_iou = project.graph.get_feature_value(edge, IoU())

        # remove all but one pixel
        pixels_to_remove = tuple(orig_pixels[d][1:] for d in range(len(orig_pixels)))
        remaining_loc = tuple(orig_pixels[d][0] for d in range(len(orig_pixels)))
        new_position = [remaining_loc[1].item(), remaining_loc[2].item()]
        remaining_pixels = tuple(
            np.array([remaining_loc[d]]) for d in range(len(orig_pixels))
        )

        action = UpdateNodeSeg(project, node_id, pixels=pixels_to_remove, added=False)
        assert graph.has_node(node_id)
        assert project.get_pixels(node_id) == remaining_pixels
        assert graph.get_feature_value(node_id, graph.features.position) == new_position
        assert graph.get_feature_value(node_id, Area()) == 1
        assert graph.get_feature_value(edge, IoU()) == pytest.approx(0.0, abs=0.001)

        inverse = action.inverse()
        assert graph.has_node(node_id)
        np.testing.assert_array_almost_equal(project.get_pixels(node_id), orig_pixels)
        assert graph.get_feature_value(node_id, graph.features.position) == orig_position
        assert graph.get_feature_value(node_id, Area()) == orig_area
        assert graph.get_feature_value(edge, IoU()) == pytest.approx(orig_iou, abs=0.01)

        inverse.inverse()
        assert graph.has_node(node_id)
        np.testing.assert_array_almost_equal(
            project.get_pixels(node_id), remaining_pixels
        )
        assert graph.get_feature_value(node_id, graph.features.position) == new_position
        assert graph.get_feature_value(node_id, Area()) == 1
        assert graph.get_feature_value(edge, IoU()) == pytest.approx(0.0, abs=0.001)
