import funlib.persistence as fp
import networkx as nx
import numpy as np
import pytest

from funtracks import NxGraph, Project, TrackingGraph
from funtracks.actions import AddNode, DeleteNode
from funtracks.features import FeatureSet
from funtracks.features.node_features import Area
from funtracks.params import ProjectParams


@pytest.mark.parametrize("ndim", [3, 4])
@pytest.mark.parametrize(
    ("use_seg", "use_graph"),
    [
        (True, False),
        (False, True),
        (True, True),
    ],
)
class TestAddDeleteNode:
    def get_project(self, request, ndim, use_seg, use_graph):
        params = ProjectParams()
        seg_name = "segmentation_2d" if ndim == 3 else "segmentation_3d"
        if use_seg:
            segmentation = request.getfixturevalue(seg_name)
            axis_names = ["t", "z", "y", "x"] if ndim == 4 else ["t", "y", "x"]
            seg = fp.Array(segmentation, axis_names=axis_names, voxel_size=None)
        else:
            seg = None

        gt_graph = self.get_gt_graph(request, ndim)
        if use_graph:
            features = FeatureSet(ndim=ndim, seg=use_seg)
            cand_graph = TrackingGraph(
                NxGraph, nx.create_empty_copy(gt_graph, with_data=True), features
            )
        else:
            cand_graph = None
        return Project("test", params, segmentation=seg, graph=cand_graph)

    def get_gt_graph(self, request, ndim):
        graph_name = "graph_2d" if ndim == 3 else "graph_3d"
        gt_graph = request.getfixturevalue(graph_name)
        return gt_graph

    def test_add_node(self, request, ndim, use_seg, use_graph):
        project = self.get_project(request, ndim, use_seg, use_graph)
        features = project.graph.features
        node_id = 7
        track_id = 3
        time = 3
        position = [50, 50, 50] if ndim == 4 else [50, 50]
        attributes = {
            features.track_id: track_id,
            features.position: position,
            features.time: time,
        }
        if use_seg:
            seg_copy = project.segmentation.data.copy().compute()
            seg_copy[time, *position] = node_id
            pixels = np.nonzero(seg_copy == node_id)
            del attributes[features.position]
        else:
            pixels = None
        graph = project.graph
        assert not graph.has_node(node_id)
        action = AddNode(project, node_id, attributes, pixels=pixels)
        assert graph.has_node(node_id)
        assert graph.get_feature_value(node_id, graph.features.position) == position
        assert graph.get_feature_value(node_id, graph.features.track_id) == track_id
        if use_seg:
            assert graph.get_feature_value(node_id, Area()) == 1

        inverse = action.inverse()
        assert not graph.has_node(node_id)
        inverse.inverse()
        assert graph.has_node(node_id)
        assert graph.get_feature_value(node_id, graph.features.position) == position
        assert graph.get_feature_value(node_id, graph.features.track_id) == track_id
        if use_seg:
            assert graph.get_feature_value(node_id, Area()) == 1
        # TODO: error if node already exists?

    def test_delete_node(self, request, ndim, use_seg, use_graph):
        project = self.get_project(request, ndim, use_seg, use_graph)
        features = project.graph.features
        if ndim == 4 and use_seg:
            for feature in features._features:
                if isinstance(feature, Area):
                    area_feature = feature
                    break
            project.graph.features._features.remove(area_feature)
        node_id = 6
        # track_id = 5
        # time = 4
        # position = [97.5, 97.5, 97.5] if ndim == 4 else [97.5, 97.5]
        graph = project.graph
        assert graph.has_node(node_id)
        action = DeleteNode(project, node_id)
        assert not graph.has_node(node_id)
        inverse = action.inverse()
        assert graph.has_node(node_id)
        inverse.inverse()
        assert not graph.has_node(node_id)
        # TODO: error if node doesn't exist?
