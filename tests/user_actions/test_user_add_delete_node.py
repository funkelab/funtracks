import funlib.persistence as fp
import numpy as np
import pytest

from funtracks import NxGraph, Project, TrackingGraph
from funtracks.features import FeatureSet
from funtracks.features.node_features import Area
from funtracks.params import ProjectParams
from funtracks.user_actions import UserAddNode, UserDeleteNode


@pytest.mark.parametrize("ndim", [3, 4])
@pytest.mark.parametrize("use_seg", [True, False])
class TestUserAddDeleteNode:
    def get_project(self, request, ndim, use_seg):
        params = ProjectParams()
        seg_name = "segmentation_2d" if ndim == 3 else "segmentation_3d"
        if use_seg:
            segmentation = request.getfixturevalue(seg_name)
            axis_names = ["t", "z", "y", "x"] if ndim == 4 else ["t", "y", "x"]
            seg = fp.Array(segmentation, axis_names=axis_names, voxel_size=None)
        else:
            seg = None

        gt_graph = self.get_gt_graph(request, ndim)
        features = FeatureSet(ndim=ndim, seg=use_seg)
        cand_graph = TrackingGraph(NxGraph, gt_graph, features)
        return Project("test", params, segmentation=seg, cand_graph=cand_graph)

    def get_gt_graph(self, request, ndim):
        graph_name = "graph_2d" if ndim == 3 else "graph_3d"
        gt_graph = request.getfixturevalue(graph_name)
        return gt_graph

    def test_user_add_node(self, request, ndim, use_seg):
        project = self.get_project(request, ndim, use_seg)
        features = project.cand_graph.features
        # add a node to replace a skip edge between node 4 in time 2 and node 5 in time 4
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
        graph = project.cand_graph
        assert not graph.has_node(node_id)
        assert graph.has_edge((4, 5))
        action = UserAddNode(project, node_id, attributes, pixels=pixels)
        assert graph.has_node(node_id)
        assert not graph.has_edge((4, 5))
        assert graph.has_edge((4, node_id))
        assert graph.has_edge((node_id, 5))
        assert graph.get_feature_value(node_id, graph.features.position) == position
        assert graph.get_feature_value(node_id, graph.features.track_id) == track_id
        assert graph.get_feature_value(node_id, graph.features.node_selected) is True
        assert graph.get_feature_value(node_id, graph.features.node_selection_pin) is True
        if use_seg:
            assert graph.get_feature_value(node_id, Area()) == 1

        inverse = action.inverse()
        assert not graph.has_node(node_id)
        assert graph.has_edge((4, 5))
        inverse.inverse()
        assert graph.has_node(node_id)
        assert not graph.has_edge((4, 5))
        assert graph.has_edge((4, node_id))
        assert graph.has_edge((node_id, 5))
        assert graph.get_feature_value(node_id, graph.features.position) == position
        assert graph.get_feature_value(node_id, graph.features.track_id) == track_id
        if use_seg:
            assert graph.get_feature_value(node_id, Area()) == 1
        # TODO: error if node already exists?

    def test_user_delete_node(self, request, ndim, use_seg):
        project = self.get_project(request, ndim, use_seg)
        features = project.cand_graph.features
        if ndim == 4 and use_seg:
            for feature in features._features:
                if isinstance(feature, Area):
                    area_feature = feature
                    break
            project.cand_graph.features._features.remove(area_feature)
        # delete node in middle of track. Should skip-connect 3 and 5 with span 3
        node_id = 4

        graph: TrackingGraph = project.cand_graph
        assert graph.has_node(node_id)
        assert graph.has_edge((3, node_id))
        assert graph.has_edge((node_id, 5))
        assert not graph.has_edge((3, 5))

        action = UserDeleteNode(project, node_id)
        assert not graph.has_node(node_id)
        assert not graph.has_edge((3, node_id))
        assert not graph.has_edge((node_id, 5))
        assert graph.has_edge((3, 5))
        assert graph.get_feature_value((3, 5), graph.features.frame_span) == 3

        inverse = action.inverse()
        assert graph.has_node(node_id)
        assert graph.has_edge((3, node_id))
        assert graph.has_edge((node_id, 5))
        assert not graph.has_edge((3, 5))

        inverse.inverse()
        assert not graph.has_node(node_id)
        assert not graph.has_edge((3, node_id))
        assert not graph.has_edge((node_id, 5))
        assert graph.has_edge((3, 5))
        assert graph.get_feature_value((3, 5), graph.features.frame_span) == 3
        # TODO: error if node doesn't exist?
