import funlib.persistence as fp
import pytest

from funtracks import NxGraph, Project, TrackingGraph
from funtracks.actions import SetSelection
from funtracks.features import FeatureSet
from funtracks.params import ProjectParams


@pytest.mark.parametrize("ndim", [3, 4])
@pytest.mark.parametrize("use_seg", [True, False])
class TestSetFeatureValues:
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

    def test_set_node_selected(self, request, ndim, use_seg):
        project = self.get_project(request, ndim, use_seg)
        features = project.cand_graph.features
        node = 1
        graph = project.cand_graph
        assert graph.get_feature_value(node, features.node_selected) is True
        assert project.solution.has_node(node)

        # set selected to False, but don't pin
        action = SetSelection(project, node, value=False)
        assert graph.get_feature_value(node, features.node_selected) is False
        assert not project.solution.has_node(node)
        inverse = action.inverse()
        assert graph.get_feature_value(node, features.node_selected) is True
        assert project.solution.has_node(node)
        inverse.inverse()
        assert graph.get_feature_value(node, features.node_selected) is False
        assert not project.solution.has_node(node)

        # set selected to True and pin
        action = SetSelection(project, node, value=True)
        assert graph.get_feature_value(node, features.node_selected) is True
        assert project.solution.has_node(node)
        inverse = action.inverse()
        assert graph.get_feature_value(node, features.node_selected) is False
        assert not project.solution.has_node(node)
        inverse.inverse()
        assert graph.get_feature_value(node, features.node_selected) is True
        assert project.solution.has_node(node)

    def test_set_edge_selected(self, request, ndim, use_seg):
        project = self.get_project(request, ndim, use_seg)
        features = project.cand_graph.features
        edge = (1, 2)
        graph = project.cand_graph
        assert graph.get_feature_value(edge, features.edge_selected) is True
        assert project.solution.has_edge(edge)

        # set selected to False
        action = SetSelection(project, edge, value=False)
        assert graph.get_feature_value(edge, features.edge_selected) is False
        assert not project.solution.has_edge(edge)
        inverse = action.inverse()
        assert graph.get_feature_value(edge, features.edge_selected) is True
        assert project.solution.has_edge(edge)
        inverse.inverse()
        assert graph.get_feature_value(edge, features.edge_selected) is False
        assert not project.solution.has_edge(edge)

        # set selected to True
        action = SetSelection(project, edge, value=True)
        assert graph.get_feature_value(edge, features.edge_selected) is True
        assert project.solution.has_edge(edge)
        inverse = action.inverse()
        assert graph.get_feature_value(edge, features.edge_selected) is False
        assert not project.solution.has_edge(edge)
        inverse.inverse()
        assert graph.get_feature_value(edge, features.edge_selected) is True
        assert project.solution.has_edge(edge)
