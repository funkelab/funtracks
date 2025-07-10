import funlib.persistence as fp
import pytest

from funtracks import NxGraph, Project, TrackingGraph
from funtracks.features import FeatureSet
from funtracks.params import ProjectParams
from funtracks.user_actions import UserDeleteEdge, UserSelectEdge


@pytest.mark.parametrize("ndim", [3, 4])
@pytest.mark.parametrize("use_seg", [True, False])
class TestUserAddDeleteEdge:
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
        return Project("test", params, segmentation=seg, graph=cand_graph)

    def get_gt_graph(self, request, ndim):
        graph_name = "graph_2d" if ndim == 3 else "graph_3d"
        gt_graph = request.getfixturevalue(graph_name)
        return gt_graph

    def test_user_add_edge(self, request, ndim, use_seg):
        project = self.get_project(request, ndim, use_seg)
        # add an edge from 4 to 6 (will make 4 a division and 5 will need to relabel track id)
        edge = (4, 6)
        attributes = {}
        graph = project.graph
        old_child = 5
        old_track_id = graph.get_feature_value(old_child, graph.features.track_id)
        assert not graph.has_edge(edge)
        assert not project.solution.has_edge(edge)
        for node in edge:
            assert (
                graph.get_feature_value(node, graph.features.node_selection_pin) is None
            )

        action = UserSelectEdge(project, edge, attributes)
        assert graph.has_edge(edge)
        assert project.solution.has_edge(edge)
        assert graph.get_feature_value(edge, graph.features.edge_selection_pin) is True
        assert graph.get_feature_value(edge, graph.features.edge_selected) is True
        assert graph.get_track_id(old_child) != old_track_id
        for node in edge:
            assert (
                graph.get_feature_value(node, graph.features.node_selection_pin) is True
            )

        inverse = action.inverse()
        assert not graph.has_edge(edge)
        assert not project.solution.has_edge(edge)
        assert graph.get_track_id(old_child) == old_track_id
        for node in edge:
            assert (
                graph.get_feature_value(node, graph.features.node_selection_pin) is None
            )

        inverse.inverse()
        assert graph.has_edge(edge)
        assert project.solution.has_edge(edge)
        assert graph.get_feature_value(edge, graph.features.edge_selection_pin) is True
        assert graph.get_feature_value(edge, graph.features.edge_selected) is True
        assert graph.get_track_id(old_child) != old_track_id
        for node in edge:
            assert (
                graph.get_feature_value(node, graph.features.node_selection_pin) is True
            )

    def test_user_delete_edge(self, request, ndim, use_seg):
        project = self.get_project(request, ndim, use_seg)
        # delete edge (1, 3). (1,2) is now not a division anymore
        edge = (1, 3)
        old_child = 2

        graph: TrackingGraph = project.graph
        old_track_id = graph.get_track_id(old_child)
        new_track_id = graph.get_track_id(1)
        assert graph.has_edge(edge)
        assert project.solution.has_edge(edge)

        action = UserDeleteEdge(project, edge)
        assert not graph.has_edge(edge)
        assert not project.solution.has_edge(edge)
        assert graph.get_track_id(old_child) == new_track_id

        inverse = action.inverse()
        assert graph.has_edge(edge)
        assert project.solution.has_edge(edge)
        assert graph.get_track_id(old_child) == old_track_id

        double_inv = inverse.inverse()
        assert not graph.has_edge(edge)
        assert not project.solution.has_edge(edge)
        assert graph.get_track_id(old_child) == new_track_id
        # TODO: error if edge doesn't exist?
        double_inv.inverse()

        # delete edge (3, 4). 4 and 5 should get new track id
        edge = (3, 4)
        old_child = 5

        graph: TrackingGraph = project.graph
        old_track_id = graph.get_track_id(old_child)
        assert graph.has_edge(edge)
        assert project.solution.has_edge(edge)

        action = UserDeleteEdge(project, edge)
        assert not graph.has_edge(edge)
        assert not project.solution.has_edge(edge)
        assert graph.get_track_id(old_child) != old_track_id

        inverse = action.inverse()
        assert graph.has_edge(edge)
        assert project.solution.has_edge(edge)
        assert graph.get_track_id(old_child) == old_track_id

        inverse.inverse()
        assert not graph.has_edge(edge)
        assert not project.solution.has_edge(edge)
        assert graph.get_track_id(old_child) != old_track_id
