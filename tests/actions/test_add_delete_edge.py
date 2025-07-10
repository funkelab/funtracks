import funlib.persistence as fp
import pytest

from funtracks import NxGraph, Project, TrackingGraph
from funtracks.actions import AddEdge
from funtracks.features import FeatureSet
from funtracks.features.edge_features import IoU
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
class TestAddDeleteEdge:
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
            cand_graph = TrackingGraph(NxGraph, gt_graph, features)
        else:
            cand_graph = None
        return Project("test", params, segmentation=seg, graph=cand_graph)

    def get_gt_graph(self, request, ndim):
        graph_name = "graph_2d" if ndim == 3 else "graph_3d"
        gt_graph = request.getfixturevalue(graph_name)
        return gt_graph

    def test_add_edge(self, request, ndim, use_seg, use_graph):
        project = self.get_project(request, ndim, use_seg, use_graph)
        features = project.graph.features
        edge_id = (4, 6)
        graph = project.graph
        assert not graph.has_edge(edge_id)
        attributes = {
            features.edge_selected: True,
            features.edge_selection_pin: True,
        }
        action = AddEdge(project, edge_id, attributes)
        assert graph.has_edge(edge_id)
        assert graph.get_feature_value(edge_id, graph.features.edge_selected)
        assert graph.get_feature_value(edge_id, graph.features.edge_selection_pin)
        if use_seg:
            assert graph.get_feature_value(edge_id, IoU()) == 0.0

        inverse = action.inverse()
        assert not graph.has_edge(edge_id)
        inverse.inverse()
        assert graph.has_edge(edge_id)
        assert graph.get_feature_value(edge_id, graph.features.edge_selected)
        assert graph.get_feature_value(edge_id, graph.features.edge_selection_pin)
        if use_seg:
            iou_feat = None
            for feature in features.edge_features:
                if isinstance(feature, IoU):
                    iou_feat = feature
            assert graph.get_feature_value(edge_id, iou_feat) == 0.0
        # TODO: error if edge already exists?

    # def test_delete_edge(self, request, ndim, use_seg, use_graph):
    #     project = self.get_project(request, ndim, use_seg, use_graph)
    #     if not use_graph:
    #         # there are no edges to delete...
    #         # project.cand_graph.initialize_cand_edges()
    #         project.cand_graph._graph.add_edge(1, 2)
    #     edge_id = (1, 2)
    #     graph = project.cand_graph
    #     assert graph.has_edge(edge_id)
    #     action = DeleteEdge(project, edge_id)
    #     assert not graph.has_edge(edge_id)
    #     inverse = action.inverse()
    #     assert graph.has_edge(edge_id)
    #     inverse.inverse()
    #     assert not graph.has_edge(edge_id)
    #     # TODO: error if edge doesn't exist?
