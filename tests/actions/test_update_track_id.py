import funlib.persistence as fp
import pytest

from funtracks import NxGraph, Project, TrackingGraph
from funtracks.actions import UpdateTrackID
from funtracks.features import FeatureSet
from funtracks.params import ProjectParams


@pytest.mark.parametrize(
    "ndim",
    [
        3,
    ],
)
class TestUpdateTrackID:
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

    def test_update_track_id(self, request, ndim):
        project = self.get_project(request, ndim)
        graph = project.graph
        node_id = 4
        nodes_in_track = [4, 5]
        new_track_id = 4

        assert graph.has_node(node_id)
        orig_track_id = graph.get_track_id(node_id)

        action = UpdateTrackID(project, node_id, track_id=new_track_id)
        for node in nodes_in_track:
            assert graph.get_track_id(node) == new_track_id

        inverse = action.inverse()
        for node in nodes_in_track:
            assert graph.get_track_id(node) == orig_track_id

        inverse.inverse()
        for node in nodes_in_track:
            assert graph.get_track_id(node) == new_track_id
