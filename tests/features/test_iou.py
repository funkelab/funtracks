import funlib.persistence as fp
import pytest

from funtracks import NxGraph, Project, TrackingGraph
from funtracks.features import FeatureSet
from funtracks.features.edge_features import IoU
from funtracks.params import ProjectParams


@pytest.mark.parametrize("ndim", [3, 4])
class TestIoU:
    def get_project(self, request, ndim):
        params = ProjectParams()
        seg_name = "segmentation_2d" if ndim == 3 else "segmentation_3d"
        segmentation = request.getfixturevalue(seg_name)
        axis_names = ["t", "z", "y", "x"] if ndim == 4 else ["t", "y", "x"]
        seg = fp.Array(segmentation, axis_names=axis_names, voxel_size=None)

        gt_graph = self.get_gt_graph(request, ndim)
        features = FeatureSet(ndim=ndim, seg=seg)
        cand_graph = TrackingGraph(NxGraph, gt_graph, features)
        return Project("test", params, segmentation=seg, graph=cand_graph)

    def get_gt_graph(self, request, ndim):
        graph_name = "graph_2d" if ndim == 3 else "graph_3d"
        gt_graph = request.getfixturevalue(graph_name)
        return gt_graph

    def test_iou_update(self, request, ndim):
        project = self.get_project(request, ndim)
        feat = IoU()
        for edge in project.graph.edges:
            iou = feat.update(project, edge)
            assert iou == pytest.approx(
                project.graph.get_feature_value(edge, feat), abs=0.01
            )
