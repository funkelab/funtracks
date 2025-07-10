import funlib.persistence as fp
import pytest

from funtracks import NxGraph, Project, TrackingGraph
from funtracks.actions import SetFeatureValues
from funtracks.features import Feature, FeatureSet, FeatureType
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
        return Project("test", params, segmentation=seg, graph=cand_graph)

    def get_gt_graph(self, request, ndim):
        graph_name = "graph_2d" if ndim == 3 else "graph_3d"
        gt_graph = request.getfixturevalue(graph_name)
        return gt_graph

    def test_set_node_features(self, request, ndim, use_seg):
        # TODO: test edge cases
        project = self.get_project(request, ndim, use_seg)
        features = project.graph.features
        mutable_feature = Feature(
            attr_name="test",
            value_names="Test mutability",
            feature_type=FeatureType.NODE,
            valid_ndim=(3, 4),
            computed=False,
            default_value=0,
            required=False,
        )
        features.add_feature(mutable_feature)
        node = 1
        graph = project.graph
        assert graph.get_feature_value(node, mutable_feature) == 0
        attributes = {
            mutable_feature: 1,
        }
        action = SetFeatureValues(project, node, attributes)
        assert graph.get_feature_value(node, mutable_feature) == 1
        inverse = action.inverse()
        assert graph.get_feature_value(node, mutable_feature) == 0
        inverse.inverse()
        assert graph.get_feature_value(node, mutable_feature) == 1

    def test_set_edge_features(self, request, ndim, use_seg):
        # TODO: test edge cases
        project = self.get_project(request, ndim, use_seg)
        features = project.graph.features
        mutable_feature = Feature(
            attr_name="test",
            value_names="Test mutability",
            feature_type=FeatureType.EDGE,
            valid_ndim=(3, 4),
            computed=False,
            default_value=0,
            required=False,
        )
        features.add_feature(mutable_feature)
        edge = (1, 2)
        graph = project.graph
        assert graph.get_feature_value(edge, mutable_feature) == 0
        attributes = {
            mutable_feature: 1,
        }
        action = SetFeatureValues(project, edge, attributes)
        assert graph.get_feature_value(edge, mutable_feature) == 1
        inverse = action.inverse()
        assert graph.get_feature_value(edge, mutable_feature) == 0
        inverse.inverse()
        assert graph.get_feature_value(edge, mutable_feature) == 1
