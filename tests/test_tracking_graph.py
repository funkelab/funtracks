import pytest

from funtracks import NxGraph
from funtracks.features import FeatureSet
from funtracks.tracking_graph import TrackingGraph


@pytest.mark.parametrize("ndim", [3, 4])
# TODO: spatial graph
class TestTrackingGraph:
    def get_gt_graph(self, request, ndim):
        graph_name = "graph_2d" if ndim == 3 else "graph_3d"
        gt_graph = request.getfixturevalue(graph_name)
        return gt_graph

    def get_tracking_graph(self, request, ndim):
        gt_graph = self.get_gt_graph(request, ndim)
        features = FeatureSet(ndim=ndim, seg=True)
        graph = TrackingGraph(NxGraph, gt_graph, features)
        return graph

    def test_get_features(self, request, ndim):
        gt_graph = self.get_gt_graph(request, ndim)
        graph = self.get_tracking_graph(request, ndim)
        assert graph.get_time(1) == 0
        assert graph.get_times([1, 2]) == [0, 1]
        with pytest.raises(KeyError):
            graph.get_time(0)

        edge = (1, 2)
        assert graph.get_distance(edge) == pytest.approx(
            gt_graph.edges[edge]["distance"], abs=0.01
        )

    def test_get_track_neighbors(self, request, ndim):
        if ndim == 4:
            return  # TODO skip properly
        graph = self.get_tracking_graph(request, ndim=ndim)
        pred, succ = graph.get_track_neighbors(track_id=3, time=3)
        assert pred == 4
        assert succ == 5
