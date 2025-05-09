from collections import Counter

import networkx as nx
import pytest

from funtracks import NxGraph
from funtracks.cand_graph import CandGraph
from funtracks.features import FeatureSet
from funtracks.params.cand_graph_params import CandGraphParams


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
        params = CandGraphParams()
        cand_graph = CandGraph(NxGraph, gt_graph, features, params)
        return cand_graph

    def test_add_cand_edges(self, request, ndim):
        graph = self.get_tracking_graph(request, ndim)
        graph._graph = nx.create_empty_copy(graph._graph, with_data=True)
        graph.params.max_move_distance = 200  # fully connected
        graph.initialize_cand_edges()
        if ndim == 3:
            expected_edges = [(1, 2), (1, 3), (2, 4), (3, 4)]
        else:
            expected_edges = [(1, 2), (1, 3)]
        assert Counter(expected_edges) == Counter(list(graph._graph.edges()))

        # update smaller
        graph.update_max_move_distance(30)
        expected_edges = [(1, 3)]
        assert Counter(expected_edges) == Counter(list(graph._graph.edges()))

        # update bigger
        graph.update_max_move_distance(50)
        expected_edges = [(1, 2), (1, 3)]
        assert Counter(expected_edges) == Counter(list(graph._graph.edges()))

        # pin an edge and then update to nothing
        edge = (1, 2)
        graph.set_feature_value(edge, graph.features.edge_pin, True)
        graph.update_max_move_distance(0)
        expected_edges = [edge]
        assert Counter(expected_edges) == Counter(list(graph._graph.edges()))

        # add skip edges
        graph.update_max_move_distance(200)  # fully conntected
        graph.update_frame_span(2)
        if ndim == 3:
            expected_edges = [(1, 2), (1, 3), (2, 4), (1, 4), (3, 4), (4, 5), (4, 6)]
        else:
            expected_edges = [(1, 2), (1, 3), (2, 4), (3, 4)]
        assert Counter(expected_edges) == Counter(list(graph._graph.edges()))

        graph.update_frame_span(1)
        if ndim == 3:
            expected_edges = [(1, 2), (1, 3), (2, 4), (3, 4)]
        else:
            expected_edges = [(1, 2), (1, 3)]
        assert Counter(expected_edges) == Counter(list(graph._graph.edges()))
