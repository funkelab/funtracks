import pytest

from funtracks.features.node_features import Time
from funtracks.nx_graph import NxGraph


class TestNxGraph:
    def test_basic_ops(self, graph_2d):
        graph = NxGraph(graph_2d)
        # basic functionality
        assert graph.has_node(1)
        assert graph.has_node(3)
        assert not graph.has_node(10)
        assert graph.has_edge((1, 2))
        assert graph.has_edge((1, 3))
        assert not graph.has_edge((1, 6))

        assert graph.successors(1) == [2, 3]
        assert graph.in_degree(1) == 0
        assert graph.in_degree(2) == 1
        assert graph.out_degree(1) == 2
        assert graph.out_degree(3) == 1
        assert graph.get_feature_value(3, Time()) == 1

    def test_subgraph(self, graph_2d):
        graph = NxGraph(graph_2d)
        # TODO: invalid nodes and edges
        nodes = [1, 2]
        edges = [(1, 2)]
        graph_view = graph.subgraph(nodes, edges)
        assert graph_view.has_node(1)
        assert not graph_view.has_node(3)
        assert graph_view.has_edge((1, 2))
        assert not graph_view.has_edge((2, 3))

        assert graph_view.successors(1) == [2]
        assert graph_view.in_degree(1) == 0
        assert graph_view.in_degree(2) == 1
        assert graph_view.out_degree(1) == 1
        assert graph_view.out_degree(2) == 0
        with pytest.raises(KeyError):
            graph_view.get_feature_value(3, Time())
