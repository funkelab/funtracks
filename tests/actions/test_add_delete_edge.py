import networkx as nx
import pytest
from numpy.testing import assert_array_almost_equal

from funtracks.actions import (
    ActionGroup,
    AddEdge,
    DeleteEdge,
)
from funtracks.data_model import SolutionTracks
from funtracks.data_model.graph_attributes import EdgeAttr


def test_add_delete_edges(graph_2d, segmentation_2d):
    node_graph = nx.create_empty_copy(graph_2d, with_data=True)
    tracks = SolutionTracks(node_graph, segmentation_2d)

    edges = [[1, 2], [1, 3], [3, 4], [4, 5]]

    action = ActionGroup(tracks=tracks, actions=[AddEdge(tracks, edge) for edge in edges])
    # TODO: What if adding an edge that already exists?
    # TODO: test all the edge cases, invalid operations, etc. for all actions
    assert set(tracks.graph.nodes()) == set(graph_2d.nodes())
    for edge in tracks.graph.edges():
        assert tracks.graph.edges[edge][EdgeAttr.IOU.value] == pytest.approx(
            graph_2d.edges[edge][EdgeAttr.IOU.value], abs=0.01
        )
    assert_array_almost_equal(tracks.segmentation, segmentation_2d)

    inverse = action.inverse()
    assert set(tracks.graph.edges()) == set()
    assert_array_almost_equal(tracks.segmentation, segmentation_2d)

    inverse.inverse()
    assert set(tracks.graph.nodes()) == set(graph_2d.nodes())
    assert set(tracks.graph.edges()) == set(graph_2d.edges())
    for edge in tracks.graph.edges():
        assert tracks.graph.edges[edge][EdgeAttr.IOU.value] == pytest.approx(
            graph_2d.edges[edge][EdgeAttr.IOU.value], abs=0.01
        )
    assert_array_almost_equal(tracks.segmentation, segmentation_2d)


def test_add_edge_missing_endpoint(graph_2d):
    tracks = SolutionTracks(graph_2d, ndim=3)
    with pytest.raises(ValueError, match="Cannot add edge .*: endpoint .* not in graph"):
        AddEdge(tracks, (10, 11))


def test_delete_missing_edge(graph_2d):
    tracks = SolutionTracks(graph_2d, ndim=3)
    with pytest.raises(
        ValueError, match="Edge .* not in the graph, and cannot be removed"
    ):
        DeleteEdge(tracks, (10, 11))
