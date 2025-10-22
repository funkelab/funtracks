import copy

import networkx as nx
import pytest
from numpy.testing import assert_array_almost_equal

from funtracks.actions import (
    ActionGroup,
    AddEdge,
    DeleteEdge,
)
from funtracks.data_model.graph_attributes import EdgeAttr


@pytest.mark.parametrize("ndim", [3, 4])
@pytest.mark.parametrize("with_seg", [True, False])
def test_add_delete_edges(get_tracks, ndim, with_seg):
    tracks = get_tracks(ndim=ndim, with_seg=with_seg, is_solution=True)
    graph_with_edges = tracks.graph
    orig_segmentation = copy.deepcopy(tracks.segmentation)

    # Create an empty tracks with just nodes (no edges)
    node_graph = nx.create_empty_copy(tracks.graph, with_data=True)
    tracks.graph = node_graph

    edges = [[1, 2], [1, 3], [3, 4], [4, 5]]

    action = ActionGroup(tracks=tracks, actions=[AddEdge(tracks, edge) for edge in edges])
    # TODO: What if adding an edge that already exists?
    # TODO: test all the edge cases, invalid operations, etc. for all actions
    assert set(tracks.graph.nodes()) == set(graph_with_edges.nodes())
    if with_seg:
        for edge in tracks.graph.edges():
            assert tracks.graph.edges[edge][EdgeAttr.IOU.value] == pytest.approx(
                graph_with_edges.edges[edge][EdgeAttr.IOU.value], abs=0.01
            )
        assert_array_almost_equal(tracks.segmentation, orig_segmentation)

    inverse = action.inverse()
    assert set(tracks.graph.edges()) == set()
    if tracks.segmentation is not None:
        assert_array_almost_equal(tracks.segmentation, orig_segmentation)

    inverse.inverse()
    assert set(tracks.graph.nodes()) == set(graph_with_edges.nodes())
    assert set(tracks.graph.edges()) == set(graph_with_edges.edges())
    if with_seg:
        for edge in tracks.graph.edges():
            assert tracks.graph.edges[edge][EdgeAttr.IOU.value] == pytest.approx(
                graph_with_edges.edges[edge][EdgeAttr.IOU.value], abs=0.01
            )
        assert_array_almost_equal(tracks.segmentation, orig_segmentation)


def test_add_edge_missing_endpoint(get_tracks):
    tracks = get_tracks(ndim=3, with_seg=True, is_solution=True)
    with pytest.raises(ValueError, match="Cannot add edge .*: endpoint .* not in graph"):
        AddEdge(tracks, (10, 11))


def test_delete_missing_edge(get_tracks):
    tracks = get_tracks(ndim=3, with_seg=True, is_solution=True)
    with pytest.raises(
        ValueError, match="Edge .* not in the graph, and cannot be removed"
    ):
        DeleteEdge(tracks, (10, 11))
