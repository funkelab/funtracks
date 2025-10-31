import copy

import networkx as nx
import pytest
from numpy.testing import assert_array_almost_equal

from funtracks.actions import (
    ActionGroup,
    AddEdge,
    DeleteEdge,
)

iou_key = "iou"


@pytest.mark.parametrize("ndim", [3, 4])
@pytest.mark.parametrize("with_seg", [True, False])
def test_add_delete_edges(get_tracks, ndim, with_seg):
    tracks = get_tracks(ndim=ndim, with_seg=with_seg, is_solution=True)
    reference_graph = copy.deepcopy(tracks.graph)
    reference_seg = copy.deepcopy(tracks.segmentation)

    # Create an empty tracks with just nodes (no edges)
    node_graph = nx.create_empty_copy(tracks.graph, with_data=True)
    tracks.graph = node_graph

    edges = [(1, 2), (1, 3), (3, 4), (4, 5)]

    action = ActionGroup(tracks=tracks, actions=[AddEdge(tracks, edge) for edge in edges])
    # TODO: What if adding an edge that already exists?
    # TODO: test all the edge cases, invalid operations, etc. for all actions
    assert set(tracks.graph.nodes()) == set(reference_graph.nodes())
    if with_seg:
        for edge in tracks.graph.edges():
            assert tracks.graph.edges[edge][iou_key] == pytest.approx(
                reference_graph.edges[edge][iou_key], abs=0.01
            )
        assert_array_almost_equal(tracks.segmentation, reference_seg)

    inverse = action.inverse()
    assert set(tracks.graph.edges()) == set()
    if tracks.segmentation is not None:
        assert_array_almost_equal(tracks.segmentation, reference_seg)

    inverse.inverse()
    assert set(tracks.graph.nodes()) == set(reference_graph.nodes())
    assert set(tracks.graph.edges()) == set(reference_graph.edges())
    if with_seg:
        for edge in tracks.graph.edges():
            assert tracks.graph.edges[edge][iou_key] == pytest.approx(
                reference_graph.edges[edge][iou_key], abs=0.01
            )
        assert_array_almost_equal(tracks.segmentation, reference_seg)


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


@pytest.mark.parametrize("ndim", [3, 4])
@pytest.mark.parametrize("with_seg", [True, False])
def test_custom_edge_attributes_preserved(get_tracks, ndim, with_seg):
    """Test custom edge attributes preserved through add/delete/re-add cycles."""
    from funtracks.features import Feature

    tracks = get_tracks(ndim=ndim, with_seg=with_seg, is_solution=True)

    # Register custom edge features so they get saved by DeleteEdge
    custom_features = {
        "edge_type": Feature(
            feature_type="edge",
            value_type="str",
            num_values=1,
            display_name="Edge Type",
            required=False,
            default_value=None,
        ),
        "confidence": Feature(
            feature_type="edge",
            value_type="float",
            num_values=1,
            display_name="Confidence",
            required=False,
            default_value=None,
        ),
        "weight": Feature(
            feature_type="edge",
            value_type="float",
            num_values=1,
            display_name="Weight",
            required=False,
            default_value=None,
        ),
    }
    for key, feature in custom_features.items():
        tracks.features[key] = feature

    # Define custom edge attributes
    custom_attrs = {
        "edge_type": "division",
        "confidence": 0.92,
        "weight": 1.5,
    }

    # Add an edge with custom attributes
    edge = (1, 2)
    action = AddEdge(tracks, edge, attributes=custom_attrs)

    # Verify all attributes are present after adding
    assert tracks.graph.has_edge(*edge)
    for key, value in custom_attrs.items():
        assert tracks.graph.edges[edge][key] == value, (
            f"Attribute {key} not preserved after add"
        )

    # Delete the edge
    delete_action = action.inverse()
    assert not tracks.graph.has_edge(*edge)

    # Re-add the edge by inverting the delete
    delete_action.inverse()
    assert tracks.graph.has_edge(*edge)

    # Verify all custom attributes are still present after re-adding
    for key, value in custom_attrs.items():
        assert tracks.graph.edges[edge][key] == value, (
            f"Attribute {key} not preserved after delete/re-add cycle"
        )
