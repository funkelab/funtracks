import copy

import networkx as nx
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from funtracks.actions import (
    ActionGroup,
    AddNode,
)


@pytest.mark.parametrize("ndim", [3, 4])
@pytest.mark.parametrize("with_seg", [True, False])
def test_add_delete_nodes(get_tracks, ndim, with_seg):
    # Get a tracks instance
    tracks = get_tracks(ndim=ndim, with_seg=with_seg, is_solution=True)
    reference_graph = tracks.graph
    reference_seg = copy.deepcopy(tracks.segmentation)

    # Start with an empty Tracks
    empty_graph = nx.DiGraph()
    empty_seg = np.zeros_like(tracks.segmentation) if with_seg else None
    tracks.graph = empty_graph
    if with_seg:
        tracks.segmentation = empty_seg

    nodes = list(reference_graph.nodes())
    actions = []
    for node in nodes:
        pixels = np.nonzero(reference_seg == node) if with_seg else None
        actions.append(
            AddNode(tracks, node, dict(reference_graph.nodes[node]), pixels=pixels)
        )
    action = ActionGroup(tracks=tracks, actions=actions)

    assert set(tracks.graph.nodes()) == set(reference_graph.nodes())
    for node, data in tracks.graph.nodes(data=True):
        reference_data = reference_graph.nodes[node]
        assert data == reference_data
    if with_seg:
        assert_array_almost_equal(tracks.segmentation, reference_seg)

    # Invert the action to delete all the nodes
    del_nodes = action.inverse()
    assert set(tracks.graph.nodes()) == set(empty_graph.nodes())
    if with_seg:
        assert_array_almost_equal(tracks.segmentation, empty_seg)

    # Re-invert the action to add back all the nodes and their attributes
    del_nodes.inverse()
    assert set(tracks.graph.nodes()) == set(reference_graph.nodes())
    for node, data in tracks.graph.nodes(data=True):
        reference_data = copy.deepcopy(reference_graph.nodes[node])
        assert data == reference_data
    if with_seg:
        assert_array_almost_equal(tracks.segmentation, reference_seg)


def test_add_node_missing_time(get_tracks):
    tracks = get_tracks(ndim=3, with_seg=True, is_solution=True)
    with pytest.raises(ValueError, match="Must provide a time attribute for node"):
        AddNode(tracks, 8, {})


def test_add_node_missing_pos(get_tracks):
    tracks = get_tracks(ndim=3, with_seg=True, is_solution=True)
    # First test: missing track_id raises an error
    with pytest.raises(ValueError, match="Must provide a track_id attribute for node"):
        AddNode(tracks, 8, {"t": 2})

    # Second test: with track_id but without segmentation, missing pos raises an error
    tracks_no_seg = get_tracks(ndim=3, with_seg=False, is_solution=True)
    with pytest.raises(
        ValueError, match="Must provide position or segmentation for node"
    ):
        AddNode(tracks_no_seg, 8, {"t": 2, "track_id": 1})


@pytest.mark.parametrize("ndim", [3, 4])
@pytest.mark.parametrize("with_seg", [True, False])
def test_custom_attributes_preserved(get_tracks, ndim, with_seg):
    """Test custom node attributes preserved through add/delete/re-add cycles."""
    from funtracks.features import Feature

    tracks = get_tracks(ndim=ndim, with_seg=with_seg, is_solution=True)

    # Register custom features so they get saved by DeleteNode
    custom_features = {
        "cell_type": Feature(
            feature_type="node",
            value_type="str",
            num_values=1,
            display_name="Cell Type",
            required=False,
            default_value=None,
        ),
        "confidence": Feature(
            feature_type="node",
            value_type="float",
            num_values=1,
            display_name="Confidence",
            required=False,
            default_value=None,
        ),
        "user_label": Feature(
            feature_type="node",
            value_type="str",
            num_values=1,
            display_name="User Label",
            required=False,
            default_value=None,
        ),
    }
    for key, feature in custom_features.items():
        tracks.features[key] = feature

    # Define attributes including custom ones
    custom_attrs = {
        "t": 2,
        "track_id": 10,
        "pos": [50.0, 50.0] if ndim == 3 else [50.0, 50.0, 50.0],
        # Custom user attributes
        "cell_type": "neuron",
        "confidence": 0.95,
        "user_label": "important_cell",
    }

    # Create segmentation if needed
    if with_seg:
        from conftest import sphere
        from skimage.draw import disk

        if ndim == 3:
            rr, cc = disk(center=(50, 50), radius=5, shape=(100, 100))
            pixels = (np.array([2]), rr, cc)
        else:
            mask = sphere(center=(50, 50, 50), radius=5, shape=(100, 100, 100))
            # Create proper 4D pixel coordinates (t, z, y, x)
            pixels = (np.array([2]), *np.nonzero(mask))
        custom_attrs.pop("pos")  # pos will be computed from segmentation
    else:
        pixels = None

    # Add a node with custom attributes
    node_id = 100
    action = AddNode(tracks, node_id, custom_attrs, pixels=pixels)

    # Verify all attributes are present after adding
    assert tracks.graph.has_node(node_id)
    for key, value in custom_attrs.items():
        assert tracks.graph.nodes[node_id][key] == value, (
            f"Attribute {key} not preserved after add"
        )

    # Delete the node
    delete_action = action.inverse()
    assert not tracks.graph.has_node(node_id)

    # Re-add the node by inverting the delete
    delete_action.inverse()
    assert tracks.graph.has_node(node_id)

    # Verify all custom attributes are still present after re-adding
    for key, value in custom_attrs.items():
        assert tracks.graph.nodes[node_id][key] == value, (
            f"Attribute {key} not preserved after delete/re-add cycle"
        )
