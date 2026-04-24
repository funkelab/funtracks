import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from polars.testing import assert_frame_equal
from tracksdata.array import GraphArrayView

from funtracks.actions import (
    ActionGroup,
    AddNode,
    DeleteNode,
)
from funtracks.utils.tracksdata_utils import (
    assert_node_attrs_equal_with_masks,
    create_empty_graphview_graph,
)

from ..conftest import make_2d_disk_mask, make_3d_sphere_mask


@pytest.mark.parametrize("ndim", [3, 4])
@pytest.mark.parametrize("with_seg", [True, False])
def test_add_delete_nodes(get_tracks, ndim, with_seg):
    # Get a tracks instance
    tracks = get_tracks(ndim=ndim, with_seg=with_seg, is_solution=True)
    reference_graph = tracks.graph
    reference_seg = np.asarray(tracks.segmentation).copy() if with_seg else None

    # Start with an empty Tracks
    node_attributes = [
        tracks.features.time_key,
        tracks.features.tracklet_key,
        tracks.features.lineage_key,
        tracks.features.position_key,
    ]
    edge_attributes = ["iou"] if with_seg else []
    empty_graph = create_empty_graphview_graph(
        node_attributes=node_attributes + (["area", "bbox", "mask"] if with_seg else []),
        edge_attributes=edge_attributes,
        ndim=ndim,
    )
    empty_seg = np.zeros_like(tracks.segmentation) if with_seg else None
    tracks.graph = empty_graph
    segmentation_shape = (5, 100, 100) if ndim == 3 else (5, 100, 100, 100)
    tracks.segmentation = (
        GraphArrayView(
            graph=tracks.graph, shape=segmentation_shape, attr_key="node_id", offset=0
        )
        if with_seg
        else None
    )

    # add all the nodes from graph_2d/seg_2d
    nodes = list(reference_graph.node_ids())

    actions = []
    for node in nodes:
        mask = reference_graph.nodes[node]["mask"] if with_seg else None

        attrs = {}
        attrs[tracks.features.time_key] = reference_graph.nodes[node][
            tracks.features.time_key
        ]
        if tracks.features.position_key == "pos":
            attrs[tracks.features.position_key] = reference_graph.nodes[node][
                tracks.features.position_key
            ].to_list()
        else:
            attrs[tracks.features.position_key] = reference_graph.nodes[node][
                tracks.features.position_key
            ]
        attrs[tracks.features.tracklet_key] = reference_graph.nodes[node][
            tracks.features.tracklet_key
        ]
        attrs[tracks.features.lineage_key] = reference_graph.nodes[node][
            tracks.features.lineage_key
        ]
        if with_seg:
            attrs["bbox"] = reference_graph.nodes[node]["bbox"]
            attrs["mask"] = reference_graph.nodes[node]["mask"]

        actions.append(AddNode(tracks, node, attributes=attrs, mask=mask))
    action = ActionGroup(tracks=tracks, actions=actions)

    assert set(tracks.graph.node_ids()) == set(reference_graph.node_ids())
    data_tracks = tracks.graph.node_attrs()
    data_reference = reference_graph.node_attrs()
    if with_seg:
        assert_array_almost_equal(tracks.segmentation, reference_seg)
        assert_node_attrs_equal_with_masks(data_tracks, data_reference)
    else:
        assert_frame_equal(
            data_reference,  # .drop(["mask", "bbox", "area"]),
            data_tracks,  # .drop(["mask", "bbox", "area"]),
            check_column_order=False,
            check_row_order=False,
            check_dtypes=False,
        )

    # Invert the action to delete all the nodes
    del_nodes = action.inverse()
    assert set(tracks.graph.node_ids()) == set(empty_graph.node_ids())
    if with_seg:
        assert_array_almost_equal(tracks.segmentation, empty_seg)

    # Re-invert the action to add back all the nodes and their attributes
    del_nodes.inverse()
    assert set(tracks.graph.node_ids()) == set(reference_graph.node_ids())
    data_tracks = tracks.graph.node_attrs()
    data_reference = reference_graph.node_attrs()
    if with_seg:
        assert_array_almost_equal(tracks.segmentation, reference_seg)
        assert_node_attrs_equal_with_masks(data_tracks, data_reference)
    else:
        assert_frame_equal(
            data_reference,  # .drop(["mask", "bbox", "area"]),
            data_tracks,  # .drop(["mask", "bbox", "area"]),
            check_column_order=False,
            check_row_order=False,
            check_dtypes=False,
        )


@pytest.mark.parametrize("ndim", [3, 4])
def test_add_node_invalidates_cache(get_tracks, ndim):
    """Test that AddNode invalidates the GraphArrayView cache for the new node's region.

    Regression test: when the cache is pre-populated before a new node is added,
    reading the segmentation afterwards must reflect the new node (not stale zeros).
    """
    tracks = get_tracks(ndim=ndim, with_seg=True, is_solution=True)

    # Pick a new node id and time point that doesn't already exist
    new_node = max(tracks.graph.node_ids()) + 1
    time = 3
    if ndim == 3:
        mask = make_2d_disk_mask(center=(15, 45), radius=5)
    else:
        mask = make_3d_sphere_mask(center=(15, 45, 75), radius=5)

    # Pre-populate the cache by reading the segmentation at the target time point
    _ = np.asarray(tracks.segmentation[time])
    assert time in tracks.segmentation._cache._store

    # Add a new node with segmentation
    attrs = {
        tracks.features.time_key: time,
        tracks.features.tracklet_key: new_node,
        tracks.features.lineage_key: new_node,
    }
    AddNode(tracks, new_node, attributes=attrs, mask=mask)

    # Reading through the cache must reflect the new node (not return stale zeros)
    center_pixel = (15, 45) if ndim == 3 else (15, 45, 75)
    assert np.asarray(tracks.segmentation[time])[center_pixel] == new_node, (
        "New node label not found in cached slice after AddNode"
    )


@pytest.mark.parametrize("ndim", [3, 4])
def test_delete_node_invalidates_cache(get_tracks, ndim):
    """Test that DeleteNode invalidates the GraphArrayView cache for the deleted node.

    Regression test: when the cache is pre-populated before a node is deleted,
    reading the segmentation afterwards must return zeros (not the stale label value).
    """
    tracks = get_tracks(ndim=ndim, with_seg=True, is_solution=True)
    node = 1
    time = tracks.get_time(node)
    # Node 1 center: (50, 50) for ndim=3, (50, 50, 50) for ndim=4
    center_pixel = (50, 50) if ndim == 3 else (50, 50, 50)

    # Pre-populate the cache by reading the time slice
    _ = np.asarray(tracks.segmentation[time])
    assert time in tracks.segmentation._cache._store

    # Confirm node's label is visible at its center pixel via the cached slice
    assert np.asarray(tracks.segmentation[time])[center_pixel] == node

    DeleteNode(tracks, node)

    # Reading through the cache must reflect the deletion (not return the stale label)
    assert np.asarray(tracks.segmentation[time])[center_pixel] == 0, (
        "Deleted node label still found in cached slice after DeleteNode"
    )


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
            default_value=None,
        ),
        "confidence": Feature(
            feature_type="node",
            value_type="float",
            num_values=1,
            display_name="Confidence",
            default_value=None,
        ),
        "user_label": Feature(
            feature_type="node",
            value_type="str",
            num_values=1,
            display_name="User Label",
            default_value=None,
        ),
    }
    for key, feature in custom_features.items():
        tracks.add_feature(key, feature)

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
        if ndim == 3:
            # Create 2D mask centered at (50, 50) with radius 5
            mask = make_2d_disk_mask(center=(50, 50), radius=5)
        else:
            # Create proper 4D pixel coordinates (t, z, y, x)
            mask = make_3d_sphere_mask(center=(50, 50, 50), radius=5)
        custom_attrs["mask"] = mask
        custom_attrs["bbox"] = mask.bbox
        custom_attrs.pop("pos")  # pos will be computed from segmentation
    else:
        mask = None

    # Add a node with custom attributes
    node_id = 100
    action = AddNode(tracks, node_id, custom_attrs.copy(), mask=mask)
    # Verify all attributes are present after adding
    assert tracks.graph.has_node(node_id)
    for key, value in custom_attrs.items():
        if key == "pos":
            assert_array_almost_equal(tracks.graph.nodes[node_id][key], np.array(value))
        elif key == "mask":
            continue
        elif key == "bbox":
            assert_array_equal(np.asarray(tracks.graph.nodes[node_id][key]), value)
        else:
            assert tracks.graph.nodes[node_id][key] == value, (
                f"Attribute {key} not preserved after add"
            )

    # Delete the node
    delete_action = action.inverse()
    assert node_id not in tracks.graph.node_ids()

    # Re-add the node by inverting the delete
    delete_action.inverse()
    assert node_id in tracks.graph.node_ids()

    # Verify all custom attributes are still present after re-adding
    for key, value in custom_attrs.items():
        if key == "pos":
            assert_array_almost_equal(tracks.graph.nodes[node_id][key], np.array(value))
        elif key == "mask":
            continue
        elif key == "bbox":
            assert_array_equal(np.asarray(tracks.graph.nodes[node_id][key]), value)
        else:
            assert tracks.graph.nodes[node_id][key] == value, (
                f"Attribute {key} not preserved after delete/re-add cycle"
            )
