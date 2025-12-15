import copy

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from polars.testing import assert_frame_equal

from funtracks.actions import (
    ActionGroup,
    AddNode,
)
from funtracks.utils.tracksdata_utils import (
    assert_node_attrs_equal_with_masks,
    create_empty_graphview_graph,
)


@pytest.mark.parametrize("ndim", [3, 4])
@pytest.mark.parametrize("with_seg", [True, False])
def test_add_delete_nodes(get_tracks, ndim, with_seg):
    # Get a tracks instance
    tracks = get_tracks(ndim=ndim, with_seg=with_seg, is_solution=True)
    reference_graph = tracks.graph
    reference_seg = copy.deepcopy(tracks.segmentation)

    # Start with an empty Tracks
    empty_graph = create_empty_graphview_graph(
        with_pos=True, with_track_id=True, with_area=with_seg, with_iou=with_seg
    )
    empty_seg = np.zeros_like(tracks.segmentation) if with_seg else None
    tracks.graph = empty_graph
    if with_seg:
        tracks.segmentation = empty_seg

    # add all the nodes from graph_2d/seg_2d
    nodes = list(reference_graph.node_ids())

    actions = []
    for node in nodes:
        pixels = np.nonzero(reference_seg == node) if with_seg else None

        attrs = {}
        attrs[tracks.features.time_key] = reference_graph[node][tracks.features.time_key]
        if tracks.features.position_key == "pos":
            attrs[tracks.features.position_key] = reference_graph[node][
                tracks.features.position_key
            ].to_list()
        else:
            attrs[tracks.features.position_key] = reference_graph[node][
                tracks.features.position_key
            ]
        attrs[tracks.features.tracklet_key] = reference_graph[node][
            tracks.features.tracklet_key
        ]

        actions.append(AddNode(tracks, node, attributes=attrs, pixels=pixels))
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
    assert node_id in tracks.graph.node_ids()
    for key, value in custom_attrs.items():
        if key == "pos":
            assert_array_almost_equal(tracks.graph[node_id][key], np.array(value))
        else:
            assert tracks.graph[node_id][key] == value, (
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
            assert_array_almost_equal(tracks.graph[node_id][key], np.array(value))
        else:
            assert tracks.graph[node_id][key] == value, (
                f"Attribute {key} not preserved after delete/re-add cycle"
            )
