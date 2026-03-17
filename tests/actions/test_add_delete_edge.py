import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from polars.testing import assert_frame_equal

from funtracks.actions import (
    ActionGroup,
    AddEdge,
    DeleteEdge,
)
from funtracks.data_model import SolutionTracks
from funtracks.features import FeatureDict, LineageID, Position, Time, TrackletID
from funtracks.utils.tracksdata_utils import create_empty_graphview_graph

iou_key = "iou"


@pytest.mark.parametrize("ndim", [3, 4])
@pytest.mark.parametrize("with_seg", [True, False])
def test_add_delete_edges(get_tracks, ndim, with_seg):
    tracks = get_tracks(ndim=ndim, with_seg=with_seg, is_solution=True)
    reference_graph = tracks.graph
    reference_seg = np.asarray(tracks.segmentation).copy()

    # Create an empty tracks with just nodes (no edges)
    for edge in tracks.graph.edge_list():
        tracks.graph.remove_edge(*edge)

    edges = [(1, 2), (1, 3), (3, 4), (4, 5)]

    action = ActionGroup(tracks=tracks, actions=[AddEdge(tracks, edge) for edge in edges])

    with pytest.raises(ValueError, match="Edge .* already exists in the graph"):
        AddEdge(tracks, (1, 2))

    # TODO: What if adding an edge that already exists?
    # TODO: test all the edge cases, invalid operations, etc. for all actions
    assert set(tracks.graph.node_ids()) == set(reference_graph.node_ids())
    assert_frame_equal(
        tracks.graph.edge_attrs(),
        reference_graph.edge_attrs(),
        check_row_order=False,
        check_column_order=False,
    )
    if with_seg:
        assert_array_almost_equal(tracks.segmentation, reference_seg)

    inverse = action.inverse()

    assert set(tracks.graph.edge_ids()) == set()
    if tracks.segmentation is not None:
        assert_array_almost_equal(tracks.segmentation, reference_seg)

    re_added = inverse.inverse()
    assert set(tracks.graph.node_ids()) == set(reference_graph.node_ids())
    assert set(tracks.graph.edge_ids()) == set(reference_graph.edge_ids())
    assert sorted(tracks.graph.edge_list()) == sorted(reference_graph.edge_list())
    assert_frame_equal(
        tracks.graph.edge_attrs(),
        reference_graph.edge_attrs(),
        check_row_order=False,
        check_column_order=False,
    )
    if with_seg:
        assert_array_almost_equal(tracks.segmentation, reference_seg)

    # Regression: calling inverse.inverse() a second time must not raise
    # ValueError about 'edge_id'. AddEdge._apply() used to mutate
    # DeleteEdge.attributes via aliasing,corrupting it for subsequent
    # inverse() calls (mirrors ActionHistory calling .inverse()on the same
    # stored action twice: undo → redo → undo). re_added.inverse() resets
    # graph state (edges absent); it uses fresh DeleteEdge objects so it does
    # NOT trigger the bug. The second inverse.inverse() calls the SAME DeleteEdge
    # objects again — that's where the corruption surfaces.
    re_added.inverse()  # reset state: edges absent (fresh objects, no bug here)
    inverse.inverse()  # same DeleteEdge objects called again — must not crash
    assert set(tracks.graph.edge_ids()) == set(reference_graph.edge_ids())


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
            default_value=None,
        ),
        "confidence": Feature(
            feature_type="edge",
            value_type="float",
            num_values=1,
            display_name="Confidence",
            default_value=None,
        ),
        "weight": Feature(
            feature_type="edge",
            value_type="float",
            num_values=1,
            display_name="Weight",
            default_value=None,
        ),
    }
    for key, feature in custom_features.items():
        tracks.add_feature(key, feature)

    # Define custom edge attributes
    custom_attrs = {
        "edge_type": "division",
        "confidence": 0.92,
        "weight": 1.5,
    }

    # Add an edge with custom attributes
    edge = (1, 5)
    action = AddEdge(tracks, edge, attributes=custom_attrs)

    # Verify all attributes are present after adding
    assert tracks.graph.has_edge(*edge)
    for key, value in custom_attrs.items():
        edge_id = tracks.graph.edge_id(*edge)
        assert tracks.graph.edges[edge_id][key] == value, (
            f"Attribute {key} not set correctly on edge"
        )

    # Delete the edge
    delete_action = action.inverse()
    assert not tracks.graph.has_edge(*edge)

    # Re-add the edge by inverting the delete
    delete_action.inverse()
    assert tracks.graph.has_edge(*edge)

    # Verify all custom attributes are still present after re-adding
    for key, value in custom_attrs.items():
        edge_id = tracks.graph.edge_id(*edge)
        assert tracks.graph.edges[edge_id][key] == value, (
            f"Attribute {key} not preserved after delete/re-add cycle"
        )


def test_add_edge_with_unregistered_edge_attr(tmp_path):
    """AddEdge must not crash when the graph has edge attrs absent from tracks.features.

    Reproduces the KeyError that occurs when a pre-built graph (e.g. from the motile
    solver) carries edge attributes such as 'iou' or custom solver scores that were
    written directly to the graph without going through tracks.add_feature().
    The existing tests are blind to this bug because the get_tracks fixture explicitly
    registers 'iou' in the FeatureDict, keeping both registries in sync.
    """
    db_path = str(tmp_path / "test.db")

    # Build a graph with "custom_score" on every edge.
    # This mirrors what the motile solver does: it writes edge attributes directly
    # to the graph without going through tracks.add_feature().
    graph = create_empty_graphview_graph(
        node_attributes=["pos", "track_id", "lineage_id"],
        edge_attributes=["custom_score"],
        database=db_path,
        position_attrs=["pos"],
        ndim=3,
    )

    graph.bulk_add_nodes(
        nodes=[
            {"t": 0, "pos": [10.0, 10.0], "track_id": 1, "lineage_id": 1, "solution": 1},
            {"t": 1, "pos": [11.0, 11.0], "track_id": 2, "lineage_id": 1, "solution": 1},
        ],
        indices=[1, 2],
    )

    # Wrap in SolutionTracks without registering "custom_score" in features —
    # this is the scenario that triggers the bug.
    features = FeatureDict(
        features={
            "t": Time(),
            "pos": Position(axes=["y", "x"]),
            "track_id": TrackletID(),
            "lineage_id": LineageID(),
        },
        time_key="t",
        position_key="pos",
        tracklet_key="track_id",
        lineage_key="lineage_id",
    )
    tracks = SolutionTracks(graph, ndim=3, features=features)

    # Sanity: "custom_score" is in the graph schema but NOT in tracks.features.
    assert "custom_score" in tracks.graph.edge_attr_keys()
    assert "custom_score" not in tracks.features

    # Before the fix this raises: KeyError: 'custom_score'
    AddEdge(tracks, (1, 2))

    assert tracks.graph.has_edge(1, 2)
