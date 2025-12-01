import networkx as nx
import numpy as np
import pytest

from funtracks.actions import AddNode
from funtracks.data_model import SolutionTracks, Tracks

track_attrs = {"time_attr": "t", "tracklet_attr": "track_id"}


def test_recompute_track_ids(graph_2d_with_position):
    tracks = SolutionTracks(
        graph_2d_with_position,
        ndim=3,
        **track_attrs,
    )
    assert tracks.get_next_track_id() == 5


def test_next_track_id(graph_2d_with_computed_features):
    tracks = SolutionTracks(graph_2d_with_computed_features, ndim=3, **track_attrs)
    assert tracks.get_next_track_id() == 6
    AddNode(
        tracks,
        node=10,
        attributes={"t": 3, "pos": [0, 0, 0, 0], "track_id": 10},
    )
    assert tracks.get_next_track_id() == 11


def test_node_id_to_track_id(graph_2d_with_computed_features):
    tracks = SolutionTracks(graph_2d_with_computed_features, ndim=3, **track_attrs)
    with pytest.warns(
        DeprecationWarning,
        match="node_id_to_track_id property will be removed in funtracks v2. ",
    ):
        tracks.node_id_to_track_id  # noqa B018


def test_from_tracks_cls(graph_2d_with_computed_features):
    tracks = Tracks(
        graph_2d_with_computed_features,
        ndim=3,
        pos_attr="POSITION",
        time_attr="TIME",
        tracklet_attr=track_attrs["tracklet_attr"],
        scale=(2, 2, 2),
    )
    solution_tracks = SolutionTracks.from_tracks(tracks)
    assert solution_tracks.graph == tracks.graph
    assert solution_tracks.segmentation == tracks.segmentation
    assert solution_tracks.features.time_key == tracks.features.time_key
    assert solution_tracks.features.position_key == tracks.features.position_key
    assert solution_tracks.scale == tracks.scale
    assert solution_tracks.ndim == tracks.ndim
    assert solution_tracks.get_node_attr(6, tracks.features.tracklet_key) == 5


def test_from_tracks_cls_recompute(graph_2d_with_computed_features):
    tracks = Tracks(
        graph_2d_with_computed_features,
        ndim=3,
        pos_attr="POSITION",
        time_attr="TIME",
        tracklet_attr=track_attrs["tracklet_attr"],
        scale=(2, 2, 2),
    )
    # delete track id on one node triggers reassignment of track_ids even when recompute
    # is False.
    tracks.graph.nodes[1].pop(tracks.features.tracklet_key, None)
    solution_tracks = SolutionTracks.from_tracks(tracks)
    # should have reassigned new track_id to node 6
    assert solution_tracks.get_node_attr(6, solution_tracks.features.tracklet_key) == 4
    assert (
        solution_tracks.get_node_attr(1, solution_tracks.features.tracklet_key) == 1
    )  # still 1


def test_next_track_id_empty():
    graph = nx.DiGraph()
    seg = np.zeros(shape=(10, 100, 100, 100), dtype=np.uint64)
    tracks = SolutionTracks(graph, segmentation=seg, **track_attrs)
    assert tracks.get_next_track_id() == 1


def test_export_to_csv(
    graph_2d_with_computed_features, graph_3d_with_computed_features, tmp_path
):
    # Test backward-compatible default format (use_display_names=False)
    tracks = SolutionTracks(graph_2d_with_computed_features, **track_attrs, ndim=3)
    temp_file = tmp_path / "test_export_2d.csv"
    tracks.export_tracks(temp_file)
    with open(temp_file) as f:
        lines = f.readlines()

    assert len(lines) == tracks.graph.number_of_nodes() + 1  # add header

    # Backward compatible format: t, y, x, id, parent_id, track_id
    header = ["t", "y", "x", "id", "parent_id", "track_id"]
    assert lines[0].strip().split(",") == header

    tracks = SolutionTracks(graph_3d_with_computed_features, **track_attrs, ndim=4)
    temp_file = tmp_path / "test_export_3d.csv"
    tracks.export_tracks(temp_file)
    with open(temp_file) as f:
        lines = f.readlines()

    assert len(lines) == tracks.graph.number_of_nodes() + 1  # add header

    # Backward compatible format: t, z, y, x, id, parent_id, track_id
    header = ["t", "z", "y", "x", "id", "parent_id", "track_id"]
    assert lines[0].strip().split(",") == header

    # Test exporting a selection of nodes. We have 6 nodes in total and we ask to save
    # node 4 and 6. Because node 1 and 3 are ancestors of node 4, we expect them to be
    # included as well to maintain a valid graph without missing parents.
    tracks.export_tracks(temp_file, node_ids=[4, 6])
    with open(temp_file) as f:
        lines = f.readlines()

    assert len(lines) == 5  # (4 nodes + 1 header)

    # In backward-compatible format, node ID is 5th column (index 4): t, z, y, x, id, ...
    node_ids_in_csv = [int(line.split(",")[4]) for line in lines[1:]]
    expected_node_ids = [1, 3, 4, 6]
    assert sorted(node_ids_in_csv) == sorted(expected_node_ids), (
        f"Unexpected nodes in CSV: {node_ids_in_csv}"
    )

    # Backward compatible format
    header = ["t", "z", "y", "x", "id", "parent_id", "track_id"]
    assert lines[0].strip().split(",") == header


def test_export_to_csv_with_display_names(
    graph_2d_with_computed_features, graph_3d_with_computed_features, tmp_path
):
    """Test CSV export with use_display_names=True option."""
    from funtracks.import_export import export_to_csv

    # Test 2D with display names
    tracks = SolutionTracks(graph_2d_with_computed_features, **track_attrs, ndim=3)
    temp_file = tmp_path / "test_export_2d_display.csv"
    export_to_csv(tracks, temp_file, use_display_names=True)
    with open(temp_file) as f:
        lines = f.readlines()

    assert len(lines) == tracks.graph.number_of_nodes() + 1  # add header

    # With display names: ID, Parent ID, Time, y, x, Tracklet ID
    header = ["ID", "Parent ID", "Time", "y", "x", "Tracklet ID"]
    assert lines[0].strip().split(",") == header

    # Test 3D with display names
    tracks = SolutionTracks(graph_3d_with_computed_features, **track_attrs, ndim=4)
    temp_file = tmp_path / "test_export_3d_display.csv"
    export_to_csv(tracks, temp_file, use_display_names=True)
    with open(temp_file) as f:
        lines = f.readlines()

    assert len(lines) == tracks.graph.number_of_nodes() + 1  # add header

    # With display names: ID, Parent ID, Time, z, y, x, Tracklet ID
    header = ["ID", "Parent ID", "Time", "z", "y", "x", "Tracklet ID"]
    assert lines[0].strip().split(",") == header
