import networkx as nx
import numpy as np
import pytest

from funtracks.actions import AddNode
from funtracks.data_model import SolutionTracks, Tracks

track_attrs = {"time_attr": "t", "tracklet_attr": "track_id"}


def test_recompute_track_ids(graph_2d_with_computed_features):
    tracks = SolutionTracks(
        graph_2d_with_computed_features,
        ndim=3,
        **track_attrs,
    )
    assert tracks.get_next_track_id() == 5


def test_next_track_id(graph_2d_with_computed_features):
    tracks = SolutionTracks(
        graph_2d_with_computed_features,
        ndim=3,
        **track_attrs,
        existing_features=[track_attrs["tracklet_attr"]],
    )
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
    tracks = SolutionTracks(graph_2d_with_computed_features, **track_attrs, ndim=3)
    temp_file = tmp_path / "test_export_2d.csv"
    tracks.export_tracks(temp_file)
    with open(temp_file) as f:
        lines = f.readlines()

    assert len(lines) == tracks.graph.number_of_nodes() + 1  # add header

    header = ["t", "y", "x", "id", "parent_id", "track_id"]
    assert lines[0].strip().split(",") == header

    tracks = SolutionTracks(graph_3d_with_computed_features, **track_attrs, ndim=4)
    temp_file = tmp_path / "test_export_3d.csv"
    tracks.export_tracks(temp_file)
    with open(temp_file) as f:
        lines = f.readlines()

    assert len(lines) == tracks.graph.number_of_nodes() + 1  # add header

    header = ["t", "z", "y", "x", "id", "parent_id", "track_id"]
    assert lines[0].strip().split(",") == header
