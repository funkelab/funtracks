import networkx as nx
import numpy as np
import pytest

from funtracks.actions import AddNode
from funtracks.data_model import NodeAttr, SolutionTracks, Tracks


def test_next_track_id(graph_2d):
    tracks = SolutionTracks(graph_2d, ndim=3)
    assert tracks.get_next_track_id() == 6
    AddNode(
        tracks,
        node=10,
        attributes={"time": 3, "pos": [0, 0, 0, 0], "track_id": 10},
    )
    assert tracks.get_next_track_id() == 11


def test_node_id_to_track_id(graph_2d):
    tracks = SolutionTracks(graph_2d, ndim=3)
    with pytest.warns(
        DeprecationWarning,
        match="node_id_to_track_id property will be removed in funtracks v2. ",
    ):
        tracks.node_id_to_track_id  # noqa B018


def test_from_tracks_cls(graph_2d):
    tracks = Tracks(
        graph_2d, ndim=3, pos_attr="POSITION", time_attr="TIME", scale=(2, 2, 2)
    )
    solution_tracks = SolutionTracks.from_tracks(tracks)
    assert solution_tracks.graph == tracks.graph
    assert solution_tracks.segmentation == tracks.segmentation
    assert solution_tracks.time_attr == tracks.time_attr
    assert solution_tracks.pos_attr == tracks.pos_attr
    assert solution_tracks.scale == tracks.scale
    assert solution_tracks.ndim == tracks.ndim
    assert solution_tracks.get_node_attr(6, NodeAttr.TRACK_ID.value) == 5
    # delete track id on one node to trigger reassignment of track_ids.
    solution_tracks.graph.nodes[1].pop(NodeAttr.TRACK_ID.value, None)
    solution_tracks._initialize_track_ids()
    # should have reassigned new track_id to node 6
    assert solution_tracks.get_node_attr(6, NodeAttr.TRACK_ID.value) == 4
    assert solution_tracks.get_node_attr(1, NodeAttr.TRACK_ID.value) == 1  # still 1


def test_next_track_id_empty():
    graph = nx.DiGraph()
    seg = np.zeros(shape=(10, 100, 100, 100), dtype=np.uint64)
    tracks = SolutionTracks(graph, segmentation=seg)
    assert tracks.get_next_track_id() == 1


def test_export_to_csv(graph_2d, graph_3d, tmp_path):
    tracks = SolutionTracks(graph_2d, ndim=3)
    temp_file = tmp_path / "test_export_2d.csv"
    tracks.export_tracks(temp_file)
    with open(temp_file) as f:
        lines = f.readlines()

    assert len(lines) == tracks.graph.number_of_nodes() + 1  # add header

    header = ["t", "y", "x", "id", "parent_id", "track_id"]
    assert lines[0].strip().split(",") == header

    tracks = SolutionTracks(graph_3d, ndim=4)
    temp_file = tmp_path / "test_export_3d.csv"
    tracks.export_tracks(temp_file)
    with open(temp_file) as f:
        lines = f.readlines()

    assert len(lines) == tracks.graph.number_of_nodes() + 1  # add header

    header = ["t", "z", "y", "x", "id", "parent_id", "track_id"]
    assert lines[0].strip().split(",") == header
