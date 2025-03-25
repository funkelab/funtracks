import networkx as nx
import numpy as np

from funtracks.data_model import SolutionTracks


def test_next_track_id(graph_2d):
    tracks = SolutionTracks(graph_2d, ndim=3)
    assert tracks.get_next_track_id() == 6
    tracks.add_node(10, time=3, position=[0, 0, 0, 0], attrs={"track_id": 10})
    assert tracks.get_next_track_id() == 11


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
