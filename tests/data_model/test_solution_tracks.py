import networkx as nx
import numpy as np

from funtracks.data_model import NodeAttr, SolutionTracks, Tracks
from funtracks.data_model.actions import AddNodes
from funtracks.data_model.utils import convert_nx_to_td_indexedrxgraph


def test_next_track_id(graph_2d):
    tracks = SolutionTracks(graph_2d, ndim=3)
    assert tracks.get_next_track_id() == 6
    AddNodes(
        tracks,
        nodes=[10],
        attributes={"t": [3], "pos": [[0, 0]], "track_id": [10]},
        # TODO: Caroline/Anniek, why did this test have a 4D pos vector?
    )
    assert tracks.get_next_track_id() == 11


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
    # solution_tracks.graph.nodes[1].pop(NodeAttr.TRACK_ID.value, None)
    solution_tracks.graph.update_node_attrs(
        attrs={NodeAttr.TRACK_ID.value: [None]}, node_ids=[1]
    )
    solution_tracks._initialize_track_ids()
    # should have reassigned new track_id to node 6
    assert solution_tracks.get_node_attr(6, NodeAttr.TRACK_ID.value) == 4
    assert solution_tracks.get_node_attr(1, NodeAttr.TRACK_ID.value) == 1  # still 1


def test_next_track_id_empty():
    # graph_td = nx.DiGraph()
    graph_td = convert_nx_to_td_indexedrxgraph(nx.DiGraph())
    # TODO: somewhere we have to make track_id a mandatory node attr
    graph_td.add_node_attr_key(key="track_id", default_value=0)
    seg = np.zeros(shape=(10, 100, 100, 100), dtype=np.uint64)
    tracks = SolutionTracks(graph_td, segmentation=seg)
    assert tracks.get_next_track_id() == 1


def test_export_to_csv(graph_2d, graph_3d, tmp_path):
    tracks = SolutionTracks(graph_2d, ndim=3)
    temp_file = tmp_path / "test_export_2d.csv"
    tracks.export_tracks(temp_file)
    with open(temp_file) as f:
        lines = f.readlines()

    assert len(lines) == tracks.graph.num_nodes + 1  # add header

    header = ["t", "y", "x", "id", "parent_id", "track_id"]
    assert lines[0].strip().split(",") == header

    tracks = SolutionTracks(graph_3d, ndim=4)
    temp_file = tmp_path / "test_export_3d.csv"
    tracks.export_tracks(temp_file)
    with open(temp_file) as f:
        lines = f.readlines()

    assert len(lines) == tracks.graph.num_nodes + 1  # add header

    header = ["t", "z", "y", "x", "id", "parent_id", "track_id"]
    assert lines[0].strip().split(",") == header
