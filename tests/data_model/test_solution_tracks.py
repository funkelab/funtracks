import numpy as np
from tracksdata.array import GraphArrayView
from tracksdata.nodes._mask import Mask

from funtracks.data_model import NodeAttr, SolutionTracks, Tracks
from funtracks.data_model.actions import AddNodes
from funtracks.data_model.tracksdata_utils import create_empty_graphview_graph


def test_next_track_id(graph_2d):
    tracks = SolutionTracks(graph_2d, segmentation_shape=(5, 100, 100), ndim=3)

    assert tracks.get_next_track_id() == 6
    mask = Mask(np.ones((3, 3)), bbox=np.array([0, 0, 3, 3]))
    AddNodes(
        tracks,
        nodes=[10],
        attributes={
            "t": [3],
            "pos": [np.array([0, 0])],
            "track_id": [10],
            "area": [9],
            "mask": [mask],
            "bbox": [mask.bbox],
        },
    )
    assert tracks.get_next_track_id() == 11


def test_from_tracks_cls(graph_2d):
    tracks = Tracks(
        graph_2d, segmentation_shape=(5,100,100), ndim=3, pos_attr="POSITION", time_attr="TIME", scale=(2, 2, 2)
    )
    solution_tracks = SolutionTracks.from_tracks(tracks)
    assert solution_tracks.graph == tracks.graph
    np.testing.assert_array_equal(np.asarray(solution_tracks.segmentation), np.asarray(solution_tracks.segmentation))
    assert solution_tracks.time_attr == tracks.time_attr
    assert solution_tracks.pos_attr == tracks.pos_attr
    assert solution_tracks.scale == tracks.scale
    assert solution_tracks.ndim == tracks.ndim
    assert solution_tracks.get_node_attr(6, NodeAttr.TRACK_ID.value) == 5
    # delete track id on one node to trigger reassignment of track_ids.
    solution_tracks.graph.update_node_attrs(
        attrs={NodeAttr.TRACK_ID.value: [None]}, node_ids=[1]
    )
    solution_tracks._initialize_track_ids()
    # should have reassigned new track_id to node 6
    assert solution_tracks.get_node_attr(6, NodeAttr.TRACK_ID.value) == 4
    assert solution_tracks.get_node_attr(1, NodeAttr.TRACK_ID.value) == 1  # still 1


def test_next_track_id_empty():
    graph_td = create_empty_graphview_graph(database=":memory:", position_attrs=["pos"])

    tracks = SolutionTracks(graph_td, segmentation_shape=(10, 100, 100, 100), ndim=4)
    assert tracks.get_next_track_id() == 1


def test_export_to_csv(graph_2d, graph_3d, tmp_path):
    tracks = SolutionTracks(graph_2d, segmentation_shape=(5, 100, 100), ndim=3)
    temp_file = tmp_path / "test_export_2d.csv"
    tracks.export_tracks(temp_file)
    with open(temp_file) as f:
        lines = f.readlines()

    assert len(lines) == tracks.graph.num_nodes + 1  # add header

    header = ["t", "y", "x", "id", "parent_id", "track_id"]
    assert lines[0].strip().split(",") == header

    tracks = SolutionTracks(graph_3d, segmentation_shape=(5, 100, 100, 100), ndim=4)
    temp_file = tmp_path / "test_export_3d.csv"
    tracks.export_tracks(temp_file)
    with open(temp_file) as f:
        lines = f.readlines()

    assert len(lines) == tracks.graph.num_nodes + 1  # add header

    header = ["t", "z", "y", "x", "id", "parent_id", "track_id"]
    assert lines[0].strip().split(",") == header
