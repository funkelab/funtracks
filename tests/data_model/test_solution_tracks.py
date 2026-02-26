import numpy as np

from funtracks.actions import AddNode
from funtracks.data_model import SolutionTracks, Tracks
from funtracks.import_export import export_to_csv
from funtracks.user_actions import UserUpdateSegmentation
from funtracks.utils.tracksdata_utils import create_empty_graphview_graph

track_attrs = {"time_attr": "t", "tracklet_attr": "track_id"}


def test_recompute_track_ids(graph_2d_with_track_id):
    tracks = SolutionTracks(
        graph_2d_with_track_id,
        ndim=3,
        **track_attrs,
    )
    assert tracks.get_next_track_id() == 6


def test_next_track_id(graph_2d_with_track_id):
    tracks = SolutionTracks(graph_2d_with_track_id, ndim=3, **track_attrs)
    assert tracks.get_next_track_id() == 6
    AddNode(
        tracks,
        node=10,
        attributes={"t": 3, "pos": [0, 0], "track_id": 10},
    )
    assert tracks.get_next_track_id() == 11


def test_from_tracks_cls(graph_2d_with_segmentation):
    tracks = Tracks(
        graph_2d_with_segmentation,
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


def test_from_tracks_cls_recompute(graph_2d_with_segmentation):
    tracks = Tracks(
        graph_2d_with_segmentation,
        ndim=3,
        pos_attr="POSITION",
        time_attr="TIME",
        tracklet_attr=track_attrs["tracklet_attr"],
        scale=(2, 2, 2),
    )
    # delete track id (default value -1) on one node triggers reassignment of
    # track_ids even when recompute is False.
    tracks.graph.nodes[1][tracks.features.tracklet_key] = -1
    solution_tracks = SolutionTracks.from_tracks(tracks)
    # should have reassigned new track_id to node 6
    assert solution_tracks.get_node_attr(6, solution_tracks.features.tracklet_key) == 4
    assert (
        solution_tracks.get_node_attr(1, solution_tracks.features.tracklet_key) == 1
    )  # still 1


def test_update_segmentation(graph_2d_with_segmentation):
    tracks = SolutionTracks(
        graph_2d_with_segmentation,
        ndim=3,
        **track_attrs,
    )
    pix = tracks.get_pixels(1)
    assert isinstance(pix, tuple)
    UserUpdateSegmentation(
        tracks,
        new_value=99,
        updated_pixels=[(pix, 0)],
        current_track_id=6,
    )
    assert np.asarray(tracks.segmentation)[0, 50, 50] == 99


def test_next_track_id_empty():
    graph = create_empty_graphview_graph(
        node_attributes=["pos", "track_id"],
        edge_attributes=[],
    )
    tracks = SolutionTracks(graph, ndim=4, **track_attrs)
    assert tracks.get_next_track_id() == 1


def test_get_lineage_id_without_lineage_key(graph_2d_with_track_id):
    """Test that get_lineage_id returns None when lineage_key is not set."""
    graph = graph_2d_with_track_id
    # TODO Teun: question: the original code added node 1, but this was already present?
    # So, what was tested here exactly?
    # graph.add_node(1, t=0, pos=[0, 0], track_id=1)
    graph.add_node(
        attrs={"t": 1, "pos": [0, 0], "track_id": 1}, index=7, validate_keys=False
    )
    tracks = SolutionTracks(graph, ndim=3, **track_attrs)

    # Unset lineage_key to test the None path
    tracks.features.lineage_key = None

    # get_lineage_id should return None when lineage_key is not set
    assert tracks.get_lineage_id(1) is None


def test_export_to_csv_with_display_names(
    graph_2d_with_segmentation, graph_3d_with_segmentation, tmp_path
):
    """Test CSV export with use_display_names=True option."""
    # Test 2D with display names
    tracks = SolutionTracks(graph_2d_with_segmentation, **track_attrs, ndim=3)
    temp_file = tmp_path / "test_export_2d_display.csv"
    export_to_csv(tracks, temp_file, use_display_names=True)
    with open(temp_file) as f:
        lines = f.readlines()

    assert len(lines) == tracks.graph.num_nodes() + 1  # add header

    # With display names: ID, Parent ID, Time, y, x, Tracklet ID, Lineage ID
    header = ["ID", "Parent ID", "Time", "y", "x", "Area", "Tracklet ID", "Lineage ID"]
    assert lines[0].strip().split(",") == header

    # Test 3D with display names
    tracks = SolutionTracks(graph_3d_with_segmentation, **track_attrs, ndim=4)
    temp_file = tmp_path / "test_export_3d_display.csv"
    export_to_csv(tracks, temp_file, use_display_names=True)
    with open(temp_file) as f:
        lines = f.readlines()

    assert len(lines) == tracks.graph.num_nodes() + 1  # add header

    # With display names: ID, Parent ID, Time, z, y, x, Tracklet ID, Lineage ID
    header = [
        "ID",
        "Parent ID",
        "Time",
        "z",
        "y",
        "x",
        "Volume",
        "Tracklet ID",
        "Lineage ID",
    ]
    assert lines[0].strip().split(",") == header
