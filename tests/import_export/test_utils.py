import polars as pl

from funtracks.data_model import Tracks
from funtracks.import_export._utils import rename_feature


def test_rename_feature_basic(get_tracks):
    """Test that rename_feature renames a feature in annotators and features dict."""
    tracks = get_tracks(ndim=3, with_seg=True, prefill_track_ids=False)

    # Rename area feature to custom name
    rename_feature(tracks, "area", "my_area")

    # Check that feature was renamed in annotators
    assert "my_area" in tracks.annotators.all_features
    assert "area" not in tracks.annotators.all_features

    # Check that feature was renamed in features dict
    assert "my_area" in tracks.features
    assert "area" not in tracks.features


def test_rename_feature_updates_position_key(get_tracks):
    """Test that renaming position feature updates position_key in FeatureDict."""
    tracks = get_tracks(ndim=3, with_seg=True, prefill_track_ids=True)

    original_pos_key = tracks.features.position_key
    new_key = "custom_position"

    rename_feature(tracks, original_pos_key, new_key)

    assert tracks.features.position_key == new_key
    assert new_key in tracks.features


def test_rename_feature_updates_one_axis_of_split_position_key(get_graph):
    """Renaming one axis of a per-axis position_key moves just that entry.

    Leaving the list stale would point positions at a column that no longer holds
    them, while the annotator writes the renamed one.
    """
    graph = get_graph(ndim=3, with_seg=True, prefill_track_ids=True)
    for key in ["y", "x"]:
        graph.add_node_attr_key(key, default_value=None, dtype=pl.Float64)
    for node in graph.node_ids():
        pos = graph.nodes[node]["pos"]
        graph.nodes[node]["y"] = float(pos[0])
        graph.nodes[node]["x"] = float(pos[1])
    graph.remove_node_attr_key("pos")
    tracks = Tracks(
        graph, ndim=3, pos_attr=["y", "x"], time_attr="t", tracklet_attr="track_id"
    )

    rename_feature(tracks, "y", "depth")

    assert tracks.features.position_key == ["depth", "x"]
    assert tracks.annotators.all_features.keys() >= {"depth", "x"}


def test_rename_feature_updates_tracklet_key(get_tracks):
    """Test that renaming tracklet feature updates tracklet_key in FeatureDict."""
    tracks = get_tracks(ndim=3, with_seg=True, prefill_track_ids=True)

    original_track_key = tracks.features.tracklet_key
    new_key = "custom_track"

    rename_feature(tracks, original_track_key, new_key)

    assert tracks.features.tracklet_key == new_key
    assert new_key in tracks.features
