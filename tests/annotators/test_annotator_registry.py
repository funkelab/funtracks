import pytest

from funtracks.annotators import EdgeAnnotator, RegionpropsAnnotator, TrackAnnotator
from funtracks.data_model import Tracks

track_attrs = {"time_attr": "t", "tracklet_attr": "track_id"}
# Even without an explicit tracklet_attr, every Tracks gets a TrackAnnotator and a
# default tracklet key (no "plain" vs "solution" split).
plain_attrs = {"time_attr": "t"}


def test_annotator_registry_init_with_segmentation(
    graph_2d_with_segmentation,
):
    """Test AnnotatorRegistry initializes regionprops and edge annotators with
    segmentation."""
    tracks = Tracks(
        graph_2d_with_segmentation,
        ndim=3,
        **plain_attrs,
    )

    annotator_types = [type(ann) for ann in tracks.annotators]
    assert RegionpropsAnnotator in annotator_types
    assert EdgeAnnotator in annotator_types
    assert TrackAnnotator in annotator_types  # every Tracks has track ids


def test_annotator_registry_init_without_segmentation(graph_2d_with_position):
    """Without segmentation: no regionprops/edge annotators, but a TrackAnnotator is
    still registered (track ids are a core feature of every Tracks)."""
    tracks = Tracks(graph_2d_with_position, ndim=3, **plain_attrs)

    annotator_types = [type(ann) for ann in tracks.annotators]
    assert RegionpropsAnnotator not in annotator_types
    assert EdgeAnnotator not in annotator_types
    assert TrackAnnotator in annotator_types


def test_annotator_registry_init_solution_tracks(
    graph_2d_with_segmentation,
):
    """Test AnnotatorRegistry creates all annotators for Tracks with
    segmentation."""
    tracks = Tracks(
        graph_2d_with_segmentation,
        ndim=3,
        **track_attrs,
    )

    annotator_types = [type(ann) for ann in tracks.annotators]
    assert RegionpropsAnnotator in annotator_types
    assert EdgeAnnotator in annotator_types
    assert TrackAnnotator in annotator_types


def test_enable_disable_features(graph_2d_with_segmentation):
    tracks = Tracks(
        graph_2d_with_segmentation,
        ndim=3,
        **track_attrs,
    )

    nodes = list(tracks.graph_solution.node_ids())
    edges = list(tracks.graph_solution.edge_ids())

    # Core features (time, pos) should be in tracks.features and computed
    assert "pos" in tracks.features
    assert "t" in tracks.features
    assert tracks.graph_solution.nodes[nodes[0]]["pos"] is not None

    # area and other features should NOT be in tracks.features initially
    assert "area" not in tracks.features
    assert "iou" not in tracks.features
    assert "circularity" not in tracks.features

    # Enable multiple features at once
    tracks.enable_features(["area", "iou", "circularity"])

    # Features should now be in FeatureDict
    assert "iou" in tracks.features
    assert "circularity" in tracks.features

    # Verify values are actually computed on the graph
    assert tracks.graph_solution.nodes[nodes[0]]["circularity"] is not None
    if edges:
        assert None not in tracks.graph_solution.edge_attrs()["iou"].to_list()

    # Disable one feature
    tracks.disable_features(["area"])

    # area should be removed from FeatureDict
    assert "area" not in tracks.features
    assert "pos" in tracks.features
    assert "iou" in tracks.features
    assert "circularity" in tracks.features

    # Values no longer exist in the graph for tracksdata
    # assert tracks.graph_solution.nodes[1]["area"] is not None

    # Disable the remaining enabled features
    tracks.disable_features(["pos", "iou", "circularity"])
    assert "pos" not in tracks.features
    assert "iou" not in tracks.features
    assert "circularity" not in tracks.features


def test_area_on_graph_not_auto_activated(graph_2d_with_segmentation):
    """Area pre-populated on a raw graph must not be auto-activated.

    Lock in the behavior introduced when area was removed from the core
    auto-detected features: even though the graph already carries area values,
    Tracks(graph) without an explicit FeatureDict must leave area out of
    tracks.features. Callers opt in via enable_features(["area"]).
    """
    assert "area" in graph_2d_with_segmentation.node_attr_keys()

    tracks = Tracks(graph_2d_with_segmentation, ndim=3, **track_attrs)

    assert "area" not in tracks.features
    assert "area" in tracks.annotators.all_features

    tracks.enable_features(["area"])
    assert "area" in tracks.features


def test_get_available_features(graph_2d_with_segmentation):
    """Test get_available_features returns all features from all annotators."""
    tracks = Tracks(
        graph_2d_with_segmentation,
        ndim=3,
        **track_attrs,
    )

    available = tracks.get_available_features()

    # Should have features from all three annotators
    assert "pos" in available  # regionprops
    assert "area" in available  # regionprops
    assert "iou" in available  # edges
    assert "track_id" in available  # tracks


def test_enable_nonexistent_feature(graph_clean):
    """Test enabling a nonexistent feature raises KeyError."""
    tracks = Tracks(graph_clean, ndim=3, **track_attrs)

    with pytest.raises(KeyError, match="Features not available"):
        tracks.enable_features(["nonexistent"])


def test_disable_nonexistent_feature(graph_clean):
    """Test disabling a nonexistent feature raises KeyError."""
    tracks = Tracks(graph_clean, ndim=3, **track_attrs)

    with pytest.raises(KeyError, match="Features not available"):
        tracks.disable_features(["nonexistent"])


def test_compute_strict_validation(graph_2d_with_segmentation):
    """Test that compute() strictly validates feature keys."""
    tracks = Tracks(
        graph_2d_with_segmentation,
        ndim=3,
        **track_attrs,
    )

    # Get the RegionpropsAnnotator from the annotators
    rp_ann = next(
        ann for ann in tracks.annotators if isinstance(ann, RegionpropsAnnotator)
    )

    # Enable area first
    tracks.enable_features(["area"])

    # Valid feature key should work
    rp_ann.compute(["area"])

    # Invalid feature key should not raise KeyError
    rp_ann.compute(["nonexistent_feature"])

    # Disabled feature should not raise KeyError
    tracks.disable_features(["area"])
    rp_ann.compute(["area"])

    # None should still work (compute all enabled features)
    rp_ann.compute()
