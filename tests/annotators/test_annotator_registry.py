import pytest

from funtracks.annotators import EdgeAnnotator, RegionpropsAnnotator, TrackAnnotator
from funtracks.data_model import SolutionTracks, Tracks

track_attrs = {"time_attr": "t", "tracklet_attr": "track_id"}


def test_annotator_registry_init_with_segmentation(graph_clean, segmentation_2d):
    """Test AnnotatorRegistry initializes regionprops and edge annotators with
    segmentation."""
    tracks = Tracks(graph_clean, segmentation=segmentation_2d, ndim=3, **track_attrs)

    annotator_types = [type(ann) for ann in tracks.annotators]
    assert RegionpropsAnnotator in annotator_types
    assert EdgeAnnotator in annotator_types
    assert TrackAnnotator not in annotator_types  # Not a SolutionTracks


def test_annotator_registry_init_without_segmentation(graph_2d_with_position):
    """Test AnnotatorRegistry doesn't create annotators without segmentation."""
    tracks = Tracks(graph_2d_with_position, segmentation=None, ndim=3, **track_attrs)

    annotator_types = [type(ann) for ann in tracks.annotators]
    assert RegionpropsAnnotator not in annotator_types
    assert EdgeAnnotator not in annotator_types
    assert TrackAnnotator not in annotator_types


def test_annotator_registry_init_solution_tracks(graph_clean, segmentation_2d):
    """Test AnnotatorRegistry creates all annotators for SolutionTracks with
    segmentation."""
    tracks = SolutionTracks(
        graph_clean, segmentation=segmentation_2d, ndim=3, **track_attrs
    )

    annotator_types = [type(ann) for ann in tracks.annotators]
    assert RegionpropsAnnotator in annotator_types
    assert EdgeAnnotator in annotator_types
    assert TrackAnnotator in annotator_types


def test_enable_disable_features(graph_clean, segmentation_2d):
    tracks = Tracks(graph_clean, segmentation=segmentation_2d, ndim=3, **track_attrs)

    nodes = list(tracks.graph.nodes())
    edges = list(tracks.graph.edges())

    # Core features (time, pos, area) should be in tracks.features and computed
    assert "pos" in tracks.features
    assert "t" in tracks.features
    assert "area" in tracks.features  # Core feature for backward compatibility
    assert tracks.graph.nodes[nodes[0]].get("pos") is not None
    assert tracks.graph.nodes[nodes[0]].get("area") is not None

    # Other features should NOT be in tracks.features initially
    assert "iou" not in tracks.features
    assert "circularity" not in tracks.features

    # Enable multiple features at once
    tracks.enable_features(["iou", "circularity"])

    # Features should now be in FeatureDict
    assert "iou" in tracks.features
    assert "circularity" in tracks.features

    # Verify values are actually computed on the graph
    assert tracks.graph.nodes[nodes[0]].get("circularity") is not None
    if edges:
        assert tracks.graph.edges[edges[0]].get("iou") is not None

    # Disable one feature
    tracks.disable_features(["area"])

    # area should be removed from FeatureDict
    assert "area" not in tracks.features
    assert "pos" in tracks.features
    assert "iou" in tracks.features
    assert "circularity" in tracks.features

    # Values still exist on the graph (disabling doesn't erase computed values)
    assert tracks.graph.nodes[nodes[0]].get("area") is not None

    # Disable the remaining enabled features
    tracks.disable_features(["pos", "iou", "circularity"])
    assert "pos" not in tracks.features
    assert "iou" not in tracks.features
    assert "circularity" not in tracks.features


def test_get_available_features(graph_clean, segmentation_2d):
    """Test get_available_features returns all features from all annotators."""
    tracks = SolutionTracks(
        graph_clean, segmentation=segmentation_2d, ndim=3, **track_attrs
    )

    available = tracks.get_available_features()

    # Should have features from all three annotators
    assert "pos" in available  # regionprops
    assert "area" in available  # regionprops
    assert "iou" in available  # edges
    assert "track_id" in available  # tracks


def test_enable_nonexistent_feature(graph_clean, segmentation_2d):
    """Test enabling a nonexistent feature raises KeyError."""
    tracks = Tracks(graph_clean, segmentation=segmentation_2d, ndim=3, **track_attrs)

    with pytest.raises(KeyError, match="Features not available"):
        tracks.enable_features(["nonexistent"])


def test_disable_nonexistent_feature(graph_clean, segmentation_2d):
    """Test disabling a nonexistent feature raises KeyError."""
    tracks = Tracks(graph_clean, segmentation=segmentation_2d, ndim=3, **track_attrs)

    with pytest.raises(KeyError, match="Features not available"):
        tracks.disable_features(["nonexistent"])


def test_compute_strict_validation(graph_clean, segmentation_2d):
    """Test that compute() strictly validates feature keys."""
    tracks = Tracks(graph_clean, segmentation=segmentation_2d, ndim=3, **track_attrs)

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
