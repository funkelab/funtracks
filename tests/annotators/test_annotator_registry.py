import pytest

from funtracks.annotators import EdgeAnnotator, RegionpropsAnnotator, TrackAnnotator
from funtracks.data_model import NodeAttr, SolutionTracks, Tracks


def test_annotator_registry_init_with_segmentation(graph_clean, segmentation_2d):
    """Test AnnotatorRegistry initializes regionprops and edge annotators with
    segmentation."""
    tracks = Tracks(graph_clean, segmentation=segmentation_2d, ndim=3)

    annotator_types = [type(ann) for ann in tracks.annotators.annotators]
    assert RegionpropsAnnotator in annotator_types
    assert EdgeAnnotator in annotator_types
    assert TrackAnnotator not in annotator_types  # Not a SolutionTracks


def test_annotator_registry_init_without_segmentation(graph_2d_with_position):
    """Test AnnotatorRegistry doesn't create annotators without segmentation."""
    tracks = Tracks(graph_2d_with_position, segmentation=None, ndim=3)

    annotator_types = [type(ann) for ann in tracks.annotators.annotators]
    assert RegionpropsAnnotator not in annotator_types
    assert EdgeAnnotator not in annotator_types
    assert TrackAnnotator not in annotator_types


def test_annotator_registry_init_solution_tracks(graph_clean, segmentation_2d):
    """Test AnnotatorRegistry creates all annotators for SolutionTracks with
    segmentation."""
    tracks = SolutionTracks(graph_clean, segmentation=segmentation_2d, ndim=3)

    annotator_types = [type(ann) for ann in tracks.annotators.annotators]
    assert RegionpropsAnnotator in annotator_types
    assert EdgeAnnotator in annotator_types
    assert TrackAnnotator in annotator_types


def test_enable_features(graph_clean, segmentation_2d):
    """Test enable_features enables and computes multiple features efficiently."""
    tracks = Tracks(graph_clean, segmentation=segmentation_2d, ndim=3)

    # Features should NOT be in tracks.features initially
    assert "pos" not in tracks.features
    assert "area" not in tracks.features
    assert "iou" not in tracks.features

    nodes = list(tracks.graph.nodes())
    edges = list(tracks.graph.edges())

    # Enable multiple features at once
    tracks.enable_features(["pos", "area", "iou"])

    # Features should now be in FeatureDict
    assert "pos" in tracks.features
    assert "area" in tracks.features
    assert "iou" in tracks.features

    # Check that node attributes were computed
    assert tracks.graph.nodes[nodes[0]].get("pos") is not None
    assert tracks.graph.nodes[nodes[0]].get("area") is not None

    # Check that edge attributes were computed
    if edges:
        assert tracks.graph.edges[edges[0]].get("iou") is not None


def test_get_available_features(graph_clean, segmentation_2d):
    """Test get_available_features returns all features from all annotators."""
    tracks = SolutionTracks(graph_clean, segmentation=segmentation_2d, ndim=3)

    available = tracks.get_available_features()

    # Should have features from all three annotators
    assert "pos" in available  # regionprops
    assert "area" in available  # regionprops
    assert "iou" in available  # edges
    assert NodeAttr.TRACK_ID.value in available  # tracks


def test_enable_disable_features(graph_clean, segmentation_2d):
    """Test enable_features, disable_features, and get_active_features."""
    tracks = Tracks(graph_clean, segmentation=segmentation_2d, ndim=3)

    # Initially, no features are active or in FeatureDict
    active = tracks.get_active_features()
    assert len(active) == 0
    assert "pos" not in tracks.features
    assert "area" not in tracks.features
    assert "iou" not in tracks.features

    # Verify features are not on the graph yet
    nodes = list(tracks.graph.nodes())
    edges = list(tracks.graph.edges())
    assert tracks.graph.nodes[nodes[0]].get("pos") is None
    assert tracks.graph.nodes[nodes[0]].get("area") is None

    # Enable multiple features at once
    tracks.enable_features(["pos", "area", "iou"])

    # All three should be active, in FeatureDict, and computed on graph
    active = tracks.get_active_features()
    assert len(active) == 3
    assert "pos" in active
    assert "area" in active
    assert "iou" in active
    assert "pos" in tracks.features
    assert "area" in tracks.features
    assert "iou" in tracks.features

    # Verify values are actually computed on the graph
    assert tracks.graph.nodes[nodes[0]].get("pos") is not None
    assert tracks.graph.nodes[nodes[0]].get("area") is not None
    if edges:
        assert tracks.graph.edges[edges[0]].get("iou") is not None

    # Disable one feature
    tracks.disable_features(["area"])

    # Only pos and IoU should remain active
    active = tracks.get_active_features()
    assert len(active) == 2
    assert "pos" in active
    assert "area" not in active  # Disabled
    assert "iou" in active
    assert "pos" in tracks.features
    assert "area" not in tracks.features  # Removed from FeatureDict
    assert "iou" in tracks.features

    # Values still exist on the graph (disabling doesn't erase computed values)
    assert tracks.graph.nodes[nodes[0]].get("area") is not None

    # Disable the remaining features
    tracks.disable_features(["pos", "iou"])
    active = tracks.get_active_features()
    assert len(active) == 0
    assert "pos" not in tracks.features
    assert "iou" not in tracks.features


def test_enable_features_computes(graph_clean, segmentation_2d):
    """Test enable_features computes the features."""
    tracks = Tracks(graph_clean, segmentation=segmentation_2d, ndim=3)

    # Enable (computes)
    tracks.enable_features(["area", "pos"])

    # Features should be computed
    nodes = list(tracks.graph.nodes())
    assert tracks.graph.nodes[nodes[0]].get("area") is not None
    assert tracks.graph.nodes[nodes[0]].get("pos") is not None


def test_enable_nonexistent_feature(graph_clean, segmentation_2d):
    """Test enabling a nonexistent feature raises KeyError."""
    tracks = Tracks(graph_clean, segmentation=segmentation_2d, ndim=3)

    with pytest.raises(KeyError, match="Features not available"):
        tracks.enable_features(["nonexistent"])


def test_disable_nonexistent_feature(graph_clean, segmentation_2d):
    """Test disabling a nonexistent feature raises KeyError."""
    tracks = Tracks(graph_clean, segmentation=segmentation_2d, ndim=3)

    with pytest.raises(KeyError, match="Features not available"):
        tracks.disable_features(["nonexistent"])


def test_compute_strict_validation(graph_clean, segmentation_2d):
    """Test that compute() strictly validates feature keys."""
    tracks = Tracks(graph_clean, segmentation=segmentation_2d, ndim=3)

    # Get the RegionpropsAnnotator from the annotators
    rp_ann = next(
        ann
        for ann in tracks.annotators.annotators
        if isinstance(ann, RegionpropsAnnotator)
    )

    # Enable area first
    tracks.enable_features(["area"])

    # Valid feature key should work
    rp_ann.compute(["area"])

    # Invalid feature key should raise KeyError
    with pytest.raises(KeyError, match="Features not available or not enabled"):
        rp_ann.compute(["nonexistent_feature"])

    # Disabled feature should raise KeyError
    tracks.disable_features(["area"])
    with pytest.raises(KeyError, match="Features not available or not enabled"):
        rp_ann.compute(["area"])

    # Mix of valid and invalid should raise KeyError
    tracks.enable_features(["area"])
    with pytest.raises(KeyError, match="Features not available or not enabled"):
        rp_ann.compute(["area", "nonexistent"])

    # None should still work (compute all enabled features)
    rp_ann.compute()
