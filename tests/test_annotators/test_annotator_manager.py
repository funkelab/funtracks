import pytest

from funtracks.annotators import AnnotatorManager
from funtracks.data_model import SolutionTracks, Tracks


def test_annotator_manager_initialization_with_segmentation(graph_2d, segmentation_2d):
    """Test AnnotatorManager initializes regionprops and edge annotators with
    segmentation."""
    tracks = Tracks(graph_2d, segmentation=segmentation_2d, ndim=3)
    manager = AnnotatorManager(tracks)

    assert "regionprops" in manager.annotators
    assert "edges" in manager.annotators
    assert "tracks" not in manager.annotators  # Not a SolutionTracks


def test_annotator_manager_initialization_without_segmentation(graph_2d):
    """Test AnnotatorManager doesn't create annotators without segmentation."""
    tracks = Tracks(graph_2d, segmentation=None, ndim=3)
    manager = AnnotatorManager(tracks)

    assert "regionprops" not in manager.annotators
    assert "edges" not in manager.annotators
    assert "tracks" not in manager.annotators


def test_annotator_manager_initialization_solution_tracks(graph_2d, segmentation_2d):
    """Test AnnotatorManager creates all annotators for SolutionTracks with
    segmentation."""
    tracks = SolutionTracks(graph_2d, segmentation=segmentation_2d, ndim=3)
    manager = AnnotatorManager(tracks)

    assert "regionprops" in manager.annotators
    assert "edges" in manager.annotators
    assert "tracks" in manager.annotators


def test_compute_all(graph_2d, segmentation_2d):
    """Test compute_all computes features from all annotators."""
    tracks = Tracks(graph_2d, segmentation=segmentation_2d, ndim=3)
    manager = AnnotatorManager(tracks)

    # Features should already be in tracks.features (added during init)
    assert "pos" in tracks.features
    assert "area" in tracks.features
    assert "IoU" in tracks.features

    nodes = list(tracks.graph.nodes())
    edges = list(tracks.graph.edges())

    # Clear existing attributes from the fixture
    for node in nodes:
        for attr in ["pos", "area"]:
            if attr in tracks.graph.nodes[node]:
                del tracks.graph.nodes[node][attr]
    for edge in edges:
        if "IoU" in tracks.graph.edges[edge]:
            del tracks.graph.edges[edge]["IoU"]

    # Compute all features
    manager.compute_all()

    # Check that node attributes were set
    assert tracks.graph.nodes[nodes[0]].get("pos") is not None
    assert tracks.graph.nodes[nodes[0]].get("area") is not None

    # Check that edge attributes were set
    if edges:
        assert tracks.graph.edges[edges[0]].get("IoU") is not None


def test_update_node(graph_2d, segmentation_2d):
    """Test update routes node updates to regionprops annotator."""
    tracks = Tracks(graph_2d, segmentation=segmentation_2d, ndim=3)
    manager = AnnotatorManager(tracks)
    manager.compute_all()

    node_id = 3

    # Modify the segmentation by removing all but one pixel
    orig_pixels = tracks.get_pixels(node_id)
    assert orig_pixels is not None
    pixels_to_remove = tuple(orig_pixels[d][1:] for d in range(len(orig_pixels)))
    tracks.set_pixels(pixels_to_remove, 0)
    expected_area = 1

    # Update the node - should recompute regionprops features
    manager.update(node_id)

    # Check that features were updated
    assert tracks.get_area(node_id) == expected_area


def test_update_edge(graph_2d, segmentation_2d):
    """Test update routes edge updates to edge annotator."""
    tracks = Tracks(graph_2d, segmentation=segmentation_2d, ndim=3)
    manager = AnnotatorManager(tracks)
    manager.compute_all()

    edge_id = (1, 3)
    node_id = 3
    orig_iou = tracks.graph.edges[edge_id]["IoU"]

    # Modify the segmentation by removing all but one pixel from node 3
    orig_pixels = tracks.get_pixels(node_id)
    assert orig_pixels is not None
    pixels_to_remove = tuple(orig_pixels[d][1:] for d in range(len(orig_pixels)))
    tracks.set_pixels(pixels_to_remove, 0)

    # Update the edge - should recompute IoU
    manager.update(edge_id)

    # Check that IoU was recomputed (value should have changed)
    new_iou = tracks.graph.edges[edge_id]["IoU"]
    assert new_iou != orig_iou
    assert new_iou == pytest.approx(0.0, abs=0.001)


def test_recompute_tracks(graph_2d, segmentation_2d):
    """Test recompute_tracks calls TrackAnnotator.compute."""
    tracks = SolutionTracks(graph_2d, segmentation=segmentation_2d, ndim=3)
    manager = AnnotatorManager(tracks)
    manager.compute_all()

    # tracklet_id should be set
    nodes = list(tracks.graph.nodes())
    assert tracks.graph.nodes[nodes[0]].get("tracklet_id") is not None

    # Recompute tracks
    manager.recompute_tracks()

    # tracklet_id should still exist
    assert tracks.graph.nodes[nodes[0]].get("tracklet_id") is not None


def test_get_available_features(graph_2d, segmentation_2d):
    """Test get_available_features returns all features from all annotators."""
    tracks = SolutionTracks(graph_2d, segmentation=segmentation_2d, ndim=3)
    manager = AnnotatorManager(tracks)

    available = manager.get_available_features()

    # Should have features from all three annotators
    assert "pos" in available  # regionprops
    assert "area" in available  # regionprops
    assert "IoU" in available  # edges
    assert "tracklet_id" in available  # tracks


def test_get_active_features(graph_2d, segmentation_2d):
    """Test get_active_features returns only active features."""
    tracks = Tracks(graph_2d, segmentation=segmentation_2d, ndim=3)
    manager = AnnotatorManager(tracks)

    # Disable a feature
    manager.disable_feature("area")

    active = manager.get_active_features()

    assert "pos" in active
    assert "area" not in active  # Disabled
    assert "IoU" in active

    # FeatureDict should also reflect the disabled feature
    assert "area" not in tracks.features


def test_get_feature_source(graph_2d, segmentation_2d):
    """Test get_feature_source returns correct annotator name."""
    tracks = SolutionTracks(graph_2d, segmentation=segmentation_2d, ndim=3)
    manager = AnnotatorManager(tracks)

    assert manager.get_feature_source("pos") == "regionprops"
    assert manager.get_feature_source("area") == "regionprops"
    assert manager.get_feature_source("IoU") == "edges"
    assert manager.get_feature_source("tracklet_id") == "tracks"
    assert manager.get_feature_source("nonexistent") is None


def test_enable_disable_feature(graph_2d, segmentation_2d):
    """Test enable_feature and disable_feature."""
    tracks = Tracks(graph_2d, segmentation=segmentation_2d, ndim=3)
    manager = AnnotatorManager(tracks)

    # Feature should be in FeatureDict initially
    assert "area" in tracks.features

    # Disable a feature
    manager.disable_feature("area")
    active = manager.get_active_features()
    assert "area" not in active
    # Feature should be removed from FeatureDict
    assert "area" not in tracks.features

    # Re-enable it
    manager.enable_feature("area")
    active = manager.get_active_features()
    assert "area" in active
    # Feature should be back in FeatureDict
    assert "area" in tracks.features


def test_enable_feature_with_compute(graph_2d, segmentation_2d):
    """Test enable_feature always computes the feature."""
    tracks = Tracks(graph_2d, segmentation=segmentation_2d, ndim=3)
    manager = AnnotatorManager(tracks)

    # Disable and clear the feature
    manager.disable_feature("area")

    # Re-enable (always computes)
    manager.enable_feature("area")

    # Feature should be computed
    nodes = list(tracks.graph.nodes())
    assert tracks.graph.nodes[nodes[0]].get("area") is not None


def test_enable_nonexistent_feature(graph_2d, segmentation_2d):
    """Test enabling a nonexistent feature raises KeyError."""
    tracks = Tracks(graph_2d, segmentation=segmentation_2d, ndim=3)
    manager = AnnotatorManager(tracks)

    with pytest.raises(KeyError, match="Feature 'nonexistent' not available"):
        manager.enable_feature("nonexistent")


def test_disable_nonexistent_feature(graph_2d, segmentation_2d):
    """Test disabling a nonexistent feature raises KeyError."""
    tracks = Tracks(graph_2d, segmentation=segmentation_2d, ndim=3)
    manager = AnnotatorManager(tracks)

    with pytest.raises(KeyError, match="Feature 'nonexistent' not available"):
        manager.disable_feature("nonexistent")


def test_compute_strict_validation(graph_2d, segmentation_2d):
    """Test that compute() strictly validates feature keys."""
    tracks = Tracks(graph_2d, segmentation=segmentation_2d, ndim=3)
    manager = AnnotatorManager(tracks)

    # Valid feature key should work
    manager.annotators["regionprops"].compute(["area"])

    # Invalid feature key should raise KeyError
    with pytest.raises(KeyError, match="Features not available or not enabled"):
        manager.annotators["regionprops"].compute(["nonexistent_feature"])

    # Disabled feature should raise KeyError
    manager.disable_feature("area")
    with pytest.raises(KeyError, match="Features not available or not enabled"):
        manager.annotators["regionprops"].compute(["area"])

    # Mix of valid and invalid should raise KeyError
    manager.enable_feature("area")
    with pytest.raises(KeyError, match="Features not available or not enabled"):
        manager.annotators["regionprops"].compute(["area", "nonexistent"])

    # None should still work (compute all enabled features)
    manager.annotators["regionprops"].compute()
