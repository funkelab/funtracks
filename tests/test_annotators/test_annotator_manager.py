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

    node = list(tracks.graph.nodes())[0]

    # Update the node (in a real scenario, segmentation would change)
    manager.update(node)

    # Position should still exist (even if it might be the same)
    assert tracks.graph.nodes[node].get("pos") is not None


def test_update_edge(graph_2d, segmentation_2d):
    """Test update routes edge updates to edge annotator."""
    tracks = Tracks(graph_2d, segmentation=segmentation_2d, ndim=3)
    manager = AnnotatorManager(tracks)
    manager.compute_all()

    edges = list(tracks.graph.edges())
    if not edges:
        pytest.skip("No edges in graph")

    edge = edges[0]

    # Update the edge
    manager.update(edge)

    # IoU should still exist
    assert tracks.graph.edges[edge].get("IoU") is not None


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

    # Disable a feature
    manager.disable_feature("area")
    active = manager.get_active_features()
    assert "area" not in active

    # Re-enable it
    manager.enable_feature("area")
    active = manager.get_active_features()
    assert "area" in active


def test_enable_feature_with_compute(graph_2d, segmentation_2d):
    """Test enable_feature with compute=True computes the feature."""
    tracks = Tracks(graph_2d, segmentation=segmentation_2d, ndim=3)
    manager = AnnotatorManager(tracks)

    # Disable and clear the feature
    manager.disable_feature("area")

    # Re-enable with compute
    manager.enable_feature("area", compute=True)

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
