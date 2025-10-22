import pytest

from funtracks.data_model import SolutionTracks, Tracks


def test_annotator_manager_initialization_with_segmentation(graph_clean, segmentation_2d):
    """Test AnnotatorManager initializes regionprops and edge annotators with
    segmentation."""
    tracks = Tracks(graph_clean, segmentation=segmentation_2d, ndim=3)

    assert "regionprops" in tracks.annotator_manager.annotators
    assert "edges" in tracks.annotator_manager.annotators
    assert "tracks" not in tracks.annotator_manager.annotators  # Not a SolutionTracks


def test_annotator_manager_initialization_without_segmentation(graph_2d_with_position):
    """Test AnnotatorManager doesn't create annotators without segmentation."""
    tracks = Tracks(graph_2d_with_position, segmentation=None, ndim=3)

    assert "regionprops" not in tracks.annotator_manager.annotators
    assert "edges" not in tracks.annotator_manager.annotators
    assert "tracks" not in tracks.annotator_manager.annotators


def test_annotator_manager_initialization_solution_tracks(graph_clean, segmentation_2d):
    """Test AnnotatorManager creates all annotators for SolutionTracks with
    segmentation."""
    tracks = SolutionTracks(graph_clean, segmentation=segmentation_2d, ndim=3)

    assert "regionprops" in tracks.annotator_manager.annotators
    assert "edges" in tracks.annotator_manager.annotators
    assert "tracks" in tracks.annotator_manager.annotators


def test_enable_features(graph_clean, segmentation_2d):
    """Test enable_features enables and computes multiple features efficiently."""
    tracks = Tracks(graph_clean, segmentation=segmentation_2d, ndim=3)
    manager = tracks.annotator_manager

    # Features should NOT be in tracks.features initially
    assert "pos" not in tracks.features
    assert "area" not in tracks.features
    assert "iou" not in tracks.features

    nodes = list(tracks.graph.nodes())
    edges = list(tracks.graph.edges())

    # Enable multiple features at once
    manager.enable_features(["pos", "area", "iou"])

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


def test_update_node(graph_2d_with_computed_features, segmentation_2d):
    """Test update routes node updates to regionprops annotator."""
    tracks = Tracks(
        graph_2d_with_computed_features,
        segmentation=segmentation_2d,
        ndim=3,
        existing_features=["pos", "area"],
    )
    manager = tracks.annotator_manager

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


def test_update_edge(graph_2d_with_computed_features, segmentation_2d):
    """Test update routes edge updates to edge annotator."""
    tracks = Tracks(
        graph_2d_with_computed_features,
        segmentation=segmentation_2d,
        ndim=3,
        existing_features=["pos", "iou"],
    )
    manager = tracks.annotator_manager

    edge_id = (1, 3)
    node_id = 3

    # Modify the segmentation by removing all but one pixel from node 3
    orig_pixels = tracks.get_pixels(node_id)
    assert orig_pixels is not None
    pixels_to_remove = tuple(orig_pixels[d][1:] for d in range(len(orig_pixels)))
    tracks.set_pixels(pixels_to_remove, 0)

    # Update the edge - should recompute IoU
    manager.update(edge_id)

    # Check that IoU was recomputed
    new_iou = tracks.graph.edges[edge_id]["iou"]
    assert new_iou == pytest.approx(0.0, abs=0.001)


def test_recompute_tracks(graph_clean, segmentation_2d):
    """Test recompute_tracks calls TrackAnnotator.compute."""
    tracks = SolutionTracks(graph_clean, segmentation=segmentation_2d, ndim=3)
    manager = tracks.annotator_manager

    # Enable tracklet_id
    manager.enable_features(["tracklet_id"])

    # tracklet_id should be set
    nodes = list(tracks.graph.nodes())
    assert tracks.graph.nodes[nodes[0]].get("tracklet_id") is not None

    # Recompute tracks
    manager.recompute_tracks()

    # tracklet_id should still exist
    assert tracks.graph.nodes[nodes[0]].get("tracklet_id") is not None


def test_get_available_features(graph_clean, segmentation_2d):
    """Test get_available_features returns all features from all annotators."""
    tracks = SolutionTracks(graph_clean, segmentation=segmentation_2d, ndim=3)
    manager = tracks.annotator_manager

    available = manager.get_available_features()

    # Should have features from all three annotators
    assert "pos" in available  # regionprops
    assert "area" in available  # regionprops
    assert "iou" in available  # edges
    assert "tracklet_id" in available  # tracks


def test_enable_disable_features(graph_clean, segmentation_2d):
    """Test enable_features, disable_features, and get_active_features."""
    tracks = Tracks(graph_clean, segmentation=segmentation_2d, ndim=3)
    manager = tracks.annotator_manager

    # Initially, no features are active or in FeatureDict
    active = manager.get_active_features()
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
    manager.enable_features(["pos", "area", "iou"])

    # All three should be active, in FeatureDict, and computed on graph
    active = manager.get_active_features()
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
    manager.disable_features(["area"])

    # Only pos and IoU should remain active
    active = manager.get_active_features()
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
    manager.disable_features(["pos", "iou"])
    active = manager.get_active_features()
    assert len(active) == 0
    assert "pos" not in tracks.features
    assert "iou" not in tracks.features


def test_get_feature_source(graph_clean, segmentation_2d):
    """Test get_feature_source returns correct annotator name."""
    tracks = SolutionTracks(graph_clean, segmentation=segmentation_2d, ndim=3)
    manager = tracks.annotator_manager

    assert manager.get_feature_source("pos") == "regionprops"
    assert manager.get_feature_source("area") == "regionprops"
    assert manager.get_feature_source("iou") == "edges"
    assert manager.get_feature_source("tracklet_id") == "tracks"
    assert manager.get_feature_source("nonexistent") is None


def test_enable_features_computes(graph_clean, segmentation_2d):
    """Test enable_features computes the features."""
    tracks = Tracks(graph_clean, segmentation=segmentation_2d, ndim=3)
    manager = tracks.annotator_manager

    # Enable (computes)
    manager.enable_features(["area", "pos"])

    # Features should be computed
    nodes = list(tracks.graph.nodes())
    assert tracks.graph.nodes[nodes[0]].get("area") is not None
    assert tracks.graph.nodes[nodes[0]].get("pos") is not None


def test_enable_nonexistent_feature(graph_clean, segmentation_2d):
    """Test enabling a nonexistent feature raises KeyError."""
    tracks = Tracks(graph_clean, segmentation=segmentation_2d, ndim=3)
    manager = tracks.annotator_manager

    with pytest.raises(KeyError, match="Feature 'nonexistent' not available"):
        manager.enable_features(["nonexistent"])


def test_disable_nonexistent_feature(graph_clean, segmentation_2d):
    """Test disabling a nonexistent feature raises KeyError."""
    tracks = Tracks(graph_clean, segmentation=segmentation_2d, ndim=3)
    manager = tracks.annotator_manager

    with pytest.raises(KeyError, match="Feature 'nonexistent' not available"):
        manager.disable_features(["nonexistent"])


def test_compute_strict_validation(graph_clean, segmentation_2d):
    """Test that compute() strictly validates feature keys."""
    tracks = Tracks(graph_clean, segmentation=segmentation_2d, ndim=3)
    manager = tracks.annotator_manager

    # Enable area first
    manager.enable_features(["area"])

    # Valid feature key should work
    manager.annotators["regionprops"].compute(["area"])

    # Invalid feature key should raise KeyError
    with pytest.raises(KeyError, match="Features not available or not enabled"):
        manager.annotators["regionprops"].compute(["nonexistent_feature"])

    # Disabled feature should raise KeyError
    manager.disable_features(["area"])
    with pytest.raises(KeyError, match="Features not available or not enabled"):
        manager.annotators["regionprops"].compute(["area"])

    # Mix of valid and invalid should raise KeyError
    manager.enable_features(["area"])
    with pytest.raises(KeyError, match="Features not available or not enabled"):
        manager.annotators["regionprops"].compute(["area", "nonexistent"])

    # None should still work (compute all enabled features)
    manager.annotators["regionprops"].compute()
