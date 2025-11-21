import pytest

from funtracks.import_export import register_computed_features


def test_register_computed_features_recompute(get_tracks):
    """Test that features with recompute=True are computed and renamed correctly."""
    # Create tracks with segmentation so we can compute features
    tracks = get_tracks(ndim=3, with_seg=True, is_solution=False)

    # Request area feature to be recomputed with custom name
    computed_features = [
        {
            "prop_name": "my_area",
            "feature": "Area",
            "recompute": True,
        }
    ]

    register_computed_features(tracks, computed_features)

    # Check that feature was registered with custom name
    assert "my_area" in tracks.features
    assert "area" not in tracks.features  # Original name should not be present

    # Check that values were computed for all nodes
    for _, data in tracks.graph.nodes(data=True):
        assert "my_area" in data
        assert data["my_area"] == data["area"]


def test_register_computed_features_load_without_recompute(get_tracks):
    """Test that features with recompute=False are loaded from graph and renamed."""
    # Create tracks with pre-computed area values
    tracks = get_tracks(ndim=3, with_seg=True, is_solution=False)

    # Simulate imported data with custom key "loaded_area" instead of default "area"
    # Store original values and rename attributes in the graph
    original_areas = {}
    for node_id, data in tracks.graph.nodes(data=True):
        original_areas[node_id] = data["area"]
        data["loaded_area"] = data.pop("area")

    # Request area feature to be loaded (not recomputed) with custom name
    computed_features = [
        {
            "prop_name": "loaded_area",
            "feature": "Area",
            "recompute": False,
        }
    ]

    register_computed_features(tracks, computed_features)

    # Check that feature was registered with custom name
    assert "loaded_area" in tracks.features
    assert "area" not in tracks.features

    # Check that original values were preserved (not recomputed)
    for node_id, data in tracks.graph.nodes(data=True):
        assert "loaded_area" in data
        assert data["loaded_area"] == original_areas[node_id]


@pytest.mark.parametrize("recompute", [True, False])
def test_invalid_feature_error(get_tracks, recompute):
    """Test that requesting an unknown feature raises ValueError."""
    tracks = get_tracks(ndim=3, with_seg=True, is_solution=False)

    computed_features = [
        {
            "prop_name": "my_feature",
            "feature": "NonexistentFeature",
            "recompute": recompute,
        }
    ]

    with pytest.raises(
        ValueError,
        match="Requested activation of feature .* but no such feature found",
    ):
        register_computed_features(tracks, computed_features)


@pytest.mark.parametrize("recompute", [True, False])
def test_missing_segmentation_error(get_tracks, recompute):
    """Test that regionprops features without segmentation raise error."""
    # Create tracks WITHOUT segmentation
    tracks = get_tracks(ndim=3, with_seg=False, is_solution=False)

    computed_features = [
        {
            "prop_name": "my_area",
            "feature": "Area",
            "recompute": recompute,
        }
    ]

    with pytest.raises(
        ValueError,
        match="Requested activation of feature .* but no such feature found .* "
        "Perhaps you need to provide a segmentation?",
    ):
        register_computed_features(tracks, computed_features)


def test_special_key_updates(get_tracks):
    """Test that renaming special features updates their keys in FeatureDict."""
    tracks = get_tracks(ndim=3, with_seg=True, is_solution=True)

    # Verify initial keys
    assert tracks.features.position_key == "pos"
    assert tracks.features.tracklet_key == "track_id"

    # Rename both position and tracklet features
    computed_features = [
        {
            "prop_name": "custom_position",
            "feature": ("y", "x"),
            "recompute": False,
        },
        {
            "prop_name": "custom_track",
            "feature": "Tracklet ID",
            "recompute": False,
        },
    ]

    register_computed_features(tracks, computed_features)

    # Check that both special keys were updated
    assert tracks.features.position_key == "custom_position"
    assert tracks.features.tracklet_key == "custom_track"
    assert "custom_position" in tracks.features
    assert "custom_track" in tracks.features
