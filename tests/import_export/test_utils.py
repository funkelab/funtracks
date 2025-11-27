from funtracks.import_export._utils import rename_feature


def test_rename_feature_basic(get_tracks):
    """Test that rename_feature renames a feature in annotators and features dict."""
    tracks = get_tracks(ndim=3, with_seg=True, is_solution=False)

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
    tracks = get_tracks(ndim=3, with_seg=True, is_solution=True)

    original_pos_key = tracks.features.position_key
    new_key = "custom_position"

    rename_feature(tracks, original_pos_key, new_key)

    assert tracks.features.position_key == new_key
    assert new_key in tracks.features


def test_rename_feature_updates_tracklet_key(get_tracks):
    """Test that renaming tracklet feature updates tracklet_key in FeatureDict."""
    tracks = get_tracks(ndim=3, with_seg=True, is_solution=True)

    original_track_key = tracks.features.tracklet_key
    new_key = "custom_track"

    rename_feature(tracks, original_track_key, new_key)

    assert tracks.features.tracklet_key == new_key
    assert new_key in tracks.features
