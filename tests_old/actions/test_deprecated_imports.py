import pytest


def test_deprecated_update_track_id_import():
    """Test that importing UpdateTrackID emits a deprecation warning."""
    with pytest.warns(DeprecationWarning, match="UpdateTrackID is deprecated"):
        from funtracks.actions import UpdateTrackID  # noqa: F401


def test_deprecated_update_track_id_is_alias():
    """Test that UpdateTrackID is an alias for UpdateTrackIDs."""
    with pytest.warns(DeprecationWarning):
        from funtracks.actions import UpdateTrackID

    from funtracks.actions import UpdateTrackIDs

    assert UpdateTrackID is UpdateTrackIDs


def test_invalid_import_raises_import_error():
    """Test that importing a non-existent name raises ImportError."""
    with pytest.raises(ImportError, match="cannot import name 'NonExistent'"):
        from funtracks.actions import NonExistent  # noqa: F401
