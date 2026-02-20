import pytest

from funtracks.user_actions import UserUpdateNodeAttrs


@pytest.mark.parametrize("ndim", [3, 4])
@pytest.mark.parametrize("with_seg", [True, False])
class TestUserUpdateNodeAttrs:
    def test_user_update_node_attrs(self, get_tracks, ndim, with_seg):
        """Test basic node attribute update functionality."""
        tracks = get_tracks(ndim=ndim, with_seg=with_seg, is_solution=True)

        # Add custom attributes to update
        custom_attrs = {"label": "my_label", "confidence": 0.95, "validated": True}

        # Update node 1 with custom attributes
        action = UserUpdateNodeAttrs(tracks, node=1, attrs=custom_attrs)

        # Verify attributes were updated
        assert tracks.get_node_attr(1, "label") == "my_label"
        assert tracks.get_node_attr(1, "confidence") == 0.95
        assert tracks.get_node_attr(1, "validated") is True

        # Verify action was added to history
        assert len(tracks.action_history.undo_stack) == 1

        # Test undo
        inverse = action.inverse()
        # After undo, custom attributes should be removed or set to None
        # (depending on whether they existed before)
        assert tracks.get_node_attr(1, "label") is None
        assert tracks.get_node_attr(1, "confidence") is None
        assert tracks.get_node_attr(1, "validated") is None

        # Test redo
        inverse.inverse()
        assert tracks.get_node_attr(1, "label") == "my_label"
        assert tracks.get_node_attr(1, "confidence") == 0.95
        assert tracks.get_node_attr(1, "validated") is True

    def test_user_update_existing_attrs(self, get_tracks, ndim, with_seg):
        """Test updating attributes that already exist."""
        tracks = get_tracks(ndim=ndim, with_seg=with_seg, is_solution=True)

        # Set initial custom attributes
        tracks._set_node_attr(1, "label", "old_label")
        tracks._set_node_attr(1, "score", 0.5)

        # Update to new values
        new_attrs = {"label": "new_label", "score": 0.9}
        action = UserUpdateNodeAttrs(tracks, node=1, attrs=new_attrs)

        # Verify new values
        assert tracks.get_node_attr(1, "label") == "new_label"
        assert tracks.get_node_attr(1, "score") == 0.9

        # Test undo restores old values
        action.inverse()
        assert tracks.get_node_attr(1, "label") == "old_label"
        assert tracks.get_node_attr(1, "score") == 0.5

    def test_protected_time_attr(self, get_tracks, ndim, with_seg):
        """Test that time attribute cannot be updated."""
        tracks = get_tracks(ndim=ndim, with_seg=with_seg, is_solution=True)
        time_key = tracks.features.time_key

        with pytest.raises(ValueError, match="Cannot update attribute"):
            UserUpdateNodeAttrs(tracks, node=1, attrs={time_key: 999})

    def test_protected_track_id_attr(self, get_tracks, ndim, with_seg):
        """Test that track_id attribute cannot be updated."""
        tracks = get_tracks(ndim=ndim, with_seg=with_seg, is_solution=True)

        with pytest.raises(ValueError, match="Cannot update attribute"):
            UserUpdateNodeAttrs(tracks, node=1, attrs={"track_id": 999})

    def test_protected_area_attr(self, get_tracks, ndim, with_seg):
        """Test that area attribute (managed by annotator) cannot be updated."""
        tracks = get_tracks(ndim=ndim, with_seg=with_seg, is_solution=True)

        if with_seg:  # area only exists when segmentation is present
            with pytest.raises(ValueError, match="Cannot update attribute"):
                UserUpdateNodeAttrs(tracks, node=1, attrs={"area": 999})

    def test_protected_pos_attr(self, get_tracks, ndim, with_seg):
        """Test that position attribute (managed by annotator) cannot be updated."""
        tracks = get_tracks(ndim=ndim, with_seg=with_seg, is_solution=True)

        if with_seg:  # pos is managed by RegionpropsAnnotator when seg exists
            with pytest.raises(ValueError, match="Cannot update attribute"):
                UserUpdateNodeAttrs(tracks, node=1, attrs={"pos": [0, 0]})

    def test_action_history_integration(self, get_tracks, ndim, with_seg):
        """Test that action integrates properly with action history."""
        tracks = get_tracks(ndim=ndim, with_seg=with_seg, is_solution=True)

        # Initially empty
        assert len(tracks.action_history.undo_stack) == 0
        assert len(tracks.action_history.redo_stack) == 0

        # Add first update
        UserUpdateNodeAttrs(tracks, node=1, attrs={"label": "first"})
        assert len(tracks.action_history.undo_stack) == 1
        assert len(tracks.action_history.redo_stack) == 0
        assert tracks.get_node_attr(1, "label") == "first"

        # Add second update
        UserUpdateNodeAttrs(tracks, node=2, attrs={"label": "second"})
        assert len(tracks.action_history.undo_stack) == 2
        assert len(tracks.action_history.redo_stack) == 0
        assert tracks.get_node_attr(2, "label") == "second"

        # Undo second update
        tracks.action_history.undo()
        assert len(tracks.action_history.undo_stack) == 2  # Actions stay in undo_stack
        assert len(tracks.action_history.redo_stack) == 1  # Inverse added to redo_stack
        assert tracks.get_node_attr(2, "label") is None
        assert tracks.get_node_attr(1, "label") == "first"

        # Undo first update
        tracks.action_history.undo()
        assert len(tracks.action_history.undo_stack) == 2
        assert len(tracks.action_history.redo_stack) == 2
        assert tracks.get_node_attr(1, "label") is None

        # Redo first update
        tracks.action_history.redo()
        assert len(tracks.action_history.undo_stack) == 2
        assert len(tracks.action_history.redo_stack) == 1
        assert tracks.get_node_attr(1, "label") == "first"

        # Redo second update
        tracks.action_history.redo()
        assert len(tracks.action_history.undo_stack) == 2
        assert len(tracks.action_history.redo_stack) == 0
        assert tracks.get_node_attr(2, "label") == "second"
