import pytest

from funtracks.user_actions import UserUpdateNodesAttrs


@pytest.mark.parametrize("ndim", [3, 4])
@pytest.mark.parametrize("with_seg", [True, False])
class TestUserUpdateNodesAttrs:
    def test_user_update_nodes_attrs(self, get_tracks, ndim, with_seg):
        """Test basic bulk node attribute update functionality."""
        tracks = get_tracks(ndim=ndim, with_seg=with_seg, is_solution=True)

        attrs = {"label": "my_label", "confidence": 0.95}
        UserUpdateNodesAttrs(tracks, nodes=[1, 2], attrs=attrs)

        for node in [1, 2]:
            assert tracks.get_node_attr(node, "label") == "my_label"
            assert tracks.get_node_attr(node, "confidence") == 0.95

    def test_single_history_entry(self, get_tracks, ndim, with_seg):
        """Updating multiple nodes creates only one history entry."""
        tracks = get_tracks(ndim=ndim, with_seg=with_seg, is_solution=True)

        action = UserUpdateNodesAttrs(tracks, nodes=[1, 2, 3], attrs={"label": "x"})

        assert len(tracks.action_history.undo_stack) == 1
        assert tracks.action_history.undo_stack[-1] is action

    def test_undo_redo(self, get_tracks, ndim, with_seg):
        """Undo restores all nodes' attrs; redo re-applies them."""
        tracks = get_tracks(ndim=ndim, with_seg=with_seg, is_solution=True)

        action = UserUpdateNodesAttrs(tracks, nodes=[1, 2], attrs={"score": 0.9})

        for node in [1, 2]:
            assert tracks.get_node_attr(node, "score") == 0.9

        inverse = action.inverse()

        for node in [1, 2]:
            assert tracks.get_node_attr(node, "score") is None

        inverse.inverse()

        for node in [1, 2]:
            assert tracks.get_node_attr(node, "score") == 0.9

    def test_protected_attr_raises(self, get_tracks, ndim, with_seg):
        """Passing a protected attribute raises ValueError."""
        tracks = get_tracks(ndim=ndim, with_seg=with_seg, is_solution=True)
        time_key = tracks.features.time_key

        with pytest.raises(ValueError, match="Cannot update attribute"):
            UserUpdateNodesAttrs(tracks, nodes=[1, 2], attrs={time_key: 999})
