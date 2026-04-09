import numpy as np
import polars as pl
import pytest

from funtracks.user_actions import UserUpdateNodesAttrs


@pytest.mark.parametrize("ndim", [3, 4])
@pytest.mark.parametrize("with_seg", [True, False])
class TestUserUpdateNodesAttrs:
    def test_user_update_nodes_attrs(self, get_tracks, ndim, with_seg):
        """Test basic bulk node attribute update functionality."""
        tracks = get_tracks(ndim=ndim, with_seg=with_seg, is_solution=True)

        tracks.graph.add_node_attr_key("label", default_value=None, dtype=pl.Object)
        tracks.graph.add_node_attr_key("confidence", default_value=0, dtype=pl.Float64)

        attrs = {"label": ["my_label", "my_label"], "confidence": [0.95, 0.95]}
        UserUpdateNodesAttrs(tracks, nodes=[1, 2], attrs=attrs)

        for node in [1, 2]:
            assert tracks.get_node_attr(node, "label") == "my_label"
            assert tracks.get_node_attr(node, "confidence") == 0.95

    def test_single_history_entry(self, get_tracks, ndim, with_seg):
        """Updating multiple nodes creates only one history entry."""
        tracks = get_tracks(ndim=ndim, with_seg=with_seg, is_solution=True)
        tracks.graph.add_node_attr_key("label", default_value=None, dtype=pl.Object)

        action = UserUpdateNodesAttrs(
            tracks, nodes=[1, 2, 3], attrs={"label": ["x", "x", "x"]}
        )

        assert len(tracks.action_history.undo_stack) == 1
        assert tracks.action_history.undo_stack[-1] is action

    def test_undo_redo(self, get_tracks, ndim, with_seg):
        """Undo restores all nodes' attrs to defaults; redo re-applies them."""
        tracks = get_tracks(ndim=ndim, with_seg=with_seg, is_solution=True)
        tracks.graph.add_node_attr_key("score", default_value=0, dtype=pl.Float64)

        action = UserUpdateNodesAttrs(tracks, nodes=[1, 2], attrs={"score": [0.9, 0.9]})

        for node in [1, 2]:
            assert tracks.get_node_attr(node, "score") == 0.9

        inverse = action.inverse()

        for node in [1, 2]:
            assert tracks.get_node_attr(node, "score") == 0

        inverse.inverse()

        for node in [1, 2]:
            assert tracks.get_node_attr(node, "score") == 0.9

    def test_per_node_attrs(self, get_tracks, ndim, with_seg):
        """Test bulk update with different values per node."""
        tracks = get_tracks(ndim=ndim, with_seg=with_seg, is_solution=True)
        tracks.graph.add_node_attr_key("score", default_value=0, dtype=pl.Float64)

        UserUpdateNodesAttrs(tracks, nodes=[1, 2], attrs={"score": [0.1, 0.9]})

        assert tracks.get_node_attr(1, "score") == 0.1
        assert tracks.get_node_attr(2, "score") == 0.9

    def test_array_attr(self, get_tracks, ndim, with_seg):
        """Test bulk update with array-valued attributes."""
        tracks = get_tracks(ndim=ndim, with_seg=with_seg, is_solution=True)
        spatial_dims = ndim - 1

        tracks.graph.add_node_attr_key(
            "custom_pos", default_value=None, dtype=pl.Array(pl.Float64, spatial_dims)
        )

        positions = [[float(i)] * spatial_dims for i in range(2)]
        UserUpdateNodesAttrs(tracks, nodes=[1, 2], attrs={"custom_pos": positions})

        assert np.all(np.asarray(tracks.get_node_attr(1, "custom_pos")) == positions[0])
        assert np.all(np.asarray(tracks.get_node_attr(2, "custom_pos")) == positions[1])

    def test_values_not_list_raises(self, get_tracks, ndim, with_seg):
        """Non-list values raise ValueError."""
        tracks = get_tracks(ndim=ndim, with_seg=with_seg, is_solution=True)

        with pytest.raises(ValueError, match="must be a list"):
            UserUpdateNodesAttrs(tracks, nodes=[1, 2], attrs={"score": 0.9})

    def test_values_length_mismatch_raises(self, get_tracks, ndim, with_seg):
        """List length not matching nodes length raises ValueError."""
        tracks = get_tracks(ndim=ndim, with_seg=with_seg, is_solution=True)

        with pytest.raises(ValueError, match="length"):
            UserUpdateNodesAttrs(tracks, nodes=[1, 2], attrs={"score": [0.1]})

    def test_protected_attr_raises(self, get_tracks, ndim, with_seg):
        """Passing a protected attribute raises ValueError."""
        tracks = get_tracks(ndim=ndim, with_seg=with_seg, is_solution=True)
        time_key = tracks.features.time_key

        with pytest.raises(ValueError, match="Cannot update attribute"):
            UserUpdateNodesAttrs(tracks, nodes=[1, 2], attrs={time_key: [0, 1]})
