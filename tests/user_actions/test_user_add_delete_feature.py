import pytest

from funtracks.features import Feature
from funtracks.user_actions import (
    UserAddFeature,
    UserDeleteFeature,
    UserUpdateNodesAttrs,
)


def _group_feature() -> Feature:
    return Feature(
        feature_type="node",
        value_type="bool",
        num_values=1,
        display_name="group",
        default_value=False,
    )


@pytest.mark.parametrize("ndim", [3, 4])
@pytest.mark.parametrize("with_seg", [True, False])
class TestUserAddDeleteFeature:
    def test_add_undo_redo(self, get_tracks, ndim, with_seg):
        tracks = get_tracks(ndim=ndim, with_seg=with_seg, prefill_track_ids=True)
        action = UserAddFeature(tracks, "group", _group_feature())
        assert tracks.action_history.undo_stack[-1] is action
        assert "group" in tracks.features

        assert tracks.undo()
        assert "group" not in tracks.features
        assert "group" not in tracks.graph_full.node_attr_keys()

        assert tracks.redo()
        assert "group" in tracks.features
        assert tracks.get_node_attr(1, "group") is False

    def test_delete_undo_redo(self, get_tracks, ndim, with_seg):
        tracks = get_tracks(ndim=ndim, with_seg=with_seg, prefill_track_ids=True)
        UserAddFeature(tracks, "group", _group_feature())
        UserUpdateNodesAttrs(tracks, nodes=[1, 3], attrs={"group": [True, True]})

        UserDeleteFeature(tracks, "group")
        assert "group" not in tracks.features

        assert tracks.undo()
        assert "group" in tracks.features
        assert [tracks.get_node_attr(n, "group") for n in [1, 2, 3]] == [
            True,
            False,
            True,
        ]

        assert tracks.redo()
        assert "group" not in tracks.features

    def test_undo_membership_after_group_delete(self, get_tracks, ndim, with_seg):
        """Deleting a group, undoing that, then undoing an earlier membership change
        must work: the column is back by the time the membership undo runs."""
        tracks = get_tracks(ndim=ndim, with_seg=with_seg, prefill_track_ids=True)
        UserAddFeature(tracks, "group", _group_feature())
        UserUpdateNodesAttrs(tracks, nodes=[1, 3], attrs={"group": [True, True]})
        UserUpdateNodesAttrs(tracks, nodes=[2], attrs={"group": [True]})
        UserDeleteFeature(tracks, "group")

        assert tracks.undo()  # the delete
        assert tracks.undo()  # adding node 2
        assert [tracks.get_node_attr(n, "group") for n in [1, 2, 3]] == [
            True,
            False,
            True,
        ]
        assert tracks.undo()  # adding nodes 1 and 3
        assert [tracks.get_node_attr(n, "group") for n in [1, 2, 3]] == [
            False,
            False,
            False,
        ]
        assert tracks.undo()  # creating the group
        assert "group" not in tracks.features

    def test_toggles_do_not_add_history(self, get_tracks, ndim, with_seg):
        tracks = get_tracks(ndim=ndim, with_seg=with_seg, prefill_track_ids=True)
        if with_seg:
            tracks.disable_features(["area"])
            tracks.enable_features(["area"])
        assert tracks.action_history.undo_stack == []
