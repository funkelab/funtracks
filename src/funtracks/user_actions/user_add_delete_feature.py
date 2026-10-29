from __future__ import annotations

from typing import TYPE_CHECKING

from ..actions._base import ActionGroup
from ..actions.add_delete_feature import AddFeature, DeleteFeature

if TYPE_CHECKING:
    from funtracks.data_model import Tracks
    from funtracks.features import Feature


class UserAddFeature(ActionGroup):
    """User action for adding a new node feature, e.g. a group.

    Wraps the basic AddFeature action and adds it to the action history, so creating
    the feature can be undone.
    """

    def __init__(
        self,
        tracks: Tracks,
        key: str,
        feature: Feature,
        _top_level: bool = True,
    ):
        """
        Args:
            tracks (Tracks): The tracks to add the feature to.
            key (str): The key of the new feature.
            feature (Feature): The feature metadata. Only node features are supported.
            _top_level (bool): If True, add this action to the history and emit
                refresh. Set to False when used as a sub-action inside a compound
                action. Defaults to True.
        """
        super().__init__(tracks, actions=[])
        self.tracks: Tracks  # Narrow type from base class
        self.actions.append(AddFeature(tracks, key, feature))

        if _top_level:
            self.tracks.action_history.add_new_action(self)
            self.tracks.refresh.emit()


class UserDeleteFeature(ActionGroup):
    """User action for deleting a user-created node feature, e.g. a group.

    Wraps the basic DeleteFeature action and adds it to the action history, so the
    feature and its values can be restored by undo.
    """

    def __init__(
        self,
        tracks: Tracks,
        key: str,
        _top_level: bool = True,
    ):
        """
        Args:
            tracks (Tracks): The tracks to delete the feature from.
            key (str): The key of the feature to delete.
            _top_level (bool): If True, add this action to the history and emit
                refresh. Set to False when used as a sub-action inside a compound
                action. Defaults to True.
        """
        super().__init__(tracks, actions=[])
        self.tracks: Tracks  # Narrow type from base class
        self.actions.append(DeleteFeature(tracks, key))

        if _top_level:
            self.tracks.action_history.add_new_action(self)
            self.tracks.refresh.emit()
