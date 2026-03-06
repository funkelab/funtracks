from __future__ import annotations

from typing import TYPE_CHECKING

from ..actions._base import ActionGroup
from ..actions.update_node_attrs import UpdateNodeAttrs

if TYPE_CHECKING:
    from typing import Any

    from funtracks.data_model import SolutionTracks


class UserUpdateNodeAttrs(ActionGroup):
    """User action for updating node attributes.

    This wraps the basic UpdateNodeAttrs action and adds it to the action history
    for undo/redo functionality. Protected attributes (time, annotator-managed features)
    cannot be updated.
    """

    def __init__(
        self,
        tracks: SolutionTracks,
        node: int,
        attrs: dict[str, Any],
    ):
        """
        Args:
            tracks (SolutionTracks): The tracks to update the node attributes for
            node (int): The node to update the attributes for
            attrs (dict[str, Any]): A mapping from attribute name to new attribute
                values for the given node.

        Raises:
            ValueError: If a protected attribute is in the given attribute mapping.
        """
        super().__init__(tracks, actions=[])
        self.tracks: SolutionTracks  # Narrow type from base class

        # Call the basic UpdateNodeAttrs action
        self.actions.append(UpdateNodeAttrs(tracks, node, attrs))

        self.tracks.action_history.add_new_action(self)
        self.tracks.refresh.emit()
