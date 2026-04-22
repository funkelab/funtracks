from __future__ import annotations

from typing import TYPE_CHECKING

from ..actions._base import ActionGroup
from .user_update_node_attrs import UserUpdateNodeAttrs

if TYPE_CHECKING:
    from typing import Any

    from funtracks.data_model import SolutionTracks


class UserUpdateNodesAttrs(ActionGroup):
    """Update attributes on multiple nodes in a single action.

    Wraps individual UserUpdateNodeAttrs calls so that history and refresh
    are handled once for the entire batch, rather than per-node.

    Args:
        tracks: The tracks to update node attributes for.
        nodes: The node ids to update.
        attrs: A mapping from attribute name to new attribute values,
            applied to all nodes.
    """

    def __init__(
        self,
        tracks: SolutionTracks,
        nodes: list[int],
        attrs: dict[str, Any],
    ):
        super().__init__(tracks, actions=[])
        self.tracks: SolutionTracks  # Narrow type from base class
        for node in nodes:
            self.actions.append(
                UserUpdateNodeAttrs(tracks, node, attrs, _top_level=False)
            )

        self.tracks.action_history.add_new_action(self)
        self.tracks.refresh.emit()
