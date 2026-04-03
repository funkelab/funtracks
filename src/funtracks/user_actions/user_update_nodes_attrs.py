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
        attrs: Either a single dict applied to all nodes, or a list of dicts
            with one entry per node (must match the length of nodes).
    """

    def __init__(
        self,
        tracks: SolutionTracks,
        nodes: list[int],
        attrs: dict[str, Any] | list[dict[str, Any]],
    ):
        super().__init__(tracks, actions=[])
        self.tracks: SolutionTracks  # Narrow type from base class
        if isinstance(attrs, list):
            if len(attrs) != len(nodes):
                raise ValueError(
                    f"attrs list length ({len(attrs)}) must match "
                    f"nodes length ({len(nodes)})"
                )
            per_node_attrs = attrs
        else:
            per_node_attrs = [attrs] * len(nodes)
        for node, node_attrs in zip(nodes, per_node_attrs, strict=True):
            self.actions.append(
                UserUpdateNodeAttrs(tracks, node, node_attrs, _top_level=False)
            )

        self.tracks.action_history.add_new_action(self)
        self.tracks.refresh.emit()
