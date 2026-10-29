from __future__ import annotations

from typing import TYPE_CHECKING

from funtracks.exceptions import InvalidActionError

from ..actions._base import ActionGroup
from .user_delete_node import UserDeleteNode

if TYPE_CHECKING:
    from funtracks.data_model import Tracks


class UserDeleteNodes(ActionGroup):
    """Delete multiple nodes in a single action.

    Wraps individual UserDeleteNode calls so that history and refresh
    are handled once for the entire batch, rather than per-node.

    Args:
        tracks: The tracks to delete nodes from.
        nodes: The node ids to delete.
        _top_level: If True, add this action to the history and emit the refresh
            signal. Set to False when this action is part of a bigger action group,
            so that the whole group is undone in one step. Defaults to True.
    """

    def __init__(
        self,
        tracks: Tracks,
        nodes: list[int],
        _top_level: bool = True,
    ):
        super().__init__(tracks, actions=[])
        self.tracks: Tracks  # Narrow type from base class
        # Check every node before deleting any, so that a failing check leaves the
        # tracks unchanged.
        nodes = list(dict.fromkeys(nodes))  # de-duplicate, keep order
        for node in nodes:
            if not tracks.graph_solution.has_node(node):
                raise InvalidActionError(f"Node {node} not in solution, can't remove")
        for node in nodes:
            self.actions.append(UserDeleteNode(tracks, node, _top_level=False))

        if _top_level:
            self.tracks.action_history.add_new_action(self)
            self.tracks.refresh.emit()
