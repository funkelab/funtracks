from __future__ import annotations

from typing import TYPE_CHECKING

from funtracks.exceptions import InvalidActionError

from ..actions._base import ActionGroup
from ._connect_utils import validate_nodes
from .user_delete_edge import UserDeleteEdge

if TYPE_CHECKING:
    from collections.abc import Iterable

    from funtracks.data_model import Tracks
    from funtracks.data_model.tracks import Node


class UserDisconnectNodes(ActionGroup):
    """Break the edges between a selection of nodes, the inverse of connecting them.

    Every existing edge between two of the selected nodes is removed, including skip
    edges, and each fragment that comes loose gets its own tracklet id.

    Args:
        tracks (Tracks): The tracks to disconnect the nodes in.
        nodes (Iterable[Node]): The nodes to disconnect. At least two nodes are
            required.
        _top_level (bool): If True, add this action to the history and emit refresh.
            Set to False when used as a sub-action inside a compound action.
            Defaults to True.

    Raises:
        InvalidActionError: If fewer than two distinct nodes are given, if a node is
            not in the solution graph, or if none of the selected nodes are connected
            to each other. Never forceable.
    """

    def __init__(
        self,
        tracks: Tracks,
        nodes: Iterable[Node],
        _top_level: bool = True,
    ):
        super().__init__(tracks, actions=[])
        self.tracks: Tracks  # Narrow type from base class

        nodes = validate_nodes(tracks, nodes)
        selected = set(nodes)
        existing = [
            (node, succ)
            for node in nodes
            for succ in tracks.successors(node)
            if succ in selected
        ]
        if not existing:
            raise InvalidActionError(
                "The selected nodes are not connected to each other, there is nothing "
                "to disconnect."
            )

        for edge in existing:
            self.actions.append(UserDeleteEdge(self.tracks, edge, _top_level=False))

        if _top_level:
            self.tracks.action_history.add_new_action(self)
            self.tracks.refresh.emit()
