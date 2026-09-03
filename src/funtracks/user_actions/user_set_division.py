from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from funtracks.exceptions import InvalidActionError

from ..actions._base import ActionGroup
from .user_add_edge import UserAddEdge
from .user_delete_edge import UserDeleteEdge

if TYPE_CHECKING:
    from funtracks.data_model import Tracks


class UserSetDivision(ActionGroup):
    """Toggle a division between a parent node and two child nodes.

    Exactly three nodes must be provided, and exactly one of them must be earlier
    in time than the other two. That node is the parent, the other two are the
    children.

    If both division edges already exist, they are broken: the parent is left as a
    track endpoint and both children become the start of a new track. Otherwise the
    division is made: any conflicting edges (other out edges of the parent, other in
    edges of the children) are deleted first, and the missing parent -> child
    edges are added.

    Since making and breaking a division are each other's inverse, undoing this
    action restores the previous state either way.

    Args:
        tracks (Tracks): The tracks to set the division on.
        nodes (tuple[int, int, int]): The three nodes of the division trio, in any
            order.
        _top_level (bool): If True, add this action to the history and emit refresh.
            Set to False when used as a sub-action inside a compound action.
            Defaults to True.

    Raises:
        InvalidActionError: If not exactly three distinct nodes are provided, or if
            there is no unique earliest node among them.
    """

    def __init__(
        self,
        tracks: Tracks,
        nodes: tuple[int, int, int],
        _top_level: bool = True,
    ):
        super().__init__(tracks, actions=[])
        self.tracks: Tracks  # Narrow type from base class

        if len(set(nodes)) != 3:
            raise InvalidActionError(
                "A division requires exactly 3 distinct nodes, got "
                f"{len(set(nodes))} ({list(nodes)})."
            )

        times = {node: tracks.get_time(node) for node in nodes}
        earliest_time = min(times.values())
        parents = [node for node, time in times.items() if time == earliest_time]
        if len(parents) != 1:
            raise InvalidActionError(
                "A division requires exactly one node to be earlier in time than the "
                f"other two, but nodes {parents} all share the earliest time "
                f"{earliest_time}."
            )

        self.parent = parents[0]
        self.children = [node for node in nodes if node != self.parent]

        # A division exists already if the parent is connected to both children
        self.breaking = all(
            tracks.graph_solution.has_edge(self.parent, child) for child in self.children
        )
        if self.breaking:
            self._break_division()
        else:
            self._make_division()

        if _top_level:
            self.tracks.action_history.add_new_action(self)
            self.tracks.refresh.emit()

    def _break_division(self) -> None:
        """Delete both parent -> child edges."""

        for child in self.children:
            self.actions.append(
                UserDeleteEdge(self.tracks, (self.parent, child), _top_level=False)
            )

    def _make_division(self) -> None:
        """Delete conflicting edges, then add the missing parent -> child edges."""

        # Other children of the parent would make it a triple division: remove them
        for successor in self.tracks.successors(self.parent):
            if successor not in self.children:
                self._delete_conflicting_edge((self.parent, successor))

        for child in self.children:
            if self.tracks.graph_solution.has_edge(self.parent, child):
                continue
            # Another parent of the child would make a merge: remove it
            for predecessor in self.tracks.predecessors(child):
                self._delete_conflicting_edge((predecessor, child))
            self.actions.append(
                UserAddEdge(self.tracks, (self.parent, child), _top_level=False)
            )

    def _delete_conflicting_edge(self, edge: tuple[int, int]) -> None:
        """Delete an edge that prevents the division from being made."""

        warnings.warn(
            f"Removing conflicting edge {edge} to make the division.",
            stacklevel=3,
        )
        self.actions.append(UserDeleteEdge(self.tracks, edge, _top_level=False))
