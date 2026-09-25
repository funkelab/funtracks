from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from funtracks.exceptions import InvalidActionError

from ..actions._base import ActionGroup
from ._connect_utils import connection_pairs, find_conflicts
from .user_add_edge import UserAddEdge
from .user_delete_edge import UserDeleteEdge

if TYPE_CHECKING:
    from collections.abc import Iterable

    from funtracks.data_model import Tracks
    from funtracks.data_model.tracks import Node


class UserConnectNodes(ActionGroup):
    """Connect a selection of nodes, according to the division rules or linearly.

    The selected nodes are sorted by time and connected pairwise, so that they form
    one chain. Pairs that are connected already are left alone, so a partly connected
    selection is completed rather than rebuilt. Skip edges are allowed, as well as
    existing outgoing edges. If the source of a new edge has no successor yet, the
    tracklet id of the first (earliest) node propagates along the whole chain.

    A source that already has exactly one successor outside of the selection is
    handled according to ``linear``. By default (linear = False), a division is created
    there, and the regular division track id rules apply. With ``linear=True`` that
    existing edge is treated as a conflict instead, so that the selection ends up as one
    linear track without divisions. Outgoing edges of the last node of the chain are
    never touched, since no new edge is added there.

    See :class:`UserDisconnectNodes` for the inverse action.

    Args:
        tracks (Tracks): The tracks to connect the nodes in.
        nodes (Iterable[Node]): The nodes to connect. At least two nodes are required,
            and no two nodes may be in the same time point.
        linear (bool, optional): Whether to connect the nodes into a linear track,
            treating existing outgoing edges of the sources as conflicts instead of
            creating divisions. Defaults to False.
        force (bool, optional): Whether to force the action by removing conflicting
            edges. Defaults to False.
        _top_level (bool): If True, add this action to the history and emit refresh.
            Set to False when used as a sub-action inside a compound action.
            Defaults to True.

    Raises:
        InvalidActionError: If fewer than two distinct nodes are given, if a node is
            not in the solution graph, if two nodes share a time point, or if the
            nodes are connected already (all never forceable). If connecting would
            conflict with existing edges, a forceable InvalidActionError is raised
            unless ``force`` is True.
    """

    def __init__(
        self,
        tracks: Tracks,
        nodes: Iterable[Node],
        linear: bool = False,
        force: bool = False,
        _top_level: bool = True,
    ):
        super().__init__(tracks, actions=[])
        self.tracks: Tracks  # Narrow type from base class

        pairs = connection_pairs(tracks, nodes)
        missing = [edge for edge in pairs if not tracks.graph_solution.has_edge(*edge)]
        if not missing:
            raise InvalidActionError(
                "The selected nodes are connected already, there is nothing to connect."
            )

        conflicts = find_conflicts(self.tracks, pairs, linear=linear)
        if conflicts:
            if not force:
                listed = ", ".join(str(edge) for edge in conflicts)
                raise InvalidActionError(
                    f"Cannot connect the selected nodes: edge(s) {listed} conflict "
                    "with the requested connection.",
                    forceable=True,
                )
            warnings.warn(
                f"Removing edge(s) {conflicts} to connect the selected nodes.",
                stacklevel=2,
            )
            for edge in conflicts:
                self.actions.append(UserDeleteEdge(self.tracks, edge, _top_level=False))

        for edge in missing:
            self.actions.append(UserAddEdge(self.tracks, edge, _top_level=False))

        if _top_level:
            self.tracks.action_history.add_new_action(self)
            self.tracks.refresh.emit()

    @staticmethod
    def has_division_choice(tracks: Tracks, nodes: Iterable[Node]) -> bool:
        """Whether connecting these nodes linearly differs from connecting them with
        divisions, i.e. whether it is worth asking the user which one they want.

        Returns False for a selection that cannot be connected at all, or that has
        nothing left to connect.
        """

        try:
            pairs = connection_pairs(tracks, nodes)
        except InvalidActionError:
            return False
        if all(tracks.graph_solution.has_edge(*edge) for edge in pairs):
            return False  # nothing to connect
        with_divisions = find_conflicts(tracks, pairs, linear=False)
        linear = find_conflicts(tracks, pairs, linear=True)
        return set(with_divisions) != set(linear)
