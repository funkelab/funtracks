from __future__ import annotations

from typing import TYPE_CHECKING

from funtracks.exceptions import InvalidActionError

from ..actions._base import ActionGroup
from .user_add_edge import UserAddEdge
from .user_delete_edge import UserDeleteEdge

if TYPE_CHECKING:
    from funtracks.data_model import SolutionTracks


class UserSwapPredecessors(ActionGroup):
    """Swap the predecessors (incoming edges) of two nodes at the same time point.

    Args:
        tracks (SolutionTracks): The tracks to perform the swap on.
        nodes (tuple[Node, Node]): A tuple with two nodes at the same time point.

    Raises:
        InvalidActionError: If the nodes are not at the same time point, or if
            not exactly two nodes are provided.
    """

    def __init__(
        self,
        tracks: SolutionTracks,
        nodes: tuple[int, int],
    ):
        super().__init__(tracks, actions=[])
        self.tracks: SolutionTracks  # narrow type

        if len(nodes) != 2:
            raise InvalidActionError("You can only swap a pair of two nodes.")

        node1, node2 = nodes

        # Validate that both nodes have the same time point
        time1 = tracks.get_time(node1)
        time2 = tracks.get_time(node2)
        if time1 != time2:
            raise InvalidActionError("Both nodes must have the same time point to swap.")

        graph = tracks.graph

        # Find predecessors
        pred1 = next(graph.predecessors(node1), None)
        pred2 = next(graph.predecessors(node2), None)

        # No-op cases: nothing to swap
        if pred1 is None and pred2 is None:
            return
        if pred1 == pred2:
            # Same predecessor - swapping would result in identical graph
            return

        # Break existing edges
        if pred1 is not None:
            self.actions.append(UserDeleteEdge(tracks, (pred1, node1)))
        if pred2 is not None:
            self.actions.append(UserDeleteEdge(tracks, (pred2, node2)))

        # Create swapped edges
        if pred1 is not None:
            self.actions.append(UserAddEdge(tracks, (pred1, node2), force=False))
        if pred2 is not None:
            self.actions.append(UserAddEdge(tracks, (pred2, node1), force=False))
