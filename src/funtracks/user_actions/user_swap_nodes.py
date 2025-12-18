from __future__ import annotations

from typing import TYPE_CHECKING

from funtracks.user_actions import UserAddEdge, UserDeleteEdge

from ..actions._base import ActionGroup

if TYPE_CHECKING:
    from funtracks.data_model import SolutionTracks


class UserSwapNodes(ActionGroup):
    """Swap the incoming edges of two horizontal nodes.

    Args:
        nodes (tuple[Node, Node]): A tuple with two nodes at the same time point.
    """

    def __init__(
        self,
        tracks: SolutionTracks,
        nodes: tuple[int, int],
    ):
        super().__init__(tracks, actions=[])
        self.tracks: SolutionTracks  # narrow type

        node1, node2 = nodes
        graph = tracks.graph

        # Find predecessors
        pred1 = next(graph.predecessors(node1), None)
        pred2 = next(graph.predecessors(node2), None)

        if pred1 is None and pred2 is None:
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
