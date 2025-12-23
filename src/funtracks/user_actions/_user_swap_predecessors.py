from __future__ import annotations

from typing import TYPE_CHECKING

from funtracks.exceptions import InvalidActionError

from ..actions._base import ActionGroup
from .user_add_edge import UserAddEdge
from .user_delete_edge import UserDeleteEdge

if TYPE_CHECKING:
    from funtracks.data_model import SolutionTracks


class UserSwapPredecessors(ActionGroup):
    """Swap the predecessors (incoming edges) of two nodes.

    The nodes do not need to be at the same time point, but both predecessors
    must be earlier in time than both nodes for the swap to be valid.

    Args:
        tracks (SolutionTracks): The tracks to perform the swap on.
        nodes (tuple[Node, Node]): A tuple with two nodes.

    Raises:
        InvalidActionError: If not exactly two nodes are provided, if swapping
            would create invalid edges (predecessor not before node), if both
            nodes have the same predecessor, or if neither node has a predecessor.
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
        graph = tracks.graph

        # Find predecessors
        pred1 = next(graph.predecessors(node1), None)
        pred2 = next(graph.predecessors(node2), None)

        # Nothing to swap - raise errors for user feedback
        if pred1 is None and pred2 is None:
            raise InvalidActionError("Cannot swap: neither node has a predecessor.")
        if pred1 == pred2:
            raise InvalidActionError("Cannot swap: both nodes have the same predecessor.")

        # Validate that swapped edges would be valid (predecessors before nodes)
        time1 = tracks.get_time(node1)
        time2 = tracks.get_time(node2)
        if pred1 is not None:
            pred1_time = tracks.get_time(pred1)
            if pred1_time >= time2:
                raise InvalidActionError(
                    f"Cannot swap: predecessor of node {node1} (time {pred1_time}) "
                    f"is not before node {node2} (time {time2})."
                )
        if pred2 is not None:
            pred2_time = tracks.get_time(pred2)
            if pred2_time >= time1:
                raise InvalidActionError(
                    f"Cannot swap: predecessor of node {node2} (time {pred2_time}) "
                    f"is not before node {node1} (time {time1})."
                )

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
