from __future__ import annotations

from typing import TYPE_CHECKING

from funtracks.exceptions import InvalidActionError

from ..actions._base import ActionGroup
from ..actions.add_delete_edge import DeleteEdge
from ..actions.update_track_id import UpdateTrackIDs

if TYPE_CHECKING:
    from funtracks.data_model import Tracks


class UserDeleteEdge(ActionGroup):
    def __init__(
        self,
        tracks: Tracks,
        edge: tuple[int, int],
        _top_level: bool = True,
    ):
        """
        Args:
            tracks (Tracks): The tracks to delete the edge from.
            edge (tuple[int, int]): The edge to delete.
            _top_level (bool): If True, add this action to the history and emit
                refresh. Set to False when used as a sub-action inside a compound
                action. Defaults to True.

        Raises:
            InvalidActionError: If the edge does not exist in the graph.
        """
        super().__init__(tracks, actions=[])
        self.tracks: Tracks  # Narrow type from base class
        if not self.tracks.graph_solution.has_edge(*edge):
            raise InvalidActionError(f"Edge {edge} not in solution, can't remove")

        # The out degree the source will have once the edge is gone. Check it before
        # applying anything, so that a failing check leaves the tracks unchanged.
        out_degree = len(self.tracks.successors(edge[0])) - 1
        if out_degree > 1:
            raise InvalidActionError(
                f"Expected degree of 0 or 1 after removing edge, got {out_degree}"
            )

        self.actions.append(DeleteEdge(tracks, edge))
        if out_degree == 0:  # removed a normal (non division) edge
            # orphaned segment gets new track id and new lineage id
            new_track_id = self.tracks.get_next_track_id()
            new_lineage_id = self.tracks.get_next_lineage_id()
            self.actions.append(
                UpdateTrackIDs(self.tracks, edge[1], new_track_id, new_lineage_id)
            )
        else:  # removed a division edge
            # sibling gets parent's track id (lineage stays the same)
            sibling = next(iter(self.tracks.graph_solution.successors(edge[0])))
            new_track_id = self.tracks.get_track_id(edge[0])
            self.actions.append(UpdateTrackIDs(self.tracks, sibling, new_track_id))
            # orphaned child gets a new lineage id (now a separate component)
            self.actions.append(
                UpdateTrackIDs(
                    self.tracks, edge[1], lineage_id=self.tracks.get_next_lineage_id()
                )
            )

        if _top_level:
            self.tracks.action_history.add_new_action(self)
            self.tracks.refresh.emit()
