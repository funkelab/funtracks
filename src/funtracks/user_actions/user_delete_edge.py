from __future__ import annotations

from typing import TYPE_CHECKING

from funtracks.exceptions import InvalidActionError

from ..actions._base import ActionGroup
from ..actions.add_delete_edge import DeleteEdge
from ..actions.update_track_id import UpdateTrackIDs

if TYPE_CHECKING:
    from funtracks.data_model import SolutionTracks


class UserDeleteEdge(ActionGroup):
    def __init__(
        self,
        tracks: SolutionTracks,
        edge: tuple[int, int],
        _top_level: bool = True,
    ):
        """
        Args:
            tracks (SolutionTracks): The tracks to delete the edge from.
            edge (tuple[int, int]): The edge to delete.
            _top_level (bool): If True, add this action to the history and emit
                refresh. Set to False when used as a sub-action inside a compound
                action. Defaults to True.

        Raises:
            InvalidActionError: If the edge does not exist in the graph.
        """
        super().__init__(tracks, actions=[])
        self.tracks: SolutionTracks  # Narrow type from base class
        if not self.tracks.graph.has_edge(*edge):
            raise InvalidActionError(f"Edge {edge} not in solution, can't remove")

        self.actions.append(DeleteEdge(tracks, edge))
        out_degree = self.tracks.graph.out_degree(edge[0])
        if out_degree == 0:  # removed a normal (non division) edge
            # orphaned segment gets new track id and new lineage id
            new_track_id = self.tracks.get_next_track_id()
            new_lineage_id = self.tracks.get_next_lineage_id()
            self.actions.append(
                UpdateTrackIDs(self.tracks, edge[1], new_track_id, new_lineage_id)
            )
        elif out_degree == 1:  # removed a division edge
            # sibling gets parent's track id (lineage stays the same)
            sibling = next(iter(self.tracks.graph.successors(edge[0])))
            new_track_id = self.tracks.get_track_id(edge[0])
            self.actions.append(UpdateTrackIDs(self.tracks, sibling, new_track_id))
        else:
            raise InvalidActionError(
                f"Expected degree of 0 or 1 after removing edge, got {out_degree}"
            )

        if _top_level:
            self.tracks.action_history.add_new_action(self)
            self.tracks.refresh.emit()
