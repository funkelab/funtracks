from __future__ import annotations

from funtracks.data_model import SolutionTracks

from ..actions._base import ActionGroup
from ..actions.add_delete_edge import DeleteEdge
from ..actions.update_track_id import UpdateTrackID


class UserDeleteEdge(ActionGroup):
    def __init__(
        self,
        tracks: SolutionTracks,
        edge: tuple[int, int],
    ):
        super().__init__(tracks, actions=[])
        if not self.tracks.graph.has_edge(*edge):
            raise ValueError(f"Edge {edge} not in solution, can't remove")

        self.actions.append(DeleteEdge(tracks, edge))
        out_degree = self.tracks.graph.out_degree(edge[0])
        if out_degree == 0:  # removed a normal (non division) edge
            new_track_id = self.tracks.get_next_track_id()
            self.actions.append(UpdateTrackID(self.tracks, edge[1], new_track_id))
        elif out_degree == 1:  # removed a division edge
            sibling = next(iter(self.tracks.graph.successors(edge[0])))
            new_track_id = self.tracks.get_track_id(edge[0])
            self.actions.append(UpdateTrackID(self.tracks, sibling, new_track_id))
        else:
            raise RuntimeError(
                f"Expected degree of 0 or 1 after removing edge, got {out_degree}"
            )
