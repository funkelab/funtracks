from __future__ import annotations

from funtracks.data_model import SolutionTracks

from ..actions._base import ActionGroup
from ..actions.add_delete_edge import AddEdge
from ..actions.update_track_id import UpdateTrackID


class UserAddEdge(ActionGroup):
    """Assumes that the endpoints already exist and have track ids"""

    def __init__(
        self,
        tracks: SolutionTracks,
        edge: tuple[int, int],
    ):
        super().__init__(tracks, actions=[])
        source, target = edge
        if not tracks.graph.has_node(source):
            raise ValueError(
                f"Source node {source} not in solution yet - must be added before edge"
            )
        if not tracks.graph.has_node(target):
            raise ValueError(
                f"Target node {target} not in solution yet - must be added before edge"
            )

        # update track ids if needed
        out_degree = self.tracks.graph.out_degree(source)
        if out_degree == 0:  # joining two segments
            # assign the track id of the source node to the target and all out
            # edges until end of track
            new_track_id = self.tracks.get_track_id(source)
            self.actions.append(UpdateTrackID(self.tracks, edge[1], new_track_id))
        elif out_degree == 1:  # creating a division
            # assign a new track id to existing child
            successor = next(iter(self.tracks.graph.successors(source)))
            self.actions.append(
                UpdateTrackID(self.tracks, successor, self.tracks.get_next_track_id())
            )
        else:
            raise RuntimeError(
                f"Expected degree of 0 or 1 before adding edge, got {out_degree}"
            )

        self.actions.append(AddEdge(tracks, edge))
