from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from funtracks.exceptions import InvalidActionError

from ..actions._base import ActionGroup
from ..actions.add_delete_edge import AddEdge
from ..actions.update_track_id import UpdateTrackID
from .user_delete_edge import UserDeleteEdge

if TYPE_CHECKING:
    from funtracks.data_model import SolutionTracks


class UserAddEdge(ActionGroup):
    """Assumes that the endpoints already exist and have track ids.

    Args:
        tracks (SolutionTracks): the tracks to add the edge to
        edge (tuple[int, int]): The edge to add
        force (bool, optional): Whether to force the action by removing any conflicting
            edges. Defaults to False.
    """

    def __init__(
        self,
        tracks: SolutionTracks,
        edge: tuple[int, int],
        force: bool = False,
    ):
        super().__init__(tracks, actions=[])
        self.tracks: SolutionTracks  # Narrow type from base class
        source, target = edge
        if not tracks.graph.has_node(source):
            raise InvalidActionError(
                f"Source node {source} not in solution yet - must be added before edge"
            )
        if not tracks.graph.has_node(target):
            raise InvalidActionError(
                f"Target node {target} not in solution yet - must be added before edge"
            )

        # Check if making a merge. If yes and force, remove the other edge and update
        # track ids.
        in_degree_target = self.tracks.graph.in_degree(target)
        if in_degree_target > 0:
            if not force:
                raise InvalidActionError(
                    f"Cannot make a merge edge in a tracking solution: node {target} "
                    "already has an in edge",
                    forceable=True,
                )
            else:
                merge_edge = list(self.tracks.graph.in_edges(target))[0]
                warnings.warn(
                    f"Removing edge {merge_edge} to add new edge without merging.",
                    stacklevel=2,
                )
                self.actions.append(UserDeleteEdge(self.tracks, merge_edge))

        # update track ids if needed
        out_degree_source = self.tracks.graph.out_degree(source)
        if out_degree_source == 0:  # joining two segments
            # assign the track id of the source node to the target and all out
            # edges until end of track
            new_track_id = self.tracks.get_track_id(source)
            self.actions.append(UpdateTrackID(self.tracks, edge[1], new_track_id))
        elif out_degree_source == 1:  # creating a division
            # assign a new track id to existing child
            successor = next(iter(self.tracks.graph.successors(source)))
            self.actions.append(
                UpdateTrackID(self.tracks, successor, self.tracks.get_next_track_id())
            )
        else:
            raise InvalidActionError(
                f"Expected degree of 0 or 1 before adding edge, got {out_degree_source}"
            )

        self.actions.append(AddEdge(tracks, edge))
