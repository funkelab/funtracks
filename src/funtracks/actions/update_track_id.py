from __future__ import annotations

from typing import TYPE_CHECKING

from ._base import TracksAction

if TYPE_CHECKING:
    from funtracks.data_model import SolutionTracks
    from funtracks.data_model.tracks import Node


class UpdateTrackID(TracksAction):
    def __init__(self, tracks: SolutionTracks, start_node: Node, track_id: int):
        """
        Args:
            tracks (Tracks): The tracks to update
            start_node (Node): The node ID of the first node in the track. All successors
                with the same track id as this node will be updated.
            track_id (int): The new track id to assign.
        """
        super().__init__(tracks)
        self.start_node = start_node
        self.old_track_id = self.tracks.get_track_id(start_node)
        self.new_track_id = track_id
        self._apply()

    def inverse(self) -> TracksAction:
        """Restore the previous track_id"""
        return UpdateTrackID(self.tracks, self.start_node, self.old_track_id)

    def _apply(self) -> None:
        """Assign a new track id to the track starting with start_node."""
        old_track_id = self.tracks.get_track_id(self.start_node)
        curr_node = self.start_node
        while self.tracks.get_track_id(curr_node) == old_track_id:
            # update the track id
            self.tracks.set_track_id(curr_node, self.new_track_id)
            # getting the next node (picks one if there are two)
            successors = list(self.tracks.graph.successors(curr_node))
            if len(successors) == 0:
                break
            curr_node = successors[0]
