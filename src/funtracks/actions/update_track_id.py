from __future__ import annotations

from typing import TYPE_CHECKING

from ._base import BasicAction

if TYPE_CHECKING:
    from funtracks.data_model import SolutionTracks
    from funtracks.data_model.tracks import Node


class UpdateTrackID(BasicAction):
    def __init__(self, tracks: SolutionTracks, start_node: Node, track_id: int):
        """
        Args:
            tracks (Tracks): The tracks to update
            start_node (Node): The node ID of the first node in the track. All successors
                with the same track id as this node will be updated.
            track_id (int): The new track id to assign.
        """
        super().__init__(tracks)
        self.tracks: SolutionTracks  # Narrow type from base class
        self.start_node = start_node
        self.old_track_id = self.tracks.get_track_id(start_node)
        self.new_track_id = track_id
        self._apply()

    def inverse(self) -> BasicAction:
        """Restore the previous track_id"""
        return UpdateTrackID(self.tracks, self.start_node, self.old_track_id)

    def _apply(self) -> None:
        """Assign a new track id to the track starting with start_node.

        Delegates to TrackAnnotator via notify_annotators(), which performs the
        actual track ID walking and updates.
        """
        self.tracks.notify_annotators(self)
