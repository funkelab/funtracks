from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..actions._base import ActionGroup
from ..actions.add_delete_edge import AddEdge, DeleteEdge
from ..actions.add_delete_node import DeleteNode
from ..actions.update_track_id import UpdateTrackID

if TYPE_CHECKING:
    from funtracks.data_model import SolutionTracks


class UserDeleteNode(ActionGroup):
    def __init__(
        self,
        tracks: SolutionTracks,
        node: int,
        pixels: None | tuple[np.ndarray, ...] = None,
    ):
        super().__init__(tracks, actions=[])
        self.tracks: SolutionTracks  # Narrow type from base class
        # delete adjacent edges
        for pred in self.tracks.predecessors(node):
            siblings = self.tracks.successors(pred)
            # if you are deleting the first node after a division, relabel
            # the track id of the other child to match the parent
            if len(siblings) == 2:
                siblings.remove(node)
                sib = siblings[0]
                new_track_id = self.tracks.get_track_id(pred)
                self.actions.append(UpdateTrackID(tracks, sib, new_track_id))
            self.actions.append(DeleteEdge(tracks, (pred, node)))
        for succ in self.tracks.successors(node):
            self.actions.append(DeleteEdge(tracks, (node, succ)))

        # connect child and parent in track, if applicable
        track_id = self.tracks.get_track_id(node)
        if track_id is not None:
            time = self.tracks.get_time(node)
            predecessor, successor = self.tracks.get_track_neighbors(track_id, time)
            if predecessor is not None and successor is not None:
                self.actions.append(AddEdge(tracks, (predecessor, successor)))

        # delete node
        self.actions.append(DeleteNode(tracks, node, pixels=pixels))
