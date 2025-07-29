from __future__ import annotations

import numpy as np

from funtracks.data_model import SolutionTracks

from ..actions._base import ActionGroup
from ..actions.add_delete_edge import AddEdge, DeleteEdge
from ..actions.add_delete_node import DeleteNode


class UserDeleteNode(ActionGroup):
    def __init__(
        self,
        tracks: SolutionTracks,
        node: int,
        pixels: None | tuple[np.ndarray, ...] = None,
    ):
        super().__init__(tracks, actions=[])
        # delete adjacent edges
        for pred in self.tracks.predecessors(node):
            self.actions.append(DeleteEdge(tracks, (pred, node)))
        for succ in self.tracks.successors(node):
            self.actions.append(DeleteEdge(tracks, (node, succ)))

        # connect child and parent in track, if applicable
        track_id = self.tracks.get_track_id(node)
        if track_id is not None:
            time = self.tracks.get_time(node)
            predecessor, succcessor = self.tracks.get_track_neighbors(track_id, time)
            if predecessor is not None and succcessor is not None:
                self.actions.append(AddEdge(tracks, (predecessor, succcessor)))

        # delete node
        self.actions.append(DeleteNode(tracks, node, pixels=pixels))

        # TODO: relabel track ids if necessary (delete one child of division)
