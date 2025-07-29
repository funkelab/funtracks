from __future__ import annotations

from typing import Any

import numpy as np

from funtracks.data_model import NodeAttr, SolutionTracks

from ..actions._base import ActionGroup
from ..actions.add_delete_edge import AddEdge, DeleteEdge
from ..actions.add_delete_node import AddNode


class UserAddNode(ActionGroup):
    def __init__(
        self,
        tracks: SolutionTracks,
        node: int,
        attributes: dict[str, Any],
        pixels: tuple[np.ndarray, ...] | None = None,
    ):
        super().__init__(tracks, actions=[])
        self.actions.append(AddNode(tracks, node, attributes, pixels))
        track_id = attributes.get(NodeAttr.TRACK_ID.value, None)
        if track_id is not None:
            time = self.tracks.get_time(node)
            pred, succ = self.tracks.get_track_neighbors(track_id, time)
            if pred is not None and succ is not None:
                self.actions.append(DeleteEdge(tracks, (pred, succ)))
            if pred is not None:
                self.actions.append(AddEdge(tracks, (pred, node)))
            if succ is not None:
                self.actions.append(AddEdge(tracks, (node, succ)))

            # TODO: more invalid track ids (if extending track in time past a division
