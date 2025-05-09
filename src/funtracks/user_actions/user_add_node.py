from __future__ import annotations

from typing import Any

import numpy as np

from ..actions._base import ActionGroup
from ..actions.add_delete_edge import AddEdge, DeleteEdge
from ..actions.add_delete_node import AddNode
from ..features._base import Feature
from ..project import Project


class UserAddNode(ActionGroup):
    def __init__(
        self,
        project: Project,
        node: int,
        attributes: dict[Feature, Any],
        pixels: tuple[np.ndarray, ...] | None = None,
    ):
        super().__init__(project, actions=[AddNode(project, node, attributes, pixels)])
        if attributes[self.project.cand_graph.features.track_id] is not None:
            pred, succ = self.project.cand_graph.get_track_neighbors()
            if pred is not None and succ is not None:
                self.actions.append(DeleteEdge((pred, succ)))
            if pred is not None:
                self.actions.append(AddEdge((pred, node)))
            if succ is not None:
                self.actions.append(AddEdge((node, succ)))

            # TODO: more invalid track ids (if extending track in time past a division
