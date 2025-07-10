from __future__ import annotations

from typing import Any

import numpy as np

from funtracks import CandGraph, Project

from ..actions._base import ActionGroup
from ..actions.add_delete_edge import AddEdge, DeleteEdge
from ..actions.add_delete_node import AddNode
from ..features._base import Feature


class UserAddNode(ActionGroup):
    def __init__(
        self,
        project: Project,
        node: int,
        attributes: dict[Feature, Any],
        pixels: tuple[np.ndarray, ...] | None = None,
    ):
        super().__init__(project, actions=[])
        features = project.graph.features
        attributes[features.node_selected] = True
        attributes[features.node_selection_pin] = True
        self.actions.append(AddNode(project, node, attributes, pixels))
        if isinstance(project.graph, CandGraph):
            for edge, features in project.graph.get_candidate_edges(node):
                self.actions.append(AddEdge(project, edge, features))
        track_id = attributes[self.project.graph.features.track_id]
        if track_id is not None:
            time = self.project.graph.get_time(node)
            pred, succ = self.project.graph.get_track_neighbors(track_id, time)
            if pred is not None and succ is not None:
                self.actions.append(DeleteEdge(project, (pred, succ)))
            if pred is not None:
                self.actions.append(AddEdge(project, (pred, node), {}))
            if succ is not None:
                self.actions.append(AddEdge(project, (node, succ), {}))

            # TODO: more invalid track ids (if extending track in time past a division
