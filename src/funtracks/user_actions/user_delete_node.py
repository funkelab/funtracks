from __future__ import annotations

import numpy as np

from ..actions._base import ActionGroup
from ..actions.add_delete_edge import AddEdge, DeleteEdge
from ..actions.add_delete_node import DeleteNode
from ..project import Project


class UserDeleteNode(ActionGroup):
    def __init__(
        self,
        project: Project,
        node: int,
        pixels: None | tuple[np.ndarray, ...] = None,
    ):
        super().__init__(project, actions=[])
        # delete adjacent edges
        for pred in self.project.graph.predecessors(node):
            self.actions.append(DeleteEdge(project, (pred, node)))
        for succ in self.project.graph.successors(node):
            self.actions.append(DeleteEdge(project, (node, succ)))

        # connect child and parent in track, if applicable
        track_id = self.project.graph.get_track_id(node)
        if track_id is not None:
            time = self.project.graph.get_time(node)
            pred, succ = self.project.graph.get_track_neighbors(track_id, time)
            if pred is not None and succ is not None:
                self.actions.append(AddEdge(project, (pred, succ), {}))

        # delete node
        self.actions.append(DeleteNode(project, node, pixels=pixels))

        # TODO: relabel track ids if necessary (delete one child of division)
