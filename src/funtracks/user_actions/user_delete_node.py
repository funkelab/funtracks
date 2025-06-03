from __future__ import annotations

from ..actions._base import ActionGroup
from ..actions.add_delete_edge import AddEdge, DeleteEdge
from ..actions.add_delete_node import DeleteNode
from ..project import Project


class UserDeleteNode(ActionGroup):
    def __init__(
        self,
        project: Project,
        node: int,
    ):
        super().__init__(project, actions=[])
        # delete adjacent edges
        for pred in self.project.cand_graph.predecessors(node):
            self.actions.append(DeleteEdge(project, (pred, node)))
        for succ in self.project.cand_graph.successors(node):
            self.actions.append(DeleteEdge(project, (node, succ)))

        # connect child and parent in track, if applicable
        track_id = self.project.cand_graph.get_track_id(node)
        if track_id is not None:
            time = self.project.cand_graph.get_time(node)
            pred, succ = self.project.cand_graph.get_track_neighbors(track_id, time)
            if pred is not None and succ is not None:
                self.actions.append(AddEdge(project, (pred, succ), {}))

        # delete node
        self.actions.append(DeleteNode(project, node))

        # TODO: relabel track ids if necessary (delete one child of division)
