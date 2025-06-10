from __future__ import annotations

from ..actions._base import ActionGroup
from ..actions.add_delete_edge import DeleteEdge
from ..actions.update_track_id import UpdateTrackID
from ..project import Project


class UserDeleteEdge(ActionGroup):
    def __init__(
        self,
        project: Project,
        edge: tuple[int, int],
    ):
        super().__init__(project, actions=[])
        if not self.project.solution.has_edge(edge):
            raise ValueError(f"Edge {edge} not in solution, can't remove")

        self.actions.append(DeleteEdge(project, edge))
        out_degree = self.project.solution.out_degree(edge[0])
        if out_degree == 0:  # removed a normal (non division) edge
            new_track_id = self.project.get_next_track_id()
            self.actions.append(UpdateTrackID(self.project, edge[1], new_track_id))
        elif out_degree == 1:  # removed a division edge
            sibling = next(iter(self.project.solution.successors(edge[0])))
            new_track_id = self.project.cand_graph.get_track_id(edge[0])
            self.actions.append(UpdateTrackID(self.project, sibling, new_track_id))
        else:
            raise RuntimeError(
                f"Expected degree of 0 or 1 after removing edge, got {out_degree}"
            )
