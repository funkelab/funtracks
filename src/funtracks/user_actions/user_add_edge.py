from __future__ import annotations

from typing import Any

from ..actions._base import ActionGroup
from ..actions.add_delete_edge import AddEdge
from ..actions.update_track_id import UpdateTrackID
from ..features._base import Feature
from ..project import Project


class UserAddEdge(ActionGroup):
    def __init__(
        self,
        project: Project,
        edge: tuple[int, int],
        attributes: dict[Feature, Any],
    ):
        super().__init__(project, actions=[AddEdge(project, edge, attributes)])
        if self.project.solution.has_edge(edge):
            out_degree = self.project.solution.out_degree(edge[0])
            if out_degree == 0:  # joining two segments
                # assign the track id of the source node to the target and all out
                # edges until end of track
                new_track_id = self.project.cand_graph.get_track_id(edge[0])
                self.actions.append(
                    UpdateTrackID(self.project.cand_graph, edge[1], new_track_id)
                )
            elif out_degree == 1:  # creating a division
                # assign a new track id to existing child
                successor = next(iter(self.project.solution.successors(edge[0])))
                self.actions.append(
                    UpdateTrackID(
                        self.project, successor, self.project.get_next_track_id()
                    )
                )
            else:
                raise RuntimeError(
                    f"Expected degree of 0 or 1 before adding edge, got {out_degree}"
                )

        self.actions.append(AddEdge(self.project, edge, attributes))
