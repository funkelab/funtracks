from __future__ import annotations

from typing import Any

from ..actions._base import ActionGroup
from ..actions.add_delete_edge import AddEdge
from ..actions.set_feature_values import SetFeatureValues
from ..actions.update_track_id import UpdateTrackID
from ..features import Feature
from ..project import Project


class UserSelectEdge(ActionGroup):
    """Assumes that the endpoints are already selected and have track ids"""

    def __init__(
        self,
        project: Project,
        edge: tuple[int, int],
        attributes: dict[Feature, Any],
    ):
        super().__init__(project, actions=[])
        source, target = edge
        if not project.solution.has_node(source):
            raise ValueError(
                f"Source node {source} not in solution yet - must be added before edge"
            )
        if not project.solution.has_node(target):
            raise ValueError(
                f"Target node {target} not in solution yet - must be added before edge"
            )

        # pin the endpoints to selected
        features = project.cand_graph.features
        node_attrs = {}
        node_attrs[features.node_selection_pin] = True

        for node in edge:
            self.actions.append(SetFeatureValues(project, node, node_attrs))

        # update track ids if needed
        out_degree = self.project.solution.out_degree(source)
        if out_degree == 0:  # joining two segments
            # assign the track id of the source node to the target and all out
            # edges until end of track
            new_track_id = self.project.cand_graph.get_track_id(source)
            self.actions.append(
                UpdateTrackID(self.project.cand_graph, edge[1], new_track_id)
            )
        elif out_degree == 1:  # creating a division
            # assign a new track id to existing child
            successor = next(iter(self.project.solution.successors(source)))
            self.actions.append(
                UpdateTrackID(self.project, successor, self.project.get_next_track_id())
            )
        else:
            raise RuntimeError(
                f"Expected degree of 0 or 1 before adding edge, got {out_degree}"
            )

        # if the edge is not in the candidate graph, add it, otherwise just set
        # the edge to be selected and pin the selection
        attributes[features.edge_selected] = True
        attributes[features.edge_selection_pin] = True
        if not self.project.cand_graph.has_edge(edge):
            self.actions.append(AddEdge(project, edge, attributes))
        else:
            self.actions.append(SetFeatureValues(project, edge, attributes))
