from __future__ import annotations

from typing import Any

from ..features._base import Feature
from ..project import Project
from ._base import TracksAction


class AddEdge(TracksAction):
    """Action for adding new edges"""

    def __init__(
        self,
        project: Project,
        edge: tuple[int, int],
        provided_features: dict[Feature, Any],
    ):
        super().__init__(project)
        self.edge = edge
        self.project.graph.features.validate_new_edge_features(provided_features)
        self.provided_features = provided_features
        self._apply()

    def inverse(self):
        """Delete edges"""
        return DeleteEdge(self.project, self.edge)

    def _apply(self):
        """
        Steps:
        - add edge to graph
        - set feature values
        """
        for node in self.edge:
            if not self.project.graph.has_node(node):
                raise KeyError(
                    f"Cannot add edge {self.edge}: endpoint {node} not in graph yet"
                )
        self.project.graph.add_edge(self.edge, self.provided_features)
        for feature in self.project.graph.features.edge_features:
            if feature.computed:
                value = feature.update(self.project, self.edge)
                self.project.graph.set_feature_value(self.edge, feature, value)


class DeleteEdge(TracksAction):
    """Action for deleting edges"""

    def __init__(self, project: Project, edge: tuple[int, int]):
        super().__init__(project)
        self.edge = edge
        self.attributes = {
            feature: self.project.graph.get_feature_value(self.edge, feature)
            for feature in self.project.graph.features.edge_features
        }
        self._apply()

    def inverse(self):
        """Restore edges and their attributes"""
        return AddEdge(self.project, self.edge, self.attributes)

    def _apply(self):
        """Steps:
        - Remove the edges from the graph
        """
        if not self.project.graph.has_edge(self.edge):
            raise KeyError(f"Edge {self.edge} not in the graph, and cannot be removed")
        # features = self.project.cand_graph.features
        # self.project.cand_graph.set_feature_value(self.edge, features.edge_selection_pin, False)
        self.project.graph.remove_edge(self.edge)
