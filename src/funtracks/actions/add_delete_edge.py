from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from ._base import TracksAction

if TYPE_CHECKING:
    from funtracks.data_model import SolutionTracks
    from funtracks.data_model.tracks import Edge


class AddEdge(TracksAction):
    """Action for adding new edges"""

    def __init__(self, tracks: SolutionTracks, edge: Edge):
        super().__init__(tracks)
        self.edge = edge
        self._apply()

    def inverse(self) -> TracksAction:
        """Delete edges"""
        return DeleteEdge(self.tracks, self.edge)

    def _apply(self) -> None:
        """
        Steps:
        - add each edge to the graph. Assumes all edges are valid (they should be checked
        at this point already)
        """
        attrs: dict[str, Sequence[Any]] = {}
        attrs.update(self.tracks._compute_edge_attrs(self.edge))
        for node in self.edge:
            if not self.tracks.graph.has_node(node):
                raise ValueError(
                    f"Cannot add edge {self.edge}: endpoint {node} not in graph yet"
                )
        self.tracks.graph.add_edge(self.edge[0], self.edge[1], **attrs)


class DeleteEdge(TracksAction):
    """Action for deleting edges"""

    def __init__(self, tracks: SolutionTracks, edge: Edge):
        super().__init__(tracks)
        self.edge = edge
        self._apply()

    def inverse(self) -> TracksAction:
        """Restore edges and their attributes"""
        return AddEdge(self.tracks, self.edge)

    def _apply(self) -> None:
        """Steps:
        - Remove the edges from the graph
        """
        if self.tracks.graph.has_edge(*self.edge):
            self.tracks.graph.remove_edge(*self.edge)
        else:
            raise ValueError(f"Edge {self.edge} not in the graph, and cannot be removed")
