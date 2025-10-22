from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from ._base import TracksAction

if TYPE_CHECKING:
    from funtracks.data_model import SolutionTracks
    from funtracks.data_model.tracks import Edge


class AddEdge(TracksAction):
    """Action for adding a new edge. Endpoints must exist already."""

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

        Raises:
            ValueError if an endpoint of the edge does not exist
        """
        # Check that both endpoints exist before computing edge attributes
        for node in self.edge:
            if not self.tracks.graph.has_node(node):
                raise ValueError(
                    f"Cannot add edge {self.edge}: endpoint {node} not in graph yet"
                )

        attrs: dict[str, Sequence[Any]] = {}
        attrs.update(self.tracks._compute_edge_attrs(self.edge))
        self.tracks.graph.add_edge(self.edge[0], self.edge[1], **attrs)


class DeleteEdge(TracksAction):
    """Action for deleting an edge. Edge must exist already."""

    def __init__(self, tracks: SolutionTracks, edge: Edge):
        """Action for deleting an edge. Edge must exist already.

        Args:
            tracks (SolutionTracks): The tracks to delete the edge from
            edge (Edge): The edge to delete
        Raises:
            ValueError: If the edge does not exist on the graph
        """
        super().__init__(tracks)
        self.edge = edge
        if not self.tracks.graph.has_edge(*self.edge):
            raise ValueError(f"Edge {self.edge} not in the graph, and cannot be removed")
        self._apply()

    def inverse(self) -> TracksAction:
        """Restore edge and their attributes"""
        return AddEdge(self.tracks, self.edge)

    def _apply(self) -> None:
        self.tracks.graph.remove_edge(*self.edge)
