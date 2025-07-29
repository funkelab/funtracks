from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from ._base import TracksAction

if TYPE_CHECKING:
    from collections.abc import Iterable

    from funtracks.data_model.tracks import Edge, Tracks


class AddEdges(TracksAction):
    """Action for adding new edges"""

    def __init__(self, tracks: Tracks, edges: Iterable[Edge]):
        super().__init__(tracks)
        self.edges = edges
        self._apply()

    def inverse(self):
        """Delete edges"""
        return DeleteEdges(self.tracks, self.edges)

    def _apply(self):
        """
        Steps:
        - add each edge to the graph. Assumes all edges are valid (they should be checked
        at this point already)
        """
        attrs: dict[str, Sequence[Any]] = {}
        attrs.update(self.tracks._compute_edge_attrs(self.edges))
        for idx, edge in enumerate(self.edges):
            for node in edge:
                if not self.tracks.graph.has_node(node):
                    raise KeyError(
                        f"Cannot add edge {edge}: endpoint {node} not in graph yet"
                    )
            self.tracks.graph.add_edge(
                edge[0], edge[1], **{key: vals[idx] for key, vals in attrs.items()}
            )


class DeleteEdges(TracksAction):
    """Action for deleting edges"""

    def __init__(self, tracks: Tracks, edges: Iterable[Edge]):
        super().__init__(tracks)
        self.edges = edges
        self._apply()

    def inverse(self):
        """Restore edges and their attributes"""
        return AddEdges(self.tracks, self.edges)

    def _apply(self):
        """Steps:
        - Remove the edges from the graph
        """
        for edge in self.edges:
            if self.tracks.graph.has_edge(*edge):
                self.tracks.graph.remove_edge(*edge)
            else:
                raise KeyError(f"Edge {edge} not in the graph, and cannot be removed")
