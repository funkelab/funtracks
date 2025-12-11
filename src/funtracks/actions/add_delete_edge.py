from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._base import BasicAction

if TYPE_CHECKING:
    from typing import Any

    from funtracks.data_model import Tracks
    from funtracks.data_model.tracks import Edge

import tracksdata as td


class AddEdge(BasicAction):
    """Action for adding a new edge. Endpoints must exist already."""

    def __init__(
        self, tracks: Tracks, edge: Edge, attributes: dict[str, Any] | None = None
    ):
        """Create an action to add a new edge, with optional attributes.

        Args:
            tracks (Tracks): The Tracks to add the edge to
            edge (Edge): The edge to add (source, target)
            attributes (dict[str, Any] | None, optional): Edge attributes to set.
               If any computed features are provided, they will be overridden by the
               Annotator. Defaults to None.
        """
        super().__init__(tracks)
        self.edge = edge
        self.attributes = attributes if attributes is not None else {}
        self._apply()

    def inverse(self) -> BasicAction:
        """Delete edges"""
        return DeleteEdge(self.tracks, self.edge)

    def _apply(self) -> None:
        """
        Steps:
        - check for valid endpoints
        - add edge to the graph with provided attributes
        - Trigger feature (re)computation

        Raises:
            ValueError if an endpoint of the edge does not exist
        """
        # Check that both endpoints exist before computing edge attributes
        for node in self.edge:
            if node not in self.tracks.graph.node_ids():
                raise ValueError(
                    f"Cannot add edge {self.edge}: endpoint {node} not in graph yet"
                )

        if self.tracks.graph.has_edge(*self.edge):
            raise ValueError(f"Edge {self.edge} already exists in the graph")

        # Add required solution attribute
        attrs = self.attributes
        attrs[td.DEFAULT_ATTR_KEYS.SOLUTION] = 1

        required_attrs = self.tracks.graph.edge_attr_keys()
        for attr in required_attrs:
            if attr not in attrs:
                attrs[attr] = None

        # Create edge attributes for this specific edge
        self.tracks.graph.add_edge(
            source_id=self.edge[0],
            target_id=self.edge[1],
            attrs=attrs,
        )

        # Notify annotators to recompute features (will overwrite computed ones)
        self.tracks.notify_annotators(self)


class DeleteEdge(BasicAction):
    """Action for deleting an edge. Edge must exist already."""

    def __init__(self, tracks: Tracks, edge: Edge):
        """Action for deleting an edge. Edge must exist already.

        Args:
            tracks (Tracks): The tracks to delete the edge from
            edge (Edge): The edge to delete
        Raises:
            ValueError: If the edge does not exist on the graph
        """
        super().__init__(tracks)
        self.edge = edge
        if not self.tracks.graph.has_edge(*self.edge):
            raise ValueError(f"Edge {self.edge} not in the graph, and cannot be removed")

        # Save all edge feature values from the features dict
        self.attributes = {}
        for key in self.tracks.features.edge_features:
            val = tracks.get_edge_attr(edge, key)
            if val is not None:
                self.attributes[key] = val

        self._apply()

    def inverse(self) -> BasicAction:
        """Restore edge and their attributes"""
        return AddEdge(self.tracks, self.edge, attributes=self.attributes)

    def _apply(self) -> None:
        self.tracks.graph.remove_edge(*self.edge)
