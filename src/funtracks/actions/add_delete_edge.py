from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._base import BasicAction

if TYPE_CHECKING:
    from typing import Any

    from funtracks.data_model import Tracks
    from funtracks.data_model.tracks import Edge


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
        # Check that both endpoints exist in the solution before adding the edge
        for node in self.edge:
            if not self.tracks.graph_solution.has_node(node):
                raise ValueError(
                    f"Cannot add edge {self.edge}: endpoint {node} not in solution yet"
                )

        if self.tracks.graph_solution.has_edge(*self.edge):
            raise ValueError(f"Edge {self.edge} already exists in the solution")

        if self.tracks.graph_full.has_edge(*self.edge):
            # Revive a soft-deleted edge (already present in the full graph as a
            # candidate): flip solution=True, apply any caller-provided attributes (so
            # revive matches the add-new branch), and re-surface it in the solution view.
            edge_id = self.tracks.graph_full.edge_id(self.edge[0], self.edge[1])
            # Values are wrapped in single-element lists because update_edge_attrs
            # reads a bare list value (e.g. a vector feature) as one-value-per-edge.
            revive_attrs = {k: [v] for k, v in self.attributes.items() if k != "solution"}
            revive_attrs["solution"] = [True]
            self.tracks.graph_full.update_edge_attrs(
                attrs=revive_attrs, edge_ids=[edge_id]
            )
            self.tracks.graph_solution.add_edge_to_view(self.edge[0], self.edge[1])
        else:
            attrs = dict(self.attributes)

            # Fill in missing edge attributes with schema defaults (includes
            # solution and any other registered edge attrs).
            schemas = self.tracks.graph_solution._edge_attr_schemas()
            for attr in self.tracks.graph_solution.edge_attr_keys():
                if attr not in attrs:
                    # An edge added to a Tracks graph is by definition part of the
                    # solution, so default `solution` to True rather than the schema
                    # default, which can be wrong (e.g. Float64/0.0) on graphs loaded
                    # from geff. An explicit caller-provided value still wins.
                    attrs[attr] = (
                        True if attr == "solution" else schemas[attr].default_value
                    )

            # Create edge attributes for this specific edge
            self.tracks.graph_solution.add_edge(
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
        if not self.tracks.graph_solution.has_edge(*self.edge):
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
        """Soft-delete the edge: flag solution=False in the full graph and remove it
        from the solution view only. The edge is preserved in graph_full (as a
        candidate) so the delete is reversible."""
        edge_id = self.tracks.graph_full.edge_id(self.edge[0], self.edge[1])
        self.tracks.graph_full.update_edge_attrs(
            attrs={"solution": False}, edge_ids=[edge_id]
        )
        self.tracks.graph_solution.remove_edge_from_view(self.edge[0], self.edge[1])
        self.tracks.notify_annotators(self)
