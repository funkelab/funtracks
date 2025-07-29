from __future__ import annotations

from typing import TYPE_CHECKING

from funtracks.data_model.graph_attributes import NodeAttr

from ._base import TracksAction

if TYPE_CHECKING:
    from collections.abc import Iterable

    from funtracks.data_model.tracks import Attrs, Node, Tracks


class UpdateNodeAttrs(TracksAction):
    """Action for user updates to node attributes. Cannot update protected
    attributes (time, area, track id), as these are controlled by internal application
    logic."""

    def __init__(
        self,
        tracks: Tracks,
        nodes: Iterable[Node],
        attrs: Attrs,
    ):
        """
        Args:
            tracks (Tracks): The tracks to update the node attributes for
            nodes (Iterable[Node]): The nodes to update the attributes for
            attrs (Attrs): A mapping from attribute name to list of new attribute values
                for the given nodes.

        Raises:
            ValueError: If a protected attribute is in the given attribute mapping.
        """
        super().__init__(tracks)
        protected_attrs = [
            tracks.time_attr,
            NodeAttr.AREA.value,
            NodeAttr.TRACK_ID.value,
        ]
        for attr in attrs:
            if attr in protected_attrs:
                raise ValueError(f"Cannot update attribute {attr} manually")
        self.nodes = nodes
        self.prev_attrs = {
            attr: self.tracks.get_nodes_attr(nodes, attr) for attr in attrs
        }
        self.new_attrs = attrs
        self._apply()

    def inverse(self):
        """Restore previous attributes"""
        return UpdateNodeAttrs(
            self.tracks,
            self.nodes,
            self.prev_attrs,
        )

    def _apply(self):
        """Set new attributes"""
        for attr, values in self.new_attrs.items():
            self.tracks._set_nodes_attr(self.nodes, attr, values)
