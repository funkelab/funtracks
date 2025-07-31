from __future__ import annotations

from typing import TYPE_CHECKING

from funtracks.data_model.graph_attributes import NodeAttr

from ._base import TracksAction

if TYPE_CHECKING:
    from typing import Any

    from funtracks.data_model import SolutionTracks
    from funtracks.data_model.tracks import Node


class UpdateNodeAttrs(TracksAction):
    """Action for user updates to node attributes. Cannot update protected
    attributes (time, area, track id), as these are controlled by internal application
    logic."""

    def __init__(
        self,
        tracks: SolutionTracks,
        node: Node,
        attrs: dict[str, Any],
    ):
        """
        Args:
            tracks (Tracks): The tracks to update the node attributes for
            node (Node): The node to update the attributes for
            attrs (dict[str, Any]): A mapping from attribute name to list of new attribute
                values for the given nodes.

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
        self.node = node
        self.prev_attrs = {attr: self.tracks.get_node_attr(node, attr) for attr in attrs}
        self.new_attrs = attrs
        self._apply()

    def inverse(self) -> TracksAction:
        """Restore previous attributes"""
        return UpdateNodeAttrs(
            self.tracks,
            self.node,
            self.prev_attrs,
        )

    def _apply(self) -> None:
        """Set new attributes"""
        for attr, value in self.new_attrs.items():
            self.tracks._set_node_attr(self.node, attr, value)
