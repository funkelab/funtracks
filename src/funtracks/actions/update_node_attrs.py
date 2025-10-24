from __future__ import annotations

from typing import TYPE_CHECKING

from ._base import BasicAction

if TYPE_CHECKING:
    from typing import Any

    from funtracks.data_model import Tracks
    from funtracks.data_model.tracks import Node


class UpdateNodeAttrs(BasicAction):
    """Action for user updates to node attributes. Cannot update protected
    attributes (time, area, track id), as these are controlled by internal application
    logic."""

    def __init__(
        self,
        tracks: Tracks,
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
        # Cannot modify annotator-managed features or time
        protected_attrs = set(tracks.annotators.all_features.keys())
        protected_attrs.add(tracks.features.time_key)

        for attr in attrs:
            if attr in protected_attrs:
                raise ValueError(f"Cannot update attribute {attr} manually")
        self.node = node
        self.prev_attrs = {attr: self.tracks.get_node_attr(node, attr) for attr in attrs}
        self.new_attrs = attrs
        self._apply()

    def inverse(self) -> BasicAction:
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
