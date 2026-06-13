from __future__ import annotations

from typing import TYPE_CHECKING

import tracksdata as td

from ._base import BasicAction

if TYPE_CHECKING:
    from typing import Any

    from funtracks.data_model.tracks import Node, Tracks


class AddNode(BasicAction):
    """Action for adding new nodes.

    All node attributes (including mask and bbox if applicable) should be
    provided in the ``attributes`` dict.
    """

    def __init__(
        self,
        tracks: Tracks,
        node: Node,
        attributes: dict[str, Any],
    ):
        """Create an action to add a new node.

        Args:
            tracks: The Tracks to add the node to.
            node: A node id.
            attributes: Node attributes including time, tracklet_id, and
                optionally position, mask, bbox, etc.

        Raises:
            ValueError: If time attribute is not in attributes.
            ValueError: If track_id is not in attributes.
            ValueError: If neither position nor a mask feature is in attributes.
        """
        super().__init__(tracks)
        self.tracks: Tracks  # Narrow type from base class
        self.node = int(node)

        # Get keys from tracks features
        time_key = tracks.features.time_key
        track_id_key = tracks.features.tracklet_key
        pos_key = tracks.features.position_key

        # validate the input
        if time_key not in attributes:
            raise ValueError(f"Must provide a time attribute for node {node}")
        if track_id_key not in attributes:
            raise ValueError(f"Must provide a {track_id_key} attribute for node {node}")

        # Check for position — not needed if the default mask is in attributes
        # (position will be computed from the mask by annotators)
        if td.DEFAULT_ATTR_KEYS.MASK not in attributes:
            if isinstance(pos_key, list):
                if not all(key in attributes for key in pos_key):
                    raise ValueError(
                        f"Must provide position or segmentation for node {node}"
                    )
            else:
                if pos_key not in attributes:
                    raise ValueError(
                        f"Must provide position or segmentation for node {node}"
                    )

        self.attributes = attributes
        self._apply()

    def inverse(self) -> BasicAction:
        """Invert the action to delete nodes instead"""
        return DeleteNode(self.tracks, self.node)

    def _apply(self) -> None:
        """Add the node, or revive a soft-deleted one.

        If the node still exists in the full graph (it was soft-deleted, so its
        topology was preserved), revive it: flip solution=True and re-surface it in the
        solution view. Otherwise add a genuinely new node.
        """
        if self.tracks.graph_full.has_node(self.node):
            # Revive: same node id, topology preserved in graph_full. Flip it back into
            # the solution and re-surface it in the view in place (incident edges are
            # revived separately by AddEdge).
            self.tracks.graph_full.update_node_attrs(
                attrs={"solution": True}, node_ids=[self.node]
            )
            self.tracks.graph_solution.add_node_to_view(self.node)
        else:
            # Genuinely new node (solution defaults to True via the schema).
            self.tracks.graph_solution.add_node(
                attrs=dict(self.attributes), index=self.node, validate_keys=False
            )

        # Always notify annotators - they will check their own preconditions
        self.tracks.notify_annotators(self)


class DeleteNode(BasicAction):
    """Action of deleting an existing node.

    Saves all node feature values so the action can be inverted.

    Low-level action — not meant to be used directly. It soft-deletes only the
    node itself (incident edges are dropped from the view by
    ``remove_node_from_view`` but keep ``solution=True`` in ``graph_full``).
    Managing the incident edges' solution flags is the responsibility of the
    enclosing user action (``UserDeleteNode``), which soft-deletes each incident
    edge with its own ``DeleteEdge`` first. Applying a bare ``DeleteNode`` to a
    node that still has in-solution edges therefore leaves ``graph_full``'s edge
    flags inconsistent with ``graph_solution`` — always go through the user action.
    """

    def __init__(
        self,
        tracks: Tracks,
        node: Node,
    ):
        super().__init__(tracks)
        self.tracks: Tracks  # Narrow type from base class
        self.node = int(node)

        # Save all node feature values from the features dict
        # (mask, bbox, and solution are now registered as Features, so they're
        # captured here automatically)
        self.attributes: dict[str, Any] = {}
        for key in self.tracks.features.node_features:
            val = self.tracks.get_node_attr(node, key)
            if val is not None:
                self.attributes[key] = val

        self._apply()

    def inverse(self) -> BasicAction:
        """Invert this action to re-add the node with its saved attributes."""
        return AddNode(self.tracks, self.node, self.attributes)

    def _apply(self) -> None:
        """Soft-delete the node: flag solution=False in the full graph and remove it
        from the solution view only. The node (and its topology) is preserved in
        graph_full so the delete is reversible and the node remains a candidate.
        """
        self.tracks.graph_full.update_node_attrs(
            attrs={"solution": False}, node_ids=[self.node]
        )
        self.tracks.graph_solution.remove_node_from_view(self.node)
        self.tracks.notify_annotators(self)
