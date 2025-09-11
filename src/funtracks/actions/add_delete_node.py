from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from funtracks.data_model.graph_attributes import NodeAttr
from funtracks.data_model.solution_tracks import SolutionTracks

from ._base import TracksAction

if TYPE_CHECKING:
    from typing import Any

    from funtracks.data_model import SolutionTracks
    from funtracks.data_model.tracks import Node, SegMask


class AddNode(TracksAction):
    """Action for adding new nodes. If a segmentation should also be added, the
    pixels for each node should be provided. The label to set the pixels will
    be taken from the node id. The existing pixel values are assumed to be
    zero - you must explicitly update any other segmentations that were overwritten
    using an UpdateNodes action if you want to be able to undo the action.
    """

    def __init__(
        self,
        tracks: SolutionTracks,
        node: Node,
        attributes: dict[str, Any],
        pixels: SegMask | None = None,
    ):
        """Create an action to add a new node, with optional segmentation

        Args:
            tracks (Tracks): The Tracks to add the node to
            node (Node): A node id
            attributes (Attrs): Includes times, track_ids, and optionally positions
            pixels (SegMask | None, optional): The segmentation associated with
                the node. Defaults to None.
        """
        super().__init__(tracks)
        self.node = node
        user_attrs = attributes.copy()
        if NodeAttr.TIME.value not in attributes:
            raise ValueError("Must provide a time attribute for each node")
        self.time = attributes.pop(NodeAttr.TIME.value)
        self.position = attributes.pop(NodeAttr.POS.value, None)
        self.pixels = pixels
        self.attributes = user_attrs
        self._apply()

    def inverse(self) -> TracksAction:
        """Invert the action to delete nodes instead"""
        return DeleteNode(self.tracks, self.node)

    def _apply(self) -> None:
        """Apply the action, and set segmentation if provided in self.pixels"""
        if self.pixels is not None:
            self.tracks.set_pixels(self.pixels, self.node)
        attrs = self.attributes
        self.tracks.graph.add_node(self.node)
        self.tracks.set_time(self.node, self.time)
        final_pos: np.ndarray
        if self.tracks.segmentation is not None:
            computed_attrs = self.tracks._compute_node_attrs(self.node, self.time)
            if self.position is None:
                final_pos = np.array(computed_attrs[NodeAttr.POS.value])
            else:
                final_pos = self.position
            attrs[NodeAttr.AREA.value] = computed_attrs[NodeAttr.AREA.value]
        elif self.position is None:
            raise ValueError("Must provide positions or segmentation and ids")
        else:
            final_pos = self.position

        self.tracks.set_position(self.node, final_pos)
        for attr, values in attrs.items():
            self.tracks._set_node_attr(self.node, attr, values)

        track_id = attrs[NodeAttr.TRACK_ID.value]
        if track_id not in self.tracks.track_id_to_node:
            self.tracks.track_id_to_node[track_id] = []
        self.tracks.track_id_to_node[track_id].append(self.node)


class DeleteNode(TracksAction):
    """Action of deleting existing nodes
    If the tracks contain a segmentation, this action also constructs a reversible
    operation for setting involved pixels to zero
    """

    def __init__(
        self,
        tracks: SolutionTracks,
        node: Node,
        pixels: SegMask | None = None,
    ):
        super().__init__(tracks)
        self.node = node
        self.attributes = {
            NodeAttr.TIME.value: self.tracks.get_time(node),
            NodeAttr.POS.value: self.tracks.get_position(node),
            NodeAttr.TRACK_ID.value: self.tracks.get_node_attr(
                node, NodeAttr.TRACK_ID.value
            ),
        }
        self.pixels = self.tracks.get_pixels(node) if pixels is None else pixels
        self._apply()

    def inverse(self) -> TracksAction:
        """Invert this action, and provide inverse segmentation operation if given"""

        return AddNode(self.tracks, self.node, self.attributes, pixels=self.pixels)

    def _apply(self) -> None:
        """ASSUMES THERE ARE NO INCIDENT EDGES - raises valueerror if an edge will be
        removed by this operation
        Steps:
        - For each node
            set pixels to 0 if self.pixels is provided
        - Remove nodes from graph
        """
        if self.pixels is not None:
            self.tracks.set_pixels(self.pixels, 0)

        if isinstance(self.tracks, SolutionTracks):
            self.tracks.track_id_to_node[self.tracks.get_track_id(self.node)].remove(
                self.node
            )

        self.tracks.graph.remove_node(self.node)
