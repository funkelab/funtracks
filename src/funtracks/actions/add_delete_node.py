from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from funtracks.data_model.graph_attributes import NodeAttr
from funtracks.data_model.solution_tracks import SolutionTracks

from ._base import TracksAction

if TYPE_CHECKING:
    from collections.abc import Iterable

    from funtracks.data_model.tracks import Attrs, Node, SegMask, Tracks


class AddNodes(TracksAction):
    """Action for adding new nodes. If a segmentation should also be added, the
    pixels for each node should be provided. The label to set the pixels will
    be taken from the node id. The existing pixel values are assumed to be
    zero - you must explicitly update any other segmentations that were overwritten
    using an UpdateNodes action if you want to be able to undo the action.
    """

    def __init__(
        self,
        tracks: Tracks,
        nodes: Iterable[Node],
        attributes: Attrs,
        pixels: Iterable[SegMask] | None = None,
    ):
        """Create an action to add new nodes, with optional segmentation

        Args:
            tracks (Tracks): The Tracks to add the nodes to
            nodes (Node): A list of node ids
            attributes (Attrs): Includes times and optionally positions
            pixels (list[SegMask] | None, optional): The segmentations associated with
                each node. Defaults to None.
        """
        super().__init__(tracks)
        self.nodes = nodes
        user_attrs = attributes.copy()
        self.times = attributes.pop(NodeAttr.TIME.value, None)
        self.positions = attributes.pop(NodeAttr.POS.value, None)
        self.pixels = pixels
        self.attributes = user_attrs
        self._apply()

    def inverse(self):
        """Invert the action to delete nodes instead"""
        return DeleteNodes(self.tracks, self.nodes)

    def _apply(self):
        """Apply the action, and set segmentation if provided in self.pixels"""
        if self.pixels is not None:
            self.tracks.set_pixels(self.pixels, self.nodes)
        attrs = self.attributes
        if attrs is None:
            attrs = {}
        self.tracks.graph.add_nodes_from(self.nodes)
        self.tracks.set_times(self.nodes, self.times)
        final_pos: np.ndarray
        if self.tracks.segmentation is not None:
            computed_attrs = self.tracks._compute_node_attrs(self.nodes, self.times)
            if self.positions is None:
                final_pos = np.array(computed_attrs[NodeAttr.POS.value])
            else:
                final_pos = self.positions
            attrs[NodeAttr.AREA.value] = computed_attrs[NodeAttr.AREA.value]
        elif self.positions is None:
            raise ValueError("Must provide positions or segmentation and ids")
        else:
            final_pos = self.positions

        self.tracks.set_positions(self.nodes, final_pos)
        for attr, values in attrs.items():
            self.tracks._set_nodes_attr(self.nodes, attr, values)

        if isinstance(self.tracks, SolutionTracks):
            for node, track_id in zip(
                self.nodes, attrs[NodeAttr.TRACK_ID.value], strict=True
            ):
                if track_id not in self.tracks.track_id_to_node:
                    self.tracks.track_id_to_node[track_id] = []
                self.tracks.track_id_to_node[track_id].append(node)


class DeleteNodes(TracksAction):
    """Action of deleting existing nodes
    If the tracks contain a segmentation, this action also constructs a reversible
    operation for setting involved pixels to zero
    """

    def __init__(
        self,
        tracks: Tracks,
        nodes: Iterable[Node],
        pixels: Iterable[SegMask] | None = None,
    ):
        super().__init__(tracks)
        self.nodes = nodes
        self.attributes = {
            NodeAttr.TIME.value: self.tracks.get_times(nodes),
            self.tracks.pos_attr: self.tracks.get_positions(nodes),
            NodeAttr.TRACK_ID.value: self.tracks.get_nodes_attr(
                nodes, NodeAttr.TRACK_ID.value
            ),
        }
        self.pixels = self.tracks.get_pixels(nodes) if pixels is None else pixels
        self._apply()

    def inverse(self):
        """Invert this action, and provide inverse segmentation operation if given"""

        return AddNodes(self.tracks, self.nodes, self.attributes, pixels=self.pixels)

    def _apply(self):
        """ASSUMES THERE ARE NO INCIDENT EDGES - raises valueerror if an edge will be
        removed by this operation
        Steps:
        - For each node
            set pixels to 0 if self.pixels is provided
        - Remove nodes from graph
        """
        if self.pixels is not None:
            self.tracks.set_pixels(
                self.pixels,
                [0] * len(self.pixels),
            )

        if isinstance(self.tracks, SolutionTracks):
            for node in self.nodes:
                self.tracks.track_id_to_node[self.tracks.get_track_id(node)].remove(node)

        self.tracks.graph.remove_nodes_from(self.nodes)
