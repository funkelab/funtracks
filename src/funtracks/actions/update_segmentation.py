from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from funtracks.data_model.graph_attributes import NodeAttr

from ._base import TracksAction

if TYPE_CHECKING:
    from funtracks.data_model import SolutionTracks
    from funtracks.data_model.tracks import Node, SegMask


class UpdateNodeSeg(TracksAction):
    """Action for updating the segmentation associated with nodes. Cannot mix adding
    and removing pixels from segmentation: the added flag applies to all nodes
    """

    def __init__(
        self,
        tracks: SolutionTracks,
        node: Node,
        pixels: SegMask,
        added: bool = True,
    ):
        """
        Args:
            tracks (Tracks): The tracks to update the segmenatations for
            node (Node): The node with updated segmenatation
            pixels (SegMask): The pixels that were updated for the node
            added (bool, optional): If the provided pixels were added (True) or deleted
                (False) from all nodes. Defaults to True. Cannot mix adding and deleting
                pixels in one action.
        """
        super().__init__(tracks)
        self.node = node
        self.pixels = pixels
        self.added = added
        self._apply()

    def inverse(self) -> TracksAction:
        """Restore previous attributes"""
        return UpdateNodeSeg(
            self.tracks,
            self.node,
            pixels=self.pixels,
            added=not self.added,
        )

    def _apply(self) -> None:
        """Set new attributes"""
        times = self.tracks.get_time(self.node)
        value = self.node if self.added else 0
        self.tracks.set_pixels(self.pixels, value)
        computed_attrs = self.tracks._compute_node_attrs(self.node, times)
        position = np.array(computed_attrs[NodeAttr.POS.value])
        self.tracks.set_position(self.node, position)
        self.tracks._set_node_attr(
            self.node, NodeAttr.AREA.value, computed_attrs[NodeAttr.AREA.value]
        )

        incident_edges = list(self.tracks.graph.in_edges(self.node)) + list(
            self.tracks.graph.out_edges(self.node)
        )
        for edge in incident_edges:
            new_edge_attrs = self.tracks._compute_edge_attrs(edge)
            self.tracks._set_edge_attributes(edge, new_edge_attrs)
