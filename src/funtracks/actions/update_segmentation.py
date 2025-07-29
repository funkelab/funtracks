from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from funtracks.data_model.graph_attributes import NodeAttr

from ._base import TracksAction

if TYPE_CHECKING:
    from collections.abc import Iterable

    from funtracks.data_model.tracks import Node, SegMask, Tracks


class UpdateNodeSegs(TracksAction):
    """Action for updating the segmentation associated with nodes. Cannot mix adding
    and removing pixels from segmentation: the added flag applies to all nodes"""

    def __init__(
        self,
        tracks: Tracks,
        nodes: Iterable[Node],
        pixels: Iterable[SegMask],
        added: bool = True,
    ):
        """
        Args:
            tracks (Tracks): The tracks to update the segmenatations for
            nodes (list[Node]): The nodes with updated segmenatations
            pixels (list[SegMask]): The pixels that were updated for each node
            added (bool, optional): If the provided pixels were added (True) or deleted
                (False) from all nodes. Defaults to True. Cannot mix adding and deleting
                pixels in one action.
        """
        super().__init__(tracks)
        self.nodes = nodes
        self.pixels = pixels
        self.added = added
        self._apply()

    def inverse(self):
        """Restore previous attributes"""
        return UpdateNodeSegs(
            self.tracks,
            self.nodes,
            pixels=self.pixels,
            added=not self.added,
        )

    def _apply(self):
        """Set new attributes"""
        times = self.tracks.get_times(self.nodes)
        values = self.nodes if self.added else [0 for _ in self.nodes]
        self.tracks.set_pixels(self.pixels, values)
        computed_attrs = self.tracks._compute_node_attrs(self.nodes, times)
        positions = np.array(computed_attrs[NodeAttr.POS.value])
        self.tracks.set_positions(self.nodes, positions)
        self.tracks._set_nodes_attr(
            self.nodes, NodeAttr.AREA.value, computed_attrs[NodeAttr.AREA.value]
        )

        incident_edges = list(self.tracks.graph.in_edges(self.nodes)) + list(
            self.tracks.graph.out_edges(self.nodes)
        )
        for edge in incident_edges:
            new_edge_attrs = self.tracks._compute_edge_attrs([edge])
            self.tracks._set_edge_attributes([edge], new_edge_attrs)
