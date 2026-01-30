from __future__ import annotations

from typing import TYPE_CHECKING

import tracksdata as td

from funtracks.utils.tracksdata_utils import pixels_to_td_mask

from ._base import BasicAction

if TYPE_CHECKING:
    from funtracks.data_model import Tracks
    from funtracks.data_model.tracks import Node, SegMask


class UpdateNodeSeg(BasicAction):
    """Action for updating the segmentation associated with a node.

    New nodes call AddNode with pixels instead of this action.
    """

    def __init__(
        self,
        tracks: Tracks,
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
                (False) from this node. Defaults to True
        """
        super().__init__(tracks)
        self.node = node
        self.pixels = pixels
        self.added = added
        self._apply()

    def inverse(self) -> BasicAction:
        """Restore previous attributes"""
        return UpdateNodeSeg(
            self.tracks,
            self.node,
            pixels=self.pixels,
            added=not self.added,
        )

    def _apply(self) -> None:
        """Set new attributes"""
        value = self.node if self.added else 0

        mask_new, area_new = pixels_to_td_mask(
            self.pixels, self.tracks.ndim, self.tracks.scale
        )

        if value == 0:
            # val=0 means deleting the pixels from the mask
            mask_old = self.tracks.graph[self.node][td.DEFAULT_ATTR_KEYS.MASK]
            mask_subtracted = mask_old.__isub__(mask_new)
            self.tracks.graph.update_node_attrs(
                attrs={
                    td.DEFAULT_ATTR_KEYS.MASK: [mask_subtracted],
                    td.DEFAULT_ATTR_KEYS.BBOX: [mask_subtracted.bbox],
                },
                node_ids=[self.node],
            )

        elif self.tracks.graph.has_node(value):
            # if node already exists:
            mask_old = self.tracks.graph[value][td.DEFAULT_ATTR_KEYS.MASK]
            mask_combined = mask_old.__or__(mask_new)
            self.tracks.graph.update_node_attrs(
                attrs={
                    td.DEFAULT_ATTR_KEYS.MASK: [mask_combined],
                    td.DEFAULT_ATTR_KEYS.BBOX: [mask_combined.bbox],
                },
                node_ids=[value],
            )

        # Invalidate cache for affected chunks
        self.tracks._update_segmentation_cache(self.pixels)

        self.tracks.notify_annotators(self)
