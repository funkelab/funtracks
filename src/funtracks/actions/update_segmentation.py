from __future__ import annotations

from typing import TYPE_CHECKING

import tracksdata as td
from tracksdata.nodes._mask import Mask

from ._base import BasicAction

if TYPE_CHECKING:
    from funtracks.data_model import Tracks
    from funtracks.data_model.tracks import Node


class UpdateNodeSeg(BasicAction):
    """Action for updating the segmentation associated with a node.

    New nodes call AddNode with mask instead of this action.
    """

    def __init__(
        self,
        tracks: Tracks,
        node: Node,
        mask: Mask,
        added: bool = True,
    ):
        """
        Args:
            tracks (Tracks): The tracks to update the segmentations for
            node (Node): The node with updated segmentation
            mask (Mask): The mask that was updated for the node
            added (bool, optional): If the provided mask were added (True) or deleted
                (False) from this node. Defaults to True
        """
        super().__init__(tracks)
        self.node = node
        self.mask = mask
        self.added = added
        self._apply()

    def inverse(self) -> BasicAction:
        """Restore previous attributes"""
        return UpdateNodeSeg(
            self.tracks,
            self.node,
            mask=self.mask,
            added=not self.added,
        )

    def _apply(self) -> None:
        """Set new attributes"""
        value = self.node if self.added else 0

        mask_new = self.mask

        if value == 0:
            # val=0 means deleting (part of) the mask
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
        time = self.tracks.get_time(self.node)
        self.tracks._update_segmentation_cache(mask=mask_new, time=time)

        self.tracks.notify_annotators(self)
