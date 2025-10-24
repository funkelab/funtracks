from __future__ import annotations

from typing import TYPE_CHECKING

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
        self.tracks.set_pixels(self.pixels, value)
        self.tracks.notify_annotators(self)
