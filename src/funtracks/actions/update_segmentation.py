import numpy as np

from ..project import Project
from ._base import TracksAction


class UpdateNodeSeg(TracksAction):
    """Action for updating the segmentation associated with node."""

    def __init__(
        self,
        project: Project,
        node: int,
        pixels: tuple[np.ndarray, ...],
        added: bool = True,
    ):
        """
        Args:
            tracks (Tracks): The project to update the segmenatation for
            nodes (list[Node]): The node with updated segmenatation
            pixels (list[SegMask]): The pixels that were updated for each node
            added (bool, optional): If the provided pixels were added (True) or deleted
                (False) from the node. Defaults to True.
        """
        super().__init__(project)
        self.node = node
        self.pixels = pixels
        self.added = added
        self._apply()

    def inverse(self):
        """Restore previous attributes"""
        return UpdateNodeSeg(
            self.project,
            self.node,
            pixels=self.pixels,
            added=not self.added,
        )

    def _apply(self):
        """Set new attributes"""
        value = self.node if self.added else 0
        self.project.set_pixels(self.pixels, value)
        # TODO: trigger computation of node and edge attrs
