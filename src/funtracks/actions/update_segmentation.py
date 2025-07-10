import numpy as np

from ..features._base import FeatureType
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
            project (Project): The project to update the segmenatation for
            node (int): The node with updated segmenatation
            pixels (tuple[np.ndarray]): The pixels that were updated for the node
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
        for feature in self.project.graph.features.get_features_to_compute(self):
            if feature.feature_type == FeatureType.NODE:
                value = feature.update(self.project, self.node)
                self.project.graph.set_feature_value(self.node, feature, value)
            elif feature.feature_type == FeatureType.EDGE:
                edges = [
                    (pred, self.node)
                    for pred in self.project.graph.predecessors(self.node)
                ]
                edges.extend(
                    [
                        (self.node, succ)
                        for succ in self.project.graph.successors(self.node)
                    ]
                )
                for edge in edges:
                    value = feature.update(self.project, edge)
                    self.project.graph.set_feature_value(edge, feature, value)
