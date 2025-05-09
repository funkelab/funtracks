from __future__ import annotations

from typing import Any

import numpy as np

from ..features._base import Feature, FeatureType
from ..project import Project
from ._base import TracksAction


class AddNode(TracksAction):
    """Action for adding new nodes. If a segmentation should also be added, the
    pixels for each node should be provided. The label to set the pixels will
    be taken from the node id. The existing pixel values are assumed to be
    zero - you must explicitly update any other segmentations that were overwritten
    using an UpdateNodes action if you want to be able to undo the action.
    """

    def __init__(
        self,
        project: Project,
        node: int,
        attributes: dict[Feature, Any],
        pixels: tuple[np.ndarray, ...] | None = None,
    ):
        super().__init__(project)
        self.node = node
        self.project.cand_graph.features.validate_new_node_features(attributes)
        self.attributes = attributes
        self.pixels = pixels
        self._apply()

    def inverse(self):
        """Invert the action to delete nodes instead"""
        return DeleteNode(self.project, self.node)

    def _apply(self):
        """Apply the action, and set segmentation if provided in self.pixels"""
        if self.pixels is not None:
            self.project.set_pixels(self.pixels, self.node)
        self.project.cand_graph.add_node(self.node, self.attributes)
        # TODO: somehow  trigger recomputation of position, area


class DeleteNode(TracksAction):
    """Action of deleting existing nodes
    If the tracks contain a segmentation, this action also constructs a reversible
    operation for setting involved pixels to zero
    """

    def __init__(
        self,
        project: Project,
        node: int,
        pixels: tuple[np.ndarray, ...] | None = None,
    ):
        super().__init__(project)
        self.node = node
        self.attributes = {
            feature: self.project.cand_graph.get_feature_value(feature)
            for feature in self.project.cand_graph.features
            if feature.feature_type == FeatureType.NODE
        }
        self.pixels = self.project.get_pixels(node) if pixels is not None else pixels
        self._apply()

    def inverse(self):
        """Invert this action, and provide inverse segmentation operation if given"""
        return AddNode(self.project, self.node, self.attributes, pixels=self.pixels)

    def _apply(self):
        """
        Steps:
        - delete incident edges
        - set pixels to 0 if self.pixels is provided
        - Pin nodes to 0 in cand graph and remove from solution (hopefully not by triggering resolve)
        """
        if self.pixels is not None:
            self.project.set_pixels(self.pixels, 0)
        features = self.project.cand_graph.features
        self.project.cand_graph.set_feature_value(self.node, features.node_pin, False)
        # TODO: Somehow remove from solution graph
