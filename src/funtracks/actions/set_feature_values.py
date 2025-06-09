from typing import Any

from ..features._base import Feature
from ..project import Project
from ._base import TracksAction


class SetFeatureValues(TracksAction):
    def __init__(
        self,
        project: Project,
        element: int | tuple[int, int],
        provided_features: dict[Feature, Any],
    ):
        super().__init__(project)
        self.element = element
        # TODO: check that the features are valid (not computed, correct element type)
        self.provided_features = provided_features

        self.original_features = {
            feature: self.project.cand_graph.get_feature_value(self.element, feature)
            for feature in provided_features
        }
        self._apply()

    def inverse(self):
        """Delete edges"""
        return SetFeatureValues(self.project, self.element, self.original_features)

    def _apply(self):
        for feature, value in self.provided_features.items():
            self.project.cand_graph.set_feature_value(self.element, feature, value)
