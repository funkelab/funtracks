from __future__ import annotations

from typing import TYPE_CHECKING

from ._edge_annotator import EdgeAnnotator
from ._graph_annotator import GraphAnnotator
from ._regionprops_annotator import RegionpropsAnnotator
from ._track_annotator import TrackAnnotator

if TYPE_CHECKING:
    from funtracks.actions import BasicAction
    from funtracks.data_model import Tracks
    from funtracks.features import Feature


class AnnotatorRegistry(GraphAnnotator):
    """A composite annotator that manages all available annotators.

    This class acts as a registry of all annotator types and automatically
    instantiates the appropriate ones based on the Tracks instance and the
    annotators' requirements (segmentation, SolutionTracks, etc.). It then
    broadcasts operations to all registered annotators.

    Attributes:
        annotators: List of instantiated annotators
        ANNOTATOR_CLASSES: List of all available annotator classes
    """

    # Registry of all available annotator classes
    ANNOTATOR_CLASSES: list[type[GraphAnnotator]] = [
        RegionpropsAnnotator,
        EdgeAnnotator,
        TrackAnnotator,
    ]

    @classmethod
    def get_managed_features(cls, tracks: Tracks) -> dict[str, Feature]:
        """Get all features that would be managed by annotators.

        This is a class method so it can be called before creating a Tracks instance,
        allowing Tracks to build a complete FeatureDict from the start.

        Args:
            tracks: The tracks to get managed features for

        Returns:
            Dictionary mapping feature keys to Feature definitions
        """
        features: dict[str, Feature] = {}

        # Get features from each annotator class
        for annotator_class in cls.ANNOTATOR_CLASSES:
            features.update(annotator_class.get_available_features(tracks))  # type: ignore[attr-defined]

        return features

    def __init__(self, tracks: Tracks):
        """Initialize the registry and create appropriate annotators.

        Args:
            tracks: The Tracks instance to create annotators for
        """
        # Don't call super().__init__() yet - we need to collect features first
        self.tracks = tracks
        self.annotators: list[GraphAnnotator] = []

        # Instantiate annotators based on their can_annotate() method
        for annotator_class in self.ANNOTATOR_CLASSES:
            if annotator_class.can_annotate(tracks):  # type: ignore[attr-defined]
                self.annotators.append(annotator_class(tracks))  # type: ignore[call-arg]

    @property
    def all_features(self) -> dict[str, tuple[Feature, bool]]:  # type: ignore[override]
        """Dynamically aggregate all_features from child annotators.

        Returns:
            Dictionary mapping feature keys to (Feature, is_enabled) tuples
        """
        aggregated = {}
        for annotator in self.annotators:
            aggregated.update(annotator.all_features)
        return aggregated

    def compute(self, feature_keys: list[str] | None = None) -> None:
        """Compute features across all annotators.

        Args:
            feature_keys: Optional list of specific feature keys to compute.
                If None, computes all currently active features.
        """
        for annotator in self.annotators:
            annotator.compute(feature_keys)

    def update(self, action: BasicAction) -> None:
        """Update features across all annotators based on the action.

        Args:
            action: The action that triggered this update
        """
        for annotator in self.annotators:
            annotator.update(action)

    def enable_features(self, keys: list[str]) -> None:
        """Enable features across all annotators.

        Args:
            keys: List of feature keys to enable

        Raises:
            KeyError: If any feature keys are not available
        """
        # Validate first - fail before making any changes
        available = self.all_features
        not_found = [k for k in keys if k not in available]
        if not_found:
            raise KeyError(f"Features not available: {not_found}")

        # All features exist - proceed with enabling
        for annotator in self.annotators:
            annotator.enable_features(keys)

    def disable_features(self, keys: list[str]) -> None:
        """Disable features across all annotators.

        Args:
            keys: List of feature keys to disable

        Raises:
            KeyError: If any feature keys are not available
        """
        # Validate first - fail before making any changes
        available = self.all_features
        not_found = [k for k in keys if k not in available]
        if not_found:
            raise KeyError(f"Features not available: {not_found}")

        # All features exist - proceed with disabling
        for annotator in self.annotators:
            annotator.disable_features(keys)
