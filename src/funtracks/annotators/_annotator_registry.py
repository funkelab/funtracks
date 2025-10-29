from __future__ import annotations

from typing import TYPE_CHECKING

from ._graph_annotator import GraphAnnotator

if TYPE_CHECKING:
    from funtracks.actions import BasicAction
    from funtracks.features import Feature


class AnnotatorRegistry(list[GraphAnnotator]):
    """A list of annotators with coordinated operations.

    Inherits from list[GraphAnnotator], so can be used directly as a list.
    Provides coordinated compute/update/enable/disable operations across all annotators.

    Example:
        annotators = AnnotatorRegistry([
            RegionpropsAnnotator(tracks, pos_key="centroid"),
            EdgeAnnotator(tracks),
            TrackAnnotator(tracks, tracklet_key="track_id"),
        ])

        # Can use as a list
        annotators.append(MyCustomAnnotator(tracks))

        # Coordinated operations
        annotators.enable_features(["area", "iou"])
        annotators.compute()
    """

    def __init__(self, annotators: list[GraphAnnotator]):
        """Initialize with a list of annotators.

        Args:
            annotators: List of instantiated annotator objects
        """
        super().__init__(annotators)

    @property
    def all_features(self) -> dict[str, tuple[Feature, bool]]:
        """Dynamically aggregate all_features from all annotators.

        Returns:
            Dictionary mapping feature keys to (Feature, is_enabled) tuples
        """
        aggregated = {}
        for annotator in self:
            aggregated.update(annotator.all_features)
        return aggregated

    @property
    def features(self) -> dict[str, Feature]:
        """Get all currently enabled features from all annotators.

        Returns:
            Dictionary mapping feature keys to Feature definitions (only enabled features)
        """
        aggregated = {}
        for annotator in self:
            aggregated.update(annotator.features)
        return aggregated

    def compute(self, feature_keys: list[str] | None = None) -> None:
        """Compute features across all annotators.

        Args:
            feature_keys: Optional list of specific feature keys to compute.
                If None, computes all currently active features.
        """
        for annotator in self:
            annotator.compute(feature_keys)

    def update(self, action: BasicAction) -> None:
        """Update features across all annotators based on the action.

        Args:
            action: The action that triggered this update
        """
        for annotator in self:
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
        for annotator in self:
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
        for annotator in self:
            annotator.disable_features(keys)
