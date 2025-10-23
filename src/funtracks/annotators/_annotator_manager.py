from __future__ import annotations

from typing import TYPE_CHECKING

from funtracks.features import Feature

from ._edge_annotator import EdgeAnnotator
from ._graph_annotator import GraphAnnotator
from ._regionprops_annotator import RegionpropsAnnotator
from ._track_annotator import TrackAnnotator

if TYPE_CHECKING:
    import numpy as np

    from funtracks.data_model import Tracks


class AnnotatorManager:
    """Manages annotators for feature computation on a Tracks instance.

    Coordinates between RegionpropsAnnotator, EdgeAnnotator, and TrackAnnotator to
    provide a unified interface for feature computation and updates.

    Attributes:
        annotators: Dictionary mapping annotator names to annotator instances
    """

    @classmethod
    def get_managed_features(
        cls,
        segmentation: np.ndarray | None,
        ndim: int,
        axis_names: list[str],
        is_solution_tracks: bool = False,
    ) -> dict[str, Feature]:
        """Get all features that would be managed by annotators.

        This is a class method so it can be called before creating a Tracks instance,
        allowing Tracks to build a complete FeatureDict from the start.

        Returns features with default keys. Custom keys can be specified when
        creating annotator instances.

        Args:
            segmentation: The segmentation array (or None)
            ndim: Number of dimensions (3 or 4)
            axis_names: Names of spatial axes
            is_solution_tracks: Whether this is for SolutionTracks

        Returns:
            Dictionary mapping feature keys to Feature definitions
        """
        features: dict[str, Feature] = {}

        # Get features from each annotator (with default keys)
        features.update(
            RegionpropsAnnotator.get_available_features(segmentation, ndim, axis_names)
        )
        features.update(EdgeAnnotator.get_available_features(segmentation))
        features.update(TrackAnnotator.get_available_features(is_solution_tracks))

        return features

    def __init__(self, tracks: Tracks, existing_features: list[str] | None = None):
        """Initialize AnnotatorManager with all appropriate annotators.

        Only includes features specified in existing_features in the FeatureDict.
        Other features remain in all_features but are disabled until explicitly enabled.

        Args:
            tracks: The Tracks instance to manage features for
            existing_features: Feature keys already computed on the graph.
                These will be included in FeatureDict and marked as active.
                All other features start disabled. Defaults to empty list.
        """
        self.annotators: dict[str, GraphAnnotator] = {}
        self.features = tracks.features  # Reference to FeatureDict for enable/disable

        # Import here to avoid circular imports
        from funtracks.data_model import SolutionTracks

        # Always try to create all annotators
        # RegionpropsAnnotator requires segmentation
        if tracks.segmentation is not None:
            rp_annotator = RegionpropsAnnotator(tracks)
            self.annotators["regionprops"] = rp_annotator
            tracks.features.position_key = rp_annotator.pos_key

        # EdgeAnnotator requires segmentation
        if tracks.segmentation is not None:
            self.annotators["edges"] = EdgeAnnotator(tracks)

        # TrackAnnotator requires SolutionTracks
        if isinstance(tracks, SolutionTracks):
            track_annotator = TrackAnnotator(tracks)
            self.annotators["tracks"] = track_annotator
            tracks.features.tracklet_key = track_annotator.tracklet_key

        # Determine which features to include
        features_to_include = set(existing_features or [])

        # Include only existing features in tracks.features and disable the rest
        for annotator in self.annotators.values():
            for key, (feature, _) in annotator.all_features.items():
                if key in features_to_include:
                    # Include this feature (already computed)
                    if key not in tracks.features:
                        tracks.features[key] = feature
                else:
                    # Disable this feature (not yet computed)
                    annotator.remove_feature(key)

    # ========== Feature Computation ==========

    def update(self, element: int | tuple[int, int]) -> None:
        """Update features for a specific node or edge.

        Broadcasts the update to all annotators. Each annotator decides whether
        to handle the element based on its type and responsibilities.

        Args:
            element: Either a node ID (int) or edge tuple (int, int)
        """
        for annotator in self.annotators.values():
            annotator.update(element)

    def recompute_tracks(self) -> None:
        """Recompute track-level features (tracklet_id, lineage_id).

        This is needed when the graph structure changes (edges added/removed).
        TrackAnnotator doesn't support single-element updates.
        """
        if "tracks" in self.annotators:
            self.annotators["tracks"].compute()

    # ========== Feature Introspection ==========

    def get_available_features(self) -> dict[str, Feature]:
        """Get all features that can be computed across all annotators."""
        available = {}
        for annotator in self.annotators.values():
            for key, (feature, _) in annotator.all_features.items():
                available[key] = feature
        return available

    def get_active_features(self) -> dict[str, Feature]:
        """Get all currently active (included) features."""
        active = {}
        for annotator in self.annotators.values():
            active.update(annotator.features)
        return active

    def get_feature_source(self, feature_key: str) -> str | None:
        """Get the name of the annotator that provides a feature.

        Args:
            feature_key: The feature key to look up

        Returns:
            The annotator name ("regionprops", "edges", or "tracks") or None
        """
        for name, annotator in self.annotators.items():
            if feature_key in annotator.all_features:
                return name
        return None

    # ========== Feature Enable/Disable ==========

    def enable_features(self, feature_keys: list[str]) -> None:
        """Enable multiple features for computation efficiently.

        Groups features by annotator and computes them together for efficiency.
        Adds features to FeatureDict and computes their values.

        Args:
            feature_keys: List of feature keys to enable

        Raises:
            KeyError: If any feature is not available
        """
        # Group features by annotator
        features_by_annotator: dict[str, list[str]] = {}
        for key in feature_keys:
            annotator_name = self.get_feature_source(key)
            if annotator_name is None:
                raise KeyError(f"Feature '{key}' not available")

            if annotator_name not in features_by_annotator:
                features_by_annotator[annotator_name] = []
            features_by_annotator[annotator_name].append(key)

        # Enable and compute features for each annotator
        for annotator_name, keys in features_by_annotator.items():
            annotator = self.annotators[annotator_name]

            # Add features to annotator's active features and FeatureDict
            for key in keys:
                annotator.add_feature(key)
                if key not in self.features:
                    feature, _ = annotator.all_features[key]
                    self.features[key] = feature

            # Compute all features for this annotator at once
            annotator.compute(keys)

    def disable_features(self, feature_keys: list[str]) -> None:
        """Disable multiple features from computation.

        Removes features from FeatureDict and marks them as inactive in annotators.

        Args:
            feature_keys: List of feature keys to disable

        Raises:
            KeyError: If any feature is not available
        """
        for key in feature_keys:
            annotator_name = self.get_feature_source(key)
            if annotator_name is None:
                raise KeyError(f"Feature '{key}' not available")

            annotator = self.annotators[annotator_name]

            # Remove feature from annotator's active features
            annotator.remove_feature(key)

            # Remove feature from FeatureDict
            if key in self.features:
                del self.features[key]
