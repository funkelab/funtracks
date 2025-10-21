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

        Args:
            segmentation: The segmentation array (or None)
            ndim: Number of dimensions (3 or 4)
            axis_names: Names of spatial axes
            is_solution_tracks: Whether this is for SolutionTracks

        Returns:
            Dictionary mapping feature keys to Feature definitions
        """
        from funtracks.features import (
            Area,
            Centroid,
            Circularity,
            EllipsoidAxes,
            Perimeter,
        )

        features: dict[str, Feature] = {}

        # RegionpropsAnnotator and EdgeAnnotator features (require segmentation)
        if segmentation is not None:
            # Regionprops features
            features["pos"] = Centroid(axes=axis_names)
            features["area"] = Area(ndim=ndim)
            features["ellipse_axis_radii"] = EllipsoidAxes(ndim=ndim)
            features["circularity"] = Circularity(ndim=ndim)
            features["perimeter"] = Perimeter(ndim=ndim)

            # Edge features
            from ._edge_annotator import IoU

            features["IoU"] = IoU()

        # TrackAnnotator features (require SolutionTracks)
        if is_solution_tracks:
            from ._track_annotator import LineageID, TrackletID

            features["tracklet_id"] = TrackletID()
            features["lineage_id"] = LineageID()

        return features

    def __init__(self, tracks: Tracks):
        """Initialize AnnotatorManager with all appropriate annotators.

        Adds managed features to the tracks.features FeatureDict and updates
        the position_key if needed.

        Args:
            tracks: The Tracks instance to manage features for
        """
        self.annotators: dict[str, GraphAnnotator] = {}

        # Import here to avoid circular imports
        from funtracks.data_model import SolutionTracks

        # Always try to create all annotators
        # RegionpropsAnnotator requires segmentation
        if tracks.segmentation is not None:
            self.annotators["regionprops"] = RegionpropsAnnotator(tracks)

        # EdgeAnnotator requires segmentation
        if tracks.segmentation is not None:
            self.annotators["edges"] = EdgeAnnotator(tracks)

        # TrackAnnotator requires SolutionTracks
        if isinstance(tracks, SolutionTracks):
            self.annotators["tracks"] = TrackAnnotator(tracks)

        # Add managed features to tracks.features
        for annotator in self.annotators.values():
            for key, (feature, _) in annotator.all_features.items():
                if key not in tracks.features:
                    tracks.features[key] = feature
                # Update position_key if this is the position feature
                if key == "pos" and tracks.features.position_key is None:
                    tracks.features.position_key = "pos"

    # ========== Feature Computation ==========

    def compute_all(self) -> None:
        """Compute all features from all annotators.

        Features are already registered in tracks.features during initialization.
        This method computes the actual values and stores them as node/edge attributes.
        """
        for annotator in self.annotators.values():
            annotator.compute()

    def update(self, element: int | tuple[int, int]) -> None:
        """Update features for a specific node or edge.

        Routes the update to the appropriate annotator(s) based on element type.

        Args:
            element: Either a node ID (int) or edge tuple (int, int)
        """
        if isinstance(element, tuple):
            # Edge update
            if "edges" in self.annotators:
                self.annotators["edges"].update(element)
        else:
            # Node update
            if "regionprops" in self.annotators:
                self.annotators["regionprops"].update(element)

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

    def enable_feature(self, feature_key: str, compute: bool = False) -> None:
        """Enable a feature for computation.

        Args:
            feature_key: The key of the feature to enable
            compute: If True, immediately compute the feature

        Raises:
            KeyError: If the feature is not available
        """
        annotator_name = self.get_feature_source(feature_key)
        if annotator_name is None:
            raise KeyError(f"Feature '{feature_key}' not available")

        annotator = self.annotators[annotator_name]
        annotator.add_feature(feature_key)

        if compute:
            annotator.compute()

    def disable_feature(self, feature_key: str) -> None:
        """Disable a feature from computation.

        Args:
            feature_key: The key of the feature to disable

        Raises:
            KeyError: If the feature is not available
        """
        annotator_name = self.get_feature_source(feature_key)
        if annotator_name is None:
            raise KeyError(f"Feature '{feature_key}' not available")

        annotator = self.annotators[annotator_name]
        annotator.remove_feature(feature_key)
