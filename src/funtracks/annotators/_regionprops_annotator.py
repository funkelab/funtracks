from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from funtracks.actions.add_delete_node import AddNode
from funtracks.actions.update_segmentation import UpdateNodeSeg
from funtracks.features import (
    Area,
    Circularity,
    EllipsoidAxes,
    Feature,
    Perimeter,
    Position,
)

from ._graph_annotator import GraphAnnotator
from ._regionprops_extended import regionprops_extended

if TYPE_CHECKING:
    from funtracks.actions import TracksAction
    from funtracks.data_model import Tracks


class FeatureSpec(NamedTuple):
    """Specification for a regionprops feature.

    Attributes:
        key: The key to use in the graph attributes and feature dict
        feature: The Feature TypedDict definition
        regionprops_attr: The name of the corresponding regionprops attribute
    """

    key: str
    feature: Feature
    regionprops_attr: str


class RegionpropsAnnotator(GraphAnnotator):
    """A graph annotator using regionprops to extract node features from segmentations.

    The possible features include:
    - centroid (to use as node position)
    - area/volume
    - ellipsoid major/minor/semi-minor axes
    - circularity/sphericity
    - perimeter/surface area

    Defaults to computing all features, but individual ones can be turned off by changing
    the self.include value at the corresponding index to the feature in self.features.
    """

    @classmethod
    def can_annotate(cls, tracks) -> bool:
        """Check if this annotator can annotate the given tracks.

        Requires segmentation data to be present.

        Args:
            tracks: The tracks to check compatibility with

        Returns:
            True if tracks have segmentation, False otherwise
        """
        return tracks.segmentation is not None

    def __init__(
        self,
        tracks: Tracks,
        pos_key: str = "pos",
        area_key: str = "area",
        ellipse_axis_radii_key: str = "ellipse_axis_radii",
        circularity_key: str = "circularity",
        perimeter_key: str = "perimeter",
    ):
        self.pos_key = pos_key
        self.area_key = area_key
        self.ellipse_axis_radii_key = ellipse_axis_radii_key
        self.circularity_key = circularity_key
        self.perimeter_key = perimeter_key

        specs = RegionpropsAnnotator._build_feature_specs(
            tracks,
            pos_key,
            area_key,
            ellipse_axis_radii_key,
            circularity_key,
            perimeter_key,
        )
        feats = {spec.key: spec.feature for spec in specs}
        super().__init__(tracks, feats)
        # Build regionprops name mapping from specs
        self.regionprops_names = {spec.key: spec.regionprops_attr for spec in specs}

    @classmethod
    def _build_feature_specs(
        cls,
        tracks: Tracks,
        pos_key: str = "pos",
        area_key: str = "area",
        ellipse_axis_radii_key: str = "ellipse_axis_radii",
        circularity_key: str = "circularity",
        perimeter_key: str = "perimeter",
    ) -> list[FeatureSpec]:
        """Build feature specifications for all supported regionprops features.

        Single source of truth for feature definitions. Returns FeatureSpec objects
        that include the regionprops attribute mapping needed for computation.

        Args:
            tracks: The tracks to build feature specs for
            pos_key: The key to use for the position/centroid feature. Defaults to "pos".
            area_key: The key to use for the area feature. Defaults to "area".
            ellipse_axis_radii_key: The key to use for the ellipse axis radii feature.
                Defaults to "ellipse_axis_radii".
            circularity_key: The key to use for the circularity feature.
                Defaults to "circularity".
            perimeter_key: The key to use for the perimeter feature.
                Defaults to "perimeter".

        Returns:
            list[FeatureSpec]: List of feature specifications with key, feature,
                and regionprops attribute name. Empty list if no segmentation.
        """
        if not cls.can_annotate(tracks):
            return []
        return [
            FeatureSpec(
                pos_key, Position(axes=tracks.axis_names, recompute=True), "centroid"
            ),
            FeatureSpec(area_key, Area(ndim=tracks.ndim), "area"),
            # TODO: Add in intensity when image is passed
            # FeatureSpec("intensity", Intensity(ndim=tracks.ndim), "intensity"),
            FeatureSpec(ellipse_axis_radii_key, EllipsoidAxes(ndim=tracks.ndim), "axes"),
            FeatureSpec(circularity_key, Circularity(ndim=tracks.ndim), "circularity"),
            FeatureSpec(perimeter_key, Perimeter(ndim=tracks.ndim), "perimeter"),
        ]

    @classmethod
    def get_available_features(cls, tracks) -> dict[str, Feature]:
        """Get all features that can be computed by this annotator.

        Returns features with default keys. Custom keys can be specified at
        initialization time.

        Args:
            tracks: The tracks to get available features for

        Returns:
            Dictionary mapping feature keys to Feature definitions. Empty if no
            segmentation.
        """
        if not cls.can_annotate(tracks):
            return {}
        specs = RegionpropsAnnotator._build_feature_specs(tracks)
        return {spec.key: spec.feature for spec in specs}

    def compute(self, feature_keys: list[str] | None = None) -> None:
        """Compute the currently included features and add them to the tracks.

        Args:
            feature_keys: Optional list of specific feature keys to compute.
                If None, computes all currently active features. Keys not in
                self.features (not enabled) are ignored.

        Raises:
            ValueError: If the segmentation is missing from the tracks.
        """
        if self.tracks.segmentation is None:
            raise ValueError("Cannot compute regionprops features without segmentation.")

        keys_to_compute = self._filter_feature_keys(feature_keys)
        if not keys_to_compute:
            return

        seg = self.tracks.segmentation
        for t in range(seg.shape[0]):
            self._regionprops_update(seg[t], keys_to_compute)

    def _regionprops_update(self, seg_frame: np.ndarray, feature_keys: list[str]) -> None:
        """Perform the regionprops computation and update all feature values for a
        single frame of segmentation data.

        Args:
            seg_frame (np.ndarray): A 2D or 3D numpy array representing one time point
                of segmentation data.
            feature_keys: List of feature keys to compute (already filtered to enabled).
        """
        spacing = None if self.tracks.scale is None else tuple(self.tracks.scale[1:])
        for region in regionprops_extended(seg_frame, spacing=spacing):
            node = region.label
            for key in feature_keys:
                value = getattr(region, self.regionprops_names[key])
                if isinstance(value, tuple):
                    value = list(value)
                self.tracks._set_node_attr(node, key, value)

    def update(self, element: int | tuple[int, int], action: TracksAction):
        """Update the regionprops features for the given node.

        Only responds to AddNode and UpdateNodeSeg actions that affect segmentation.

        Args:
            element (int | tuple[int, int]): The node to update. Edges are ignored.
            action (TracksAction): The action that triggered this update

        Raises:
            ValueError: If the tracks do not have a segmentation
        """
        # Only update for actions that change segmentation
        if not isinstance(action, (AddNode, UpdateNodeSeg)):
            return

        if self.tracks.segmentation is None:
            raise ValueError("Cannot update regionprops features without segmentation.")

        if isinstance(element, tuple):
            return  # Silently ignore edges

        keys_to_compute = list(self.features.keys())
        if not keys_to_compute:
            return

        time = self.tracks.get_time(element)
        seg_frame = self.tracks.segmentation[time]
        masked_frame = np.where(seg_frame == element, element, 0)

        if np.max(masked_frame) == 0:
            warnings.warn(
                f"Cannot find label {element} in frame {time}: "
                "updating regionprops values to None",
                stacklevel=2,
            )
            for key in keys_to_compute:
                value = None
                self.tracks._set_node_attr(element, key, value)
        else:
            self._regionprops_update(masked_frame, keys_to_compute)
