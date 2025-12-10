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
    from funtracks.actions import BasicAction
    from funtracks.data_model import Tracks

DEFAULT_POS_KEY = "pos"
DEFAULT_AREA_KEY = "area"
DEFAULT_ELLIPSE_AXIS_KEY = "ellipse_axis_radii"
DEFAULT_CIRCULARITY_KEY = "circularity"
DEFAULT_PERIMETER_KEY = "perimeter"


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
        pos_key: str | None = DEFAULT_POS_KEY,
    ):
        self.pos_key: str = pos_key if pos_key is not None else DEFAULT_POS_KEY
        self.area_key = DEFAULT_AREA_KEY
        self.ellipse_axis_radii_key = DEFAULT_ELLIPSE_AXIS_KEY
        self.circularity_key = DEFAULT_CIRCULARITY_KEY
        self.perimeter_key = DEFAULT_PERIMETER_KEY

        specs = RegionpropsAnnotator._define_features(
            tracks.ndim,
        )
        # update position key in spec
        if self.pos_key != DEFAULT_POS_KEY:
            for feat in specs:
                if feat.key == DEFAULT_POS_KEY:
                    specs.remove(feat)
                    new_feat = FeatureSpec(
                        self.pos_key, feat.feature, feat.regionprops_attr
                    )
                    specs.append(new_feat)
                    break

        feats = {spec.key: spec.feature for spec in specs}
        super().__init__(tracks, feats)
        # Build regionprops name mapping from specs
        self.regionprops_names = {spec.key: spec.regionprops_attr for spec in specs}

    @classmethod
    def _define_features(
        cls,
        ndim: int,
    ) -> list[FeatureSpec]:
        """Define all supported regionprops features along with keys and function names.

        Single source of truth for feature definitions. Returns FeatureSpec objects
        that include the regionprops attribute mapping needed for computation.

        Args:
            ndim: Total number of dimensions including time (3 or 4)

        Returns:
            list[FeatureSpec]: List of feature specifications with key, feature,
                and regionprops attribute name.
        """
        # Derive axis names from ndim (spatial dimensions only, no time)
        # Default to 3D when ndim is None to enable matching all position columns
        axis_names = ["z", "y", "x"] if ndim is None or ndim == 4 else ["y", "x"]

        return [
            FeatureSpec(DEFAULT_POS_KEY, Position(axes=axis_names), "centroid"),
            FeatureSpec(DEFAULT_AREA_KEY, Area(ndim=ndim), "area"),
            # TODO: Add in intensity when image is passed
            # FeatureSpec("intensity", Intensity(ndim=ndim), "intensity"),
            FeatureSpec(DEFAULT_ELLIPSE_AXIS_KEY, EllipsoidAxes(ndim=ndim), "axes"),
            FeatureSpec(DEFAULT_CIRCULARITY_KEY, Circularity(ndim=ndim), "circularity"),
            FeatureSpec(DEFAULT_PERIMETER_KEY, Perimeter(ndim=ndim), "perimeter"),
        ]

    @classmethod
    def get_available_features(cls, ndim: int = 3) -> dict[str, Feature]:
        """Get all features that can be computed by this annotator.

        Returns features with default keys. Custom keys can be specified at
        initialization time.

        Args:
            ndim: Total number of dimensions including time (3 or 4). Defaults to 3.

        Returns:
            Dictionary mapping feature keys to Feature definitions.
        """
        specs = RegionpropsAnnotator._define_features(ndim)
        return {spec.key: spec.feature for spec in specs}

    def compute(self, feature_keys: list[str] | None = None) -> None:
        """Compute the currently included features and add them to the tracks.

        Args:
            feature_keys: Optional list of specific feature keys to compute.
                If None, computes all currently active features. Keys not in
                self.features (not enabled) are ignored.
        """
        # Can only compute features if segmentation is present
        if self.tracks.segmentation is None:
            return

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
            # Skip labels that aren't nodes in the graph (e.g., unselected detections)
            if node not in self.tracks.graph:
                continue
            for key in feature_keys:
                value = getattr(region, self.regionprops_names[key])
                if isinstance(value, tuple):
                    value = list(value)
                self.tracks._set_node_attr(node, key, value)

    def update(self, action: BasicAction):
        """Update the regionprops features based on the action.

        Only responds to AddNode and UpdateNodeSeg actions that affect segmentation.

        Args:
            action (BasicAction): The action that triggered this update
        """
        # Only update for actions that change segmentation
        if not isinstance(action, (AddNode, UpdateNodeSeg)):
            return

        # Can only compute features if segmentation is present
        if self.tracks.segmentation is None:
            return

        # Get the node from the action
        node = action.node

        keys_to_compute = list(self.features.keys())
        if not keys_to_compute:
            return

        time = self.tracks.get_time(node)
        seg_frame = self.tracks.segmentation[time]
        masked_frame = np.where(seg_frame == node, node, 0)

        if np.max(masked_frame) == 0:
            warnings.warn(
                f"Cannot find label {node} in frame {time}: "
                "updating regionprops values to None",
                stacklevel=2,
            )
            for key in keys_to_compute:
                value = None
                self.tracks._set_node_attr(node, key, value)
        else:
            self._regionprops_update(masked_frame, keys_to_compute)

    def change_key(self, old_key: str, new_key: str) -> None:
        """Rename a feature key in this annotator, and related mappings.

        Overrides base implementation to also update the regionprops name mapping.

        Args:
            old_key: Existing key to rename.
            new_key: New key to replace it with.

        Raises:
            KeyError: If old_key does not exist.
        """
        # Call base implementation to update all_features
        super().change_key(old_key, new_key)

        # Update regionprops-specific name mapping
        if old_key in self.regionprops_names:
            rp_name = self.regionprops_names.pop(old_key)
            self.regionprops_names[new_key] = rp_name
