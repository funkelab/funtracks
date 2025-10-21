from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from funtracks.features import (
    Area,
    Centroid,
    Circularity,
    EllipsoidAxes,
    Feature,
    Perimeter,
)

from ._graph_annotator import GraphAnnotator
from ._regionprops_extended import regionprops_extended

if TYPE_CHECKING:
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

    def __init__(self, tracks: Tracks):
        specs = RegionpropsAnnotator.get_feature_specs(tracks)
        feats = {spec.key: spec.feature for spec in specs}
        super().__init__(tracks, feats)
        # Build regionprops name mapping from specs
        self.regionprops_names = {spec.key: spec.regionprops_attr for spec in specs}

    @staticmethod
    def get_feature_specs(tracks: Tracks) -> list[FeatureSpec]:
        """Get specifications for all supported regionprops features.

        Args:
            tracks (Tracks): The tracks to get feature specs for.

        Returns:
            list[FeatureSpec]: List of feature specifications with key, feature,
                and regionprops attribute name. Empty list if no segmentation.
        """
        if tracks.segmentation is None:
            return []
        ndim = tracks.ndim
        return [
            FeatureSpec("pos", Centroid(axes=tracks.axis_names), "centroid"),
            FeatureSpec("area", Area(ndim=ndim), "area"),
            # TODO: Add in intensity when image is passed
            # FeatureSpec("intensity", Intensity(ndim=ndim), "intensity"),
            FeatureSpec("ellipse_axis_radii", EllipsoidAxes(ndim=ndim), "axes"),
            FeatureSpec("circularity", Circularity(ndim=ndim), "circularity"),
            FeatureSpec("perimeter", Perimeter(ndim=ndim), "perimeter"),
        ]

    def compute(self, add_to_set=False) -> None:
        """Compute the currently included features and add them to the tracks.

        Args:
            add_to_set (bool, optional): Whether to add the Features to the Tracks
            FeatureSet. Defaults to False. Should usually be set to True on the initial
            computation, but False on subsequent re-computations.

        Raises:
            ValueError: If the segmentation is missing from the tracks.
        """
        if self.tracks.segmentation is None:
            raise ValueError("Cannot compute regionprops features without segmentation.")

        if add_to_set:
            self.add_features_to_set()

        seg = self.tracks.segmentation
        for t in range(seg.shape[0]):
            self._regionprops_update(seg[t])

    def _regionprops_update(self, seg_frame: np.ndarray) -> None:
        """Perform the regionprops computation and update all feature values for a
        single frame of segmentation data.

        Args:
            seg_frame (np.ndarray): A 2D or 3D numpy array representing one time point
                of segmentation data.
        """
        spacing = None if self.tracks.scale is None else tuple(self.tracks.scale[1:])
        for region in regionprops_extended(seg_frame, spacing=spacing):
            node = region.label
            for key in self.features:
                value = getattr(region, self.regionprops_names[key])
                if isinstance(value, tuple):
                    value = list(value)
                self.tracks._set_node_attr(node, key, value)

    def update(self, element: int | tuple[int, int]):
        """Update the regionprops features for the given node.

        Args:
            element (int | tuple[int, int]): The node to update. Should be a node
                and not an edge, but has possible edge type to match generic signature.

        Raises:
            ValueError: If the tracks do not have a segmentation
            ValueError: If an edge element is passed instead of a node.
        """
        if self.tracks.segmentation is None:
            raise ValueError("Cannot update regionprops features without segmentation.")

        if isinstance(element, tuple):
            raise ValueError(
                f"RegionpropsAnnotator update expected a node, got edge {element}"
            )

        time = self.tracks.get_time(element)
        seg_frame = self.tracks.segmentation[time]
        masked_frame = np.where(seg_frame == element, element, 0)

        if np.max(masked_frame) == 0:
            warnings.warn(
                f"Cannot find label {element} in frame {time}: "
                "updating regionprops values to None",
                stacklevel=2,
            )
            for key in self.features:
                value = None
                self.tracks._set_node_attr(element, key, value)
        else:
            self._regionprops_update(masked_frame)
