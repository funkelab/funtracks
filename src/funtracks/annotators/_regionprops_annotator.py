from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

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
        feats, rp_names = RegionpropsAnnotator.all_supported_features(tracks)
        super().__init__(tracks, feats)
        self.regionprops_names = rp_names

    @classmethod
    def all_supported_features(
        cls, tracks: Tracks
    ) -> tuple[dict[str, Feature], dict[str, str]]:
        """Get all regionprops features that can be computed for the tracks.

        Args:
            tracks (Tracks): The tracks to get regionprops features for.

        Returns:
            tuple[dict[str, Feature], dict[str, str]]: A dict mapping keys to
                Features, and a dict mapping keys to regionprops attribute names
        """
        if tracks.segmentation is None:
            return {}, {}
        ndim = tracks.ndim
        features = {
            "pos": Centroid(axes=tracks.axis_names),
            "area": Area(ndim=ndim),
            # TODO: Add in intensity when image is passed
            # "intensity": Intensity(ndim=ndim),
            "ellipse_axis_radii": EllipsoidAxes(ndim=ndim),
            "circularity": Circularity(ndim=ndim),
            "perimeter": Perimeter(ndim=ndim),
        }
        regionprops_names = {
            "pos": "centroid",
            "area": "area",
            "ellipse_axis_radii": "axes",
            "circularity": "circularity",
            "perimeter": "perimeter",
        }
        return features, regionprops_names

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
