from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

from .feature import Feature, FeatureType
from .graph_annotator import GraphAnnotator
from .regionprops_extended import regionprops_extended

if TYPE_CHECKING:
    from funtracks.data_model import Tracks


class RPFeature(Feature):
    regionprops_name: str
    feature_type: FeatureType = FeatureType.NODE
    recompute: bool = True


# TODO: make position a regionprops feature when we have a segmentation!
class Area(RPFeature):
    def __init__(self, ndim=3):
        super().__init__(
            key="area",
            value_type=float,
            display_name="Area" if ndim == 3 else "Volume",
            valid_ndim=(3, 4),
            regionprops_name="area",
        )

class Intensity(RPFeature):
    def __init__(self, ndim=3):
        super().__init__(
            key="intensity",
            value_type=float,
            display_name="Intensity",
            valid_ndim=(3, 4),
            regionprops_name="intensity_mean",
        )

class EllipsoidAxes(RPFeature):
    def __init__(self, ndim=3):
        super().__init__(
            key="ellipse_axis_radii",
            value_type=float,
            display_name="Ellipse axis radii" if ndim == 3 else "Ellipsoid axis radii",
            valid_ndim=(3, 4),
            regionprops_name="axes",
        )


class Circularity(RPFeature):
    def __init__(self, ndim=3):
        super().__init__(
            key="circularity",
            value_type=float,
            display_name="Circularity" if ndim == 3 else "Sphericity",
            valid_ndim=(3, 4),
            regionprops_name="circularity" if ndim == 3 else "sphericity",
        )


class Perimeter(RPFeature):
    def __init__(self, ndim=3):
        super().__init__(
            key="perimeter",
            value_type=float,
            display_name="Perimeter" if ndim == 3 else "Surface Area",
            valid_ndim=(3, 4),
            regionprops_name="perimeter" if ndim == 3 else "surface_area",
        )


class RegionpropsAnnotator(GraphAnnotator):
    """A graph annotator using regionprops to extract node features from segmentations.

    The possible features include:
    - area/volume
    - ellipsoid major/minor/semi-minor axes
    - circularity/sphericity
    - perimeter/surface area

    Defaults to computing all features, but individual ones can be turned off by changing
    the self.include value at the corresponding index to the feature in self.features.
    """

    def __init__(self, tracks: Tracks):
        self.all_features: list[Feature] = RegionpropsAnnotator.all_supported_features(
            tracks
        )
        self.include = [
            True,
        ] * len(self.all_features)
        # whether to include each of the features or not. Can update later to exclude
        # features to be more efficient.
        super().__init__(tracks, self.all_features.copy())

    def remove_feature(self, feature: Feature, update_set: bool = True) -> None:
        """Stop computing the given feature in the regionprops annotation process.
        This will not actually remove the feature from the tracks nodes. It will just
        remove it from the list of features that this annotator is updating/computing.

        Args:
            feature (Feature): The feature to remove. Must be in all_features list.
            update_set (bool, optional): Whether to update the tracks FeatureSet or not.
                Defaults to True. Will error if the feature is not already in the
                FeatureSet and the value is True.
        """
        if feature in self.all_features:
            self.include[self.all_features.index(feature)] = False
        if update_set:
            self.tracks.features._features.remove(feature)

    def add_feature(self, feature: Feature, update_set: bool = True) -> None:
        """Start computing the given feature in the regionprops annotation process.
        This will not actually add the feature to the tracks. It will just add it to
        the list of features that this annotator is updating/computing.

        Args:
            feature (Feature): The feature to add. Must be in all_features list.
            update_set (bool, optional): Whether to update the tracks FeatureSet or not.
                Defaults to True. Will error if the feature is already in the
                FeatureSet and the value is True.
        """
        if feature in self.all_features:
            self.include[self.all_features.index(feature)] = True
        if update_set:
            self.tracks.features.add_feature(feature)

    @classmethod
    def all_supported_features(cls, tracks: Tracks) -> list[Feature]:
        """Get a list of all regionprops features that can be computed for the tracks.

        Args:
            tracks (Tracks): The tracks to get regionprops features for.

        Returns:
            list[Feature]: A list of all regionprops features that can be computed
        """
        if tracks.segmentation is None:
            return []
        ndim = tracks.ndim
        features = [
            Area(ndim=ndim),
            Intensity(ndim=ndim),
            EllipsoidAxes(ndim=ndim),
            Circularity(ndim=ndim),
            Perimeter(ndim=ndim),
        ]
        return features

    @property
    def features(self):
        """The list of features that this annotator currently controls.

        In this case, it is the list of all features filtered by the include flags.
        """
        return [
            feat
            for feat, include in zip(self.all_features, self.include, strict=True)
            if include
        ]

    def add_features_to_set(self) -> None:
        """Add the currently included features to the tracks FeatureSet. Usually
        performed during initial computation.
        """
        for feature in self.features:
            self.tracks.features.add_feature(feature)

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
            for feature in self.features:
                value = getattr(region, feature.regionprops_name)
                if isinstance(value, tuple):
                    value = list(value)
                self.tracks._set_node_attr(node, feature.key, value)

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
            for feature in self.features:
                value = None
                self.tracks._set_node_attr(element, feature.key, value)
        else:
            self._regionprops_update(masked_frame)
