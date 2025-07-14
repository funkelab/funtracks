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


class EllipsoidAxes(RPFeature):
    def __init__(self, ndim=3):
        super().__init__(
            key="ellipse_axis_radii",
            value_type=float,
            display_name="Ellipse axis radii" if ndim == 3 else "Ellipsoid axis radii",
            # TODO: I see why the display name and the value names could be different
            # (e.g. in the selection menus). I am not sure what to do about this yet.
            # value_names=("Major Axis", "Minor Axis") if ndim == 3 else ("Major Axis",
            # "Semi-minor Axis", "Minor Axis"),
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
    - number of pixels (not scaled by scaling)
    - ellipsoid major/minor/semi-minor axes
    - circularity/sphericity
    - perimeter/surface area

    TODO: Determine if we want to always update all of them, or allow updating only some.
    I think in this case, always updating all makes sense, but I'm not sure about
    other cases. Perhaps that is a good guideline for when to split your Annotator into
    two.
    """

    def __init__(self, tracks: Tracks):
        features = RegionpropsAnnotator.all_features(tracks)
        super().__init__(tracks, features)

    @classmethod
    def all_features(cls, tracks: Tracks) -> list[Feature]:
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
            EllipsoidAxes(ndim=ndim),
            Circularity(ndim=ndim),
            Perimeter(ndim=ndim),
        ]
        return features

    def compute(self):  # noqa: D102
        if self.tracks.segmentation is None:
            raise ValueError("Cannot compute regionprops features without segmentation.")

        # TODO: figure out a more robust way to determine if we should add the
        # features or not
        for feature in self.features:
            if feature not in self.tracks.features._features:
                self.tracks.features.add_feature(feature)

        seg = self.tracks.segmentation
        for t in range(seg.shape[0]):
            self._regionprops_update(seg[t])

    def _regionprops_update(self, seg_frame):
        spacing = None if self.tracks.scale is None else self.tracks.scale[1:]
        for region in regionprops_extended(seg_frame, spacing=spacing):
            node = region.label
            for feature in self.features:
                value = getattr(region, feature.regionprops_name)
                if isinstance(value, tuple):
                    value = list(value)
                self.tracks._set_node_attr(node, feature.key, value)

    def update(self, element: int | tuple[int, int]):  # noqa: D102
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
