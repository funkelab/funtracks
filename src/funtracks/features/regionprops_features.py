from __future__ import annotations

from typing import TYPE_CHECKING

from .feature import Feature, FeatureType

if TYPE_CHECKING:
    from collections.abc import Sequence


class RPFeature(Feature):
    regionprops_name: str
    feature_type: FeatureType = FeatureType.NODE
    recompute: bool = True


class Centroid(RPFeature):
    def __init__(self, axes: Sequence[str], key: str | None = None):
        super().__init__(
            key=key if key is not None else "pos",
            display_name=list(axes),
            num_values=len(axes),
            value_type=float,
            regionprops_name="centroid",
        )


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
