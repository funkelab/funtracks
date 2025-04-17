from __future__ import annotations

from ._base import (
    Feature,
    FeatureType,
)


class Time(Feature):
    def __init__(self):
        super().__init__(
            attr_name="time",
            value_names="Time",
            feature_type=FeatureType.NODE,
            valid_ndim=(3, 4),
        )


class Position(Feature):
    def __init__(self, axes: tuple[str, ...], attr_name="pos"):
        super().__init__(
            attr_name=attr_name,
            value_names=axes,
            feature_type=FeatureType.NODE,
            valid_ndim=(3, 4),
        )


class Area(Feature):
    def __init__(self, ndim=3, attr_name="area"):
        super().__init__(
            attr_name=attr_name,
            value_names="Area" if ndim == 3 else "Volume",
            feature_type=FeatureType.NODE,
            valid_ndim=(3, 4),
        )


class TrackID(Feature):
    def __init__(self, attr_name="track_id"):
        super().__init__(
            attr_name=attr_name,
            value_names="Track ID",
            feature_type=FeatureType.NODE,
            valid_ndim=(3, 4),
        )
