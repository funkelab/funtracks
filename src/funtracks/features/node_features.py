from __future__ import annotations

from ._base import (
    Feature,
    FeatureType,
)


class Time(Feature):
    def __init__(self, attr_name="time"):
        super().__init__(
            attr_name=attr_name,
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


class NodeSelected(Feature):
    def __init__(self, attr_name: str = "selected"):
        super().__init__(
            attr_name=attr_name,
            value_names="Node Selected",
            feature_type=FeatureType.NODE,
            valid_ndim=(3, 4),
        )


class NodeSelectionPin(Feature):
    def __init__(self, attr_name: str = "pin"):
        super().__init__(
            attr_name=attr_name,
            value_names="Node Pinned",
            feature_type=FeatureType.NODE,
            valid_ndim=(3, 4),
        )
