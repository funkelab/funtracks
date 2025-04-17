from __future__ import annotations

from ._base import (
    Feature,
    FeatureType,
)


class IoU(Feature):
    def __init__(self, attr_name: str = "iou"):
        super().__init__(
            attr_name=attr_name,
            value_names="IoU",
            feature_type=FeatureType.EDGE,
            valid_ndim=(3, 4),
        )
