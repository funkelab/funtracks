from __future__ import annotations

from .feature import (
    Feature,
    FeatureType,
)


class Time(Feature):
    def __init__(self, key: str | None = None):
        super().__init__(
            key=key if key is not None else "time",
            feature_type=FeatureType.NODE,
            value_type=int,
            required=True,
            display_name="Time",
        )


class Position(Feature):
    """_summary_

    Set recompute to True if there is a segmentation.
    Must provide axes to get number of values.

    Args:
        Feature (_type_): _description_
    """

    def __init__(self, axes: list[str], key: str | None = None, recompute: bool = False):
        super().__init__(
            key=key if key is not None else "pos",
            display_name=axes,
            num_values=len(axes),
            feature_type=FeatureType.NODE,
            value_type=float,
            recompute=recompute,
        )
