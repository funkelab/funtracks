from __future__ import annotations

from typing import TYPE_CHECKING

from .feature import (
    Feature,
    FeatureType,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


class Time(Feature):
    """A feature to hold the integer time frame a node is in"""

    def __init__(self, key: str | None = None):
        """
        Args:
            key (str | None, optional): The key used to access the time feature on the
                graph. Defaults to None, which sets it to "time".
        """
        super().__init__(
            key=key if key is not None else "time",
            feature_type=FeatureType.NODE,
            value_type=int,
            required=True,
            display_name="Time",
        )


class Position(Feature):
    """A feature to hold the position of a node (time not included)"""

    def __init__(
        self, axes: Sequence[str], key: str | None = None, recompute: bool = False
    ):
        """
        Args:
            axes (Sequence[str]): A sequence of the axis names. Used to infer the number
                of dimensions of the position values.
            key (str | None, optional): The key used to access the position on the graph.
                Defaults to None, which sets it to "pos". The position must be stored
                in a single key, not each dimension in a separate key.
            recompute (bool, optional): Whether or not to recompute the position when
                node (segmentations) are updated. Defaults to False.
        """
        super().__init__(
            key=key if key is not None else "pos",
            display_name=list(axes),
            num_values=len(axes),
            feature_type=FeatureType.NODE,
            value_type=float,
            recompute=recompute,
        )
