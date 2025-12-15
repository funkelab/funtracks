from __future__ import annotations

from typing import TYPE_CHECKING

from ._feature import Feature

if TYPE_CHECKING:
    from collections.abc import Sequence


def Time() -> Feature:
    """A feature to hold the integer time frame a node is in.

    Returns:
        Feature: A feature dict representing time
    """
    return {
        "feature_type": "node",
        "value_type": "int",
        "num_values": 1,
        "display_name": "Time",
        "required": True,
        "default_value": None,
    }


def Position(axes: Sequence[str]) -> Feature:
    """A feature to hold the position of a node (time not included).

    Args:
        axes (Sequence[str]): A sequence of the axis names. Used to infer the number
            of dimensions of the position values.

    Returns:
        Feature: A feature dict representing position
    """
    return {
        "feature_type": "node",
        "value_type": "float",
        "num_values": len(axes),
        "display_name": "position",
        "value_names": list(axes),
        "required": True,
        "default_value": None,
        "spatial_dims": True,
    }
