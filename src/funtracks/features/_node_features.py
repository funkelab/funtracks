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
        "default_value": None,
    }


def SegMask(ndim: int, bbox_key: str = "bbox") -> Feature:
    """A feature to hold a segmentation mask for a node.

    The mask is stored as a pl.Object column. The ``derived_features``
    list records keys (e.g. the bounding-box key) that should be
    cascade-deleted when this mask feature is removed.

    Args:
        ndim: Number of dimensions (3 for 2D+time, 4 for 3D+time).
        bbox_key: The node attribute key for the paired bounding box
            feature. Defaults to ``"bbox"``.

    Returns:
        Feature: A feature dict representing a segmentation mask.
    """
    return {
        "feature_type": "node",
        "value_type": "mask",
        "num_values": 1,
        "display_name": "Segmentation mask",
        "default_value": None,
        "derived_features": [bbox_key],
    }


def SegBbox(ndim: int) -> Feature:
    """A feature to hold the bounding box for a segmentation mask.

    The bounding box is stored as a fixed-size integer array with
    ``2 * spatial_ndim`` values (min coords followed by max coords).

    Args:
        ndim: Number of dimensions (3 for 2D+time, 4 for 3D+time).

    Returns:
        Feature: A feature dict representing a bounding box.
    """
    spatial_ndim = ndim - 1
    return {
        "feature_type": "node",
        "value_type": "int",
        "num_values": 2 * spatial_ndim,
        "display_name": "Bounding box",
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
        "default_value": None,
        "spatial_dims": True,
    }
