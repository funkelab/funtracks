from __future__ import annotations

from ._feature import Feature


def IoU() -> Feature:
    """A feature representing Intersection over Union for edges.

    Returns:
        Feature: A feature dict representing IoU
    """
    return {
        "feature_type": "edge",
        "value_type": "float",
        "num_values": 1,
        "display_name": "IoU",
        "required": True,
        "default_value": None,
    }
