from __future__ import annotations

from ._feature import Feature


def TrackletID() -> Feature:
    """A feature representing tracklet ID for nodes.

    Returns:
        Feature: A feature dict representing tracklet ID
    """
    return {
        "feature_type": "node",
        "value_type": "int",
        "num_values": 1,
        "display_name": "Tracklet ID",
        "required": True,
        "default_value": None,
    }


def LineageID() -> Feature:
    """A feature representing lineage ID for nodes.

    Returns:
        Feature: A feature dict representing lineage ID
    """
    return {
        "feature_type": "node",
        "value_type": "int",
        "num_values": 1,
        "display_name": "Lineage ID",
        "required": True,
        "default_value": None,
    }
