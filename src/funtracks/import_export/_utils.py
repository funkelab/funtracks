from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from funtracks.features import Feature, ValueType


def get_default_key_to_feature_mapping(
    ndim: int,
    display_name=True,
) -> dict[str, str | tuple[str, ...] | Feature]:
    """Get mapping from default feature keys to their display names.

    Uses annotator classmethods to build the mapping automatically.

    Args:
        ndim: Total number of dimensions including time (3 or 4)
        display_name: If True, return display names. If False, return Feature objects.

    Returns:
        Dictionary mapping default feature keys to their display names or Feature objects.
    """
    from funtracks.annotators._edge_annotator import EdgeAnnotator
    from funtracks.annotators._regionprops_annotator import RegionpropsAnnotator
    from funtracks.annotators._track_annotator import TrackAnnotator

    mapping: dict[str, str | tuple[str, ...] | Feature] = {}

    # Collect features from all annotators
    for annotator_cls in [RegionpropsAnnotator, EdgeAnnotator, TrackAnnotator]:
        features = annotator_cls.get_available_features(ndim=ndim)  # type: ignore[attr-defined]
        for key, feature in features.items():
            if display_name:
                value = feature["display_name"]
                # Convert list to tuple for hashability
                if isinstance(value, list):
                    value = tuple(value)
            else:
                value = feature
            mapping[key] = value

    return mapping


def _infer_dtype_from_array(arr: ArrayLike) -> ValueType:
    """Infer ValueType from numpy array dtype.

    Args:
        arr: Array-like object with a dtype attribute

    Returns:
        String representation of the inferred type ("int", "float", "bool", or "str")
    """
    arr_np = np.asarray(arr)
    if np.issubdtype(arr_np.dtype, np.integer):
        return "int"
    elif np.issubdtype(arr_np.dtype, np.floating):
        return "float"
    elif np.issubdtype(arr_np.dtype, np.bool_):
        return "bool"
    else:
        return "str"
