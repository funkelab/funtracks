from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from funtracks.features import ValueType


def _get_default_key_to_display_name_mapping(
    ndim: int,
) -> dict[str, str | tuple[str, ...]]:
    """Get mapping from default feature keys to their display names.

    Uses annotator classmethods to build the mapping automatically.

    Args:
        ndim: Total number of dimensions including time (3 or 4)

    Returns:
        Dictionary mapping default feature keys to their display names.
    """
    from funtracks.annotators._edge_annotator import EdgeAnnotator
    from funtracks.annotators._regionprops_annotator import RegionpropsAnnotator
    from funtracks.annotators._track_annotator import TrackAnnotator

    mapping = {}

    # Collect features from all annotators
    for annotator_cls in [RegionpropsAnnotator, EdgeAnnotator, TrackAnnotator]:
        features = annotator_cls.get_available_features(ndim=ndim)  # type: ignore[attr-defined]
        for key, feature in features.items():
            display_name = feature["display_name"]
            # Convert list to tuple for hashability
            if isinstance(display_name, list):
                display_name = tuple(display_name)
            mapping[key] = display_name

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
