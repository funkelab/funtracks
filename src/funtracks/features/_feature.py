from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, TypedDict

from typing_extensions import NotRequired

# Type alias for feature value types
ValueType = Literal["int", "float", "str", "bool"]


class Feature(TypedDict):
    """TypedDict for storing metadata associated with a graph feature.

    Use factory functions like Time(), Position(), Area() etc. to create features with
    standard defaults.

    The key is stored separately in the FeatureDict mapping (not in the Feature itself).

    Attributes:
        feature_type (Literal["node", "edge"]): Specifies which graph elements
            the feature applies to.
        value_type (ValueType): The data type of the feature values.
        num_values (int): The number of values expected for this feature.
        display_name (str): Optional. A display name for the feature.
            If not provided, the feature key is used.
        value_names (Sequence[str]): Optional. Individual display names for each
            value when num_values > 1. Length should match num_values.
        required (bool): If True, all nodes/edges in the graph are required
            to have this feature.
        default_value (Any): If required is False, this value is returned
            whenever the feature value is missing on the graph.
        spatial_dims (bool): Optional. If True, num_values must match the number
            of spatial dimensions (e.g., 2 for 2D, 3 for 3D). Used for features
            like Position and EllipsoidAxes.
    """

    feature_type: Literal["node", "edge"]
    value_type: ValueType
    num_values: int
    display_name: NotRequired[str]
    value_names: NotRequired[Sequence[str]]
    required: bool
    default_value: Any
    spatial_dims: NotRequired[bool]
