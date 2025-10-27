from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, TypedDict


class Feature(TypedDict):
    """TypedDict for storing metadata associated with a graph feature.

    Use factory functions like Time(), Position(), Area() etc. to create features with
    standard defaults.

    The key is stored separately in the FeatureDict mapping (not in the Feature itself).

    Attributes:
        feature_type (Literal["node", "edge"]): Specifies which graph elements
            the feature applies to.
        value_type (Literal["int", "float", "str"]): The data type of the feature
            values.
        num_values (int): The number of values expected for this feature.
        display_name (str | Sequence[str] | None): The name to use to display the
            feature.
        required (bool): If True, all nodes/edges in the graph are required
            to have this feature.
        default_value (Any): If required is False, this value is returned
            whenever the feature value is missing on the graph.
    """

    feature_type: Literal["node", "edge"]
    value_type: Literal["int", "float", "str"]
    num_values: int
    display_name: str | Sequence[str] | None
    required: bool
    default_value: Any
