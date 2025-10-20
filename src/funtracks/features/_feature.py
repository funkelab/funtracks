from __future__ import annotations

from typing import Any, Literal, TypedDict


class Feature(TypedDict):
    """Data class for storing metadata associated with a graph feature.

    A TypedDict representing feature metadata. Use factory functions like
    Time(), Position(), Area() etc. to create features with proper defaults.

    The key is stored separately in the FeatureSet mapping (not in the Feature itself).

    Attributes:
        feature_type (Literal["node", "edge"]): Specifies which graph elements
            the feature applies to.
        value_type (Literal["int", "float", "str"]): The data type of the feature
            values.
        num_values (int): The number of values expected for this feature.
        valid_ndim (tuple[Literal[3, 4], ...]): Is the feature valid in 3D
            (2D+t) data, 4D (3D+t) data, or both.
        display_name (str | list[str] | None): The name to use to display the
            feature.
        recompute (bool): If True, the feature should be recomputed
            when the element changes.
        required (bool): If True, all nodes/edges in the graph are required
            to have this feature.
        default_value (Any): If required is False, this value is returned
            whenever the feature value is missing on the graph.
    """

    feature_type: Literal["node", "edge"]
    value_type: Literal["int", "float", "str"]
    num_values: int
    valid_ndim: tuple[Literal[3, 4], ...]
    display_name: str | list[str] | None
    recompute: bool
    required: bool
    default_value: Any
