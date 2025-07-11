from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel


class FeatureType(Enum):
    NODE = "node"
    EDGE = "edge"


class Feature(BaseModel):
    """Data class for storing metadata associated with a graph feature.

    Args:
        key (str): The string used to store the feature on the graph.
        feature_type (FeatureType): Specifies which graph elements the feature applies
            to, nodes or edges.
        value_type (type): The data type of the feature values. Expected to be python
            or numpy native types, not classes. Can be strings, but not recommended.
        num_values (int, optional): The number of values expected for this feature.
            Defaults to 1.
        valid_ndim (tuple[Literal[3, 4], ...], optional): Is the feature valid in 3D
            (2D+t) data, 4D (3D+t) data, or both. Defaults to (3, 4).
        display_name (str | list[str] | None, optional): The name to use to display the
            feature. Defaults to None.
        recompute (bool, optional): If True, the feature should be recomputed
            when the element changes. Defaults to False.
        required (bool, optional): If True, all nodes/edges in the graph are required
            to have this feature. If False, the value is allowed to be missing on the
            graph. Defaults to False.
        default_value (Any, optional): If required is False, this value is returned
            whenever the feature value is missing on the graph. Defaults to None.
    """

    key: str
    feature_type: FeatureType
    value_type: type
    num_values: int = 1
    valid_ndim: tuple[Literal[3, 4], ...] = (3, 4)
    display_name: str | list[str] | None = None
    recompute: bool = False
    regionprops_name: str | None = None
    required: bool = True
    default_value: Any = None

    def __hash__(self):
        return self.key.__hash__()

    def __str__(self) -> str:
        return self.feature_type.value + "_" + self.key
