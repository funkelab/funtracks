from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, field_serializer, field_validator


class FeatureType(Enum):
    NODE = "node"
    EDGE = "edge"


class ValueType(Enum):
    int = int
    float = float
    str = str


class Feature(BaseModel):
    """Data class for storing metadata associated with a graph feature.

    Args:
        key (str): The string used to store the feature on the graph.
        feature_type (FeatureType): Specifies which graph elements the feature applies
            to, nodes or edges.
        value_type (ValueType | type | str): The data type of the feature values.
            Values are expected to be python or numpy native types, not classes.
            Can pass in: the enum fields, a string literal ("int", "float", or "str"),
            or the enum values (python builtins `int`, `float`, or `str`)
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
    value_type: ValueType
    num_values: int = 1
    valid_ndim: tuple[Literal[3, 4], ...] = (3, 4)
    display_name: str | list[str] | None = None
    recompute: bool = False
    required: bool = True
    default_value: Any = None

    def __hash__(self):
        return (self.feature_type.value, self.key).__hash__()

    def __str__(self) -> str:
        return self.feature_type.value + "_" + self.key

    @field_serializer("value_type", when_used="json")
    def _serialize_value_type(self, value_type: ValueType):
        return value_type.name

    @field_validator("value_type", mode="before")
    @classmethod
    def _validate_value_type(cls, value):
        if isinstance(value, str):
            try:
                return ValueType[value]
            except KeyError:
                return value
        return value
