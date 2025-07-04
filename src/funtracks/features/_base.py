from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel


class FeatureType(Enum):
    NODE = "node"
    EDGE = "edge"


class Feature(BaseModel):
    """A static feature representation, including necessary information to query
    the Tracks. A feature can be a list of values or a single value.
    """

    attr_name: str  # where the attribute is stored on the graph
    # dtype :
    display_name: str | None = None # name to display to the user
    value_names: str | list[str]  # the name of the values stored in the attribute
    feature_type: FeatureType
    valid_ndim: tuple[Literal[3, 4], ...]
    computed: bool = False
    regionprops_name: str | None = None
    default_value: Any = None
    required: bool = True  # If true, use default value. If false, throw error if missing

    def __hash__(self):
        return self.attr_name.__hash__()

    def __repr__(self):
        return self.feature_type.value + "_" + self.attr_name
