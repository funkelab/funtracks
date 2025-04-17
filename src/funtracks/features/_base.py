from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

if TYPE_CHECKING:
    pass


class FeatureType(Enum):
    NODE = "node"
    EDGE = "edge"


class Feature(BaseModel):
    """A static feature representation, including necessary information to query
    the Tracks. A feature can be a list of values or a single value.
    """

    attr_name: str  # where the attribute is stored on the graph
    value_names: str | list[str]  # the name of the values stored in the attribute
    feature_type: FeatureType
    valid_ndim: tuple[Literal[3, 4]]
