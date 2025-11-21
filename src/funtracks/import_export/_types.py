from __future__ import annotations

from typing import TypedDict

from funtracks.features import ValueType


class ImportedNodeFeature(TypedDict):
    """Metadata for importing a node feature from external format.

    This TypedDict supports both static and computed features through a discriminated
    union pattern using the `feature` field:
    - If `feature` is None: treats as a static feature (raw property)
    - If `feature` is a string: treats as a computed feature with that display name

    Args:
        prop_name: The property name/key on the graph node
        feature: Display name of a computed feature, or None for static features
        recompute: Whether to recompute the feature (only used when feature is not None)
        dtype: The data type of the feature values
    """

    prop_name: str
    feature: str | tuple[str, ...] | None
    recompute: bool
    dtype: ValueType


class ImportedComputedFeature(TypedDict):
    """Metadata for a computed feature being imported.

    This is a specialized version of ImportedNodeFeature used internally for features
    that come from the annotator system (not static features).

    Args:
        prop_name: The property name/key on the graph node
        feature: Display name of the computed feature in the annotator system
        recompute: Whether to recompute the feature or use existing values
    """

    prop_name: str
    feature: str | tuple[str, ...]
    recompute: bool
