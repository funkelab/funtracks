"""Helper functions for inferring name mappings from source data to standard keys."""

from __future__ import annotations

import difflib


def _match_exact(
    target_fields: list[str],
    importable_props: list[str],
    mapping: dict[str, str],
) -> list[str]:
    """Find exact matches between target fields and importable properties.

    Args:
        target_fields: List of field names to match (e.g., ["time", "x", "y"])
        importable_props: List of property names available in source data
        mapping: Mapping dict to update with matches (modified in place)

    Returns:
        List of properties that weren't matched in this step
    """
    props_left = importable_props.copy()

    for field in target_fields:
        if field in mapping:
            continue
        if field in props_left:
            mapping[field] = field
            props_left.remove(field)

    return props_left


def _match_fuzzy(
    target_fields: list[str],
    importable_props: list[str],
    mapping: dict[str, str],
    cutoff: float = 0.4,
) -> list[str]:
    """Find fuzzy matches between target fields and importable properties.

    Uses case-insensitive fuzzy matching with difflib.get_close_matches.

    Args:
        target_fields: List of field names to match (e.g., ["time", "x", "y"])
        importable_props: List of property names available in source data
        mapping: Mapping dict to update with matches (modified in place)
        cutoff: Similarity threshold for fuzzy matching (0.0 to 1.0)

    Returns:
        List of properties that weren't matched in this step
    """
    props_left = importable_props.copy()

    for field in target_fields:
        if field in mapping:
            continue
        if len(props_left) == 0:
            break

        # Create case-insensitive mapping
        lower_map = {p.lower(): p for p in props_left}
        closest = difflib.get_close_matches(
            field.lower(), lower_map.keys(), n=1, cutoff=cutoff
        )

        if closest:
            best_match = lower_map[closest[0]]
            mapping[field] = best_match
            props_left.remove(best_match)

    return props_left


def _match_display_names_exact(
    importable_props: list[str],
    display_name_to_key: dict[str, str],
    mapping: dict[str, str],
) -> list[str]:
    """Find exact matches between properties and feature display names.

    Args:
        importable_props: List of property names available in source data
        display_name_to_key: Mapping from display_name -> feature_key
            (e.g., {"Area" -> "area", "Circularity" -> "circularity"})
        mapping: Mapping dict to update with matches (modified in place)

    Returns:
        List of properties that weren't matched in this step
    """
    props_left = importable_props.copy()

    for prop in importable_props:
        if prop in display_name_to_key:
            feature_key = display_name_to_key[prop]
            mapping[feature_key] = prop
            props_left.remove(prop)

    return props_left


def _match_display_names_fuzzy(
    importable_props: list[str],
    display_name_to_key: dict[str, str],
    mapping: dict[str, str],
    cutoff: float = 0.4,
) -> list[str]:
    """Find fuzzy matches between properties and feature display names.

    Uses case-insensitive fuzzy matching with difflib.get_close_matches.

    Args:
        importable_props: List of property names available in source data
        display_name_to_key: Mapping from display_name -> feature_key
            (e.g., {"Area" -> "area", "Circularity" -> "circularity"})
        mapping: Mapping dict to update with matches (modified in place)
        cutoff: Similarity threshold for fuzzy matching (0.0 to 1.0)

    Returns:
        List of properties that weren't matched in this step
    """
    props_left = importable_props.copy()

    if not props_left:
        return props_left

    # Build case-insensitive mapping:
    # lower_display_name -> (original_display, feature_key)
    lower_display_map = {d.lower(): (d, k) for d, k in display_name_to_key.items()}

    for prop in importable_props:
        if prop not in props_left:
            continue

        closest = difflib.get_close_matches(
            prop.lower(), lower_display_map.keys(), n=1, cutoff=cutoff
        )

        if closest:
            _, feature_key = lower_display_map[closest[0]]
            mapping[feature_key] = prop
            props_left.remove(prop)

    return props_left


def _map_remaining_to_self(remaining_props: list[str]) -> dict[str, str]:
    """Map remaining properties to themselves (custom properties).

    Args:
        remaining_props: List of property names that weren't matched

    Returns:
        Dict mapping each prop -> itself (e.g., {"custom_col": "custom_col"})
    """
    return {prop: prop for prop in remaining_props}


def build_standard_fields(
    required_features: list[str],
    position_attr: list[str],
    ndim: int,
) -> list[str]:
    """Build list of standard fields to match.
    # TODO: just use required fields and computed keys
    Args:
        required_features: List of required feature names (e.g., ["time"])
        position_attr: List of position attributes (e.g., ["z", "y", "x"])
        ndim: Number of dimensions (3 for 2D+time, 4 for 3D+time)

    Returns:
        List of all standard fields to match
    """
    standard_fields = required_features.copy()
    # Add position attributes based on ndim
    required_pos_attrs = position_attr[-(ndim - 1) :]
    standard_fields.extend(required_pos_attrs)
    # Add optional standard fields
    optional_standard = ["seg_id", "track_id", "lineage_id"]
    standard_fields.extend(optional_standard)
    return standard_fields


def build_display_name_mapping(available_computed_features: dict) -> dict[str, str]:
    """Build reverse mapping from feature display names to feature keys.
    # TODO: do something about tuples
    Args:
        available_computed_features: Dict of feature_key -> feature metadata
            (should have "display_name" for each feature)

    Returns:
        Dict mapping display_name -> feature_key
        (e.g., {"Area": "area", "Circularity": "circularity"})
    """
    display_name_to_key = {}
    for feature_key, feature in available_computed_features.items():
        display_name = feature.get("display_name")
        # Only map single-string display names (skip tuples/lists)
        if isinstance(display_name, str):
            display_name_to_key[display_name] = feature_key
    return display_name_to_key


def infer_name_map(
    importable_node_properties: list[str],
    required_features: list[str],
    position_attr: list[str],
    ndim: int,
    available_computed_features: dict,
) -> dict[str, str]:
    """Infer name_map by matching importable node properties to standard keys.

    Uses difflib fuzzy matching with the following priority:
    1. Exact matches to standard keys
    2. Fuzzy matches to standard keys (case-insensitive, 40% similarity cutoff)
    3. Exact matches to feature display names
    4. Fuzzy matches to feature display names (case-insensitive, 40% cutoff)
    5. Remaining properties map to themselves (custom properties)

    Args:
        importable_node_properties: List of property names available in the source
        required_features: List of required feature names (e.g., ["time"])
        position_attr: List of position attributes (e.g., ["z", "y", "x"])
        ndim: Number of dimensions (3 for 2D+time, 4 for 3D+time)
        available_computed_features: Dict of feature_key -> feature metadata
            (should have "feature_type" and "display_name" for each feature)
            Contains both node and edge features - will be filtered to node features only

    Returns:
        Inferred name_map (standard_key -> source_property). May be incomplete
        if required features cannot be matched. Use validate_name_map() to
        ensure all required fields are present before building.
    """
    # Filter to node features only
    node_features = {
        k: v
        for k, v in available_computed_features.items()
        if v.get("feature_type") == "node"
    }

    # Setup: Build list of standard fields and display name mapping
    standard_fields = build_standard_fields(required_features, position_attr, ndim)
    display_name_to_key = build_display_name_mapping(node_features)

    # Initialize state
    mapping: dict[str, str] = {}
    props_left = importable_node_properties.copy()

    # Pipeline of matching steps
    # Step 1: Exact matches for standard fields
    props_left = _match_exact(standard_fields, props_left, mapping)

    # Step 2: Fuzzy matches for remaining standard fields
    props_left = _match_fuzzy(standard_fields, props_left, mapping)

    # Step 3: Exact matches with feature display names
    props_left = _match_display_names_exact(props_left, display_name_to_key, mapping)

    # Step 4: Fuzzy matches with feature display names
    props_left = _match_display_names_fuzzy(props_left, display_name_to_key, mapping)

    # Step 5: Map remaining properties to themselves (custom properties)
    custom_mapping = _map_remaining_to_self(props_left)
    mapping.update(custom_mapping)

    return mapping


def infer_edge_name_map(
    importable_edge_properties: list[str],
    available_computed_features: dict | None = None,
) -> dict[str, str]:
    """Infer edge_name_map by matching importable edge properties to standard keys.

    Uses difflib fuzzy matching with the following priority:
    1. Exact matches to edge feature default keys
    2. Fuzzy matches to edge feature default keys (case-insensitive, 40%
       similarity cutoff)
    3. Exact matches to edge feature display names
    4. Fuzzy matches to edge feature display names (case-insensitive, 40% cutoff)
    5. Remaining properties map to themselves (custom properties)

    Args:
        importable_edge_properties: List of edge property names available in source
        available_computed_features: Optional dict of feature_key -> feature metadata
            (should have "feature_type" and "display_name" for each feature)
            Contains both node and edge features - will be filtered to edge features only

    Returns:
        Inferred edge_name_map (standard_key -> source_property)
    """
    # Filter to edge features only
    edge_features = {}
    if available_computed_features is not None:
        edge_features = {
            k: v
            for k, v in available_computed_features.items()
            if v.get("feature_type") == "edge"
        }

    # Extract edge feature keys and display name mapping
    edge_feature_keys = list(edge_features.keys())
    display_name_to_key = build_display_name_mapping(edge_features)

    # Initialize state
    mapping: dict[str, str] = {}
    props_left = importable_edge_properties.copy()

    # Pipeline of matching steps
    # Step 1: Exact matches for edge feature keys
    props_left = _match_exact(edge_feature_keys, props_left, mapping)

    # Step 2: Fuzzy matches for edge feature keys
    props_left = _match_fuzzy(edge_feature_keys, props_left, mapping)

    # Step 3: Exact matches with edge feature display names
    if display_name_to_key:
        props_left = _match_display_names_exact(props_left, display_name_to_key, mapping)

    # Step 4: Fuzzy matches with edge feature display names
    if display_name_to_key:
        props_left = _match_display_names_fuzzy(props_left, display_name_to_key, mapping)

    # Step 5: Map remaining properties to themselves (custom properties)
    custom_mapping = _map_remaining_to_self(props_left)
    mapping.update(custom_mapping)

    return mapping
