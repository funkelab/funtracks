"""Helper functions for inferring name mappings from source data to standard keys."""

from __future__ import annotations

import difflib


def _match_exact(
    target_fields: list[str],
    importable_props: list[str],
    mapping: dict[str, str | list[str]],
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
    mapping: dict[str, str | list[str]],
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
    display_name_to_key: dict[str, tuple[str, int]],
    mapping: dict[str, str | list[str]],
) -> list[str]:
    """Find exact matches between properties and feature display names.

    Args:
        importable_props: List of property names available in source data
        display_name_to_key: Mapping from display_name -> (feature_key, index)
            (e.g., {"Area": ("area", 0), "major_axis": ("ellipsoid_axes", 0)})
        mapping: Mapping dict to update with matches (modified in place).
            For single-value features: feature_key -> column_name
            For multi-value features: feature_key -> [col1, col2, ...] in index order

    Returns:
        List of properties that weren't matched in this step
    """
    props_left = importable_props.copy()

    # Track multi-value matches: feature_key -> {index: prop}
    multi_value_matches: dict[str, dict[int, str]] = {}

    for prop in importable_props:
        if prop in display_name_to_key:
            feature_key, idx = display_name_to_key[prop]
            # Check if this is a multi-value feature (has other indices)
            is_multi_value = any(
                k == feature_key and i != idx for _, (k, i) in display_name_to_key.items()
            )
            if is_multi_value:
                if feature_key not in multi_value_matches:
                    multi_value_matches[feature_key] = {}
                multi_value_matches[feature_key][idx] = prop
            else:
                # Single-value feature
                mapping[feature_key] = prop
            props_left.remove(prop)

    # Convert multi-value matches to ordered lists (sorted by index)
    for feature_key, idx_to_prop in multi_value_matches.items():
        if idx_to_prop:
            sorted_indices = sorted(idx_to_prop.keys())
            ordered = [idx_to_prop[i] for i in sorted_indices]
            mapping[feature_key] = ordered

    return props_left


def _match_display_names_fuzzy(
    importable_props: list[str],
    display_name_to_key: dict[str, tuple[str, int]],
    mapping: dict[str, str | list[str]],
    cutoff: float = 0.4,
) -> list[str]:
    """Find fuzzy matches between properties and feature display names.

    Uses case-insensitive fuzzy matching with difflib.get_close_matches.

    Args:
        importable_props: List of property names available in source data
        display_name_to_key: Mapping from display_name -> (feature_key, index)
            (e.g., {"Area": ("area", 0), "major_axis": ("ellipsoid_axes", 0)})
        mapping: Mapping dict to update with matches (modified in place).
            For single-value features: feature_key -> column_name
            For multi-value features: feature_key -> [col1, col2, ...] in index order
        cutoff: Similarity threshold for fuzzy matching (0.0 to 1.0)

    Returns:
        List of properties that weren't matched in this step
    """
    props_left = importable_props.copy()

    if not props_left:
        return props_left

    # Build case-insensitive mapping:
    # lower_display_name -> (original_display, feature_key, index)
    lower_display_map = {
        d.lower(): (d, k, i) for d, (k, i) in display_name_to_key.items()
    }

    # Track multi-value matches: feature_key -> {index: prop}
    multi_value_matches: dict[str, dict[int, str]] = {}

    for prop in importable_props:
        if prop not in props_left:
            continue

        closest = difflib.get_close_matches(
            prop.lower(), lower_display_map.keys(), n=1, cutoff=cutoff
        )

        if closest:
            _, feature_key, idx = lower_display_map[closest[0]]
            # Check if this is a multi-value feature
            is_multi_value = any(
                k == feature_key and i != idx for _, (k, i) in display_name_to_key.items()
            )
            if is_multi_value:
                if feature_key not in multi_value_matches:
                    multi_value_matches[feature_key] = {}
                multi_value_matches[feature_key][idx] = prop
            else:
                mapping[feature_key] = prop
            props_left.remove(prop)

    # Convert multi-value matches to ordered lists (sorted by index)
    for feature_key, idx_to_prop in multi_value_matches.items():
        if idx_to_prop:
            sorted_indices = sorted(idx_to_prop.keys())
            ordered = [idx_to_prop[i] for i in sorted_indices]
            mapping[feature_key] = ordered

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
) -> list[str]:
    """Build list of standard fields to match.

    Position attributes (z, y, x) are NOT included here - they are matched
    via Position feature's value_names to create composite "pos" mapping.

    Args:
        required_features: List of required feature names (e.g., ["time"])

    Returns:
        List of all standard fields to match
    """
    standard_fields = required_features.copy()
    # Add optional standard fields
    optional_standard = ["seg_id"]
    standard_fields.extend(optional_standard)
    return standard_fields


def build_display_name_mapping(
    available_computed_features: dict,
) -> dict[str, tuple[str, int]]:
    """Build reverse mapping from feature display names to feature keys.

    For single-value features, maps display_name -> (feature_key, 0).
    For multi-value features, maps each value_name -> (feature_key, index).

    Args:
        available_computed_features: Dict of feature_key -> feature metadata
            (should have "display_name" and/or "value_names" for each feature)

    Returns:
        Dict mapping display_name or value_name -> (feature_key, index)
        (e.g., {"Area": ("area", 0), "major_axis": ("ellipsoid_axes", 0)})
    """
    display_name_to_key: dict[str, tuple[str, int]] = {}
    for feature_key, feature in available_computed_features.items():
        if feature.get("num_values", 1) > 1:
            # Multi-value feature: map each value_name to (feature_key, index)
            value_names = feature.get("value_names", [])
            for idx, value_name in enumerate(value_names):
                display_name_to_key[value_name] = (feature_key, idx)
        else:
            # Single-value feature: map display_name to (feature_key, 0)
            display_name = feature.get("display_name")
            if isinstance(display_name, str):
                display_name_to_key[display_name] = (feature_key, 0)
    return display_name_to_key


def infer_node_name_map(
    importable_node_properties: list[str],
    required_features: list[str],
    available_computed_features: dict,
) -> dict[str, str | list[str]]:
    """Infer node_name_map by matching importable node properties to standard keys.

    Uses difflib fuzzy matching with the following priority:
    1. Exact matches to standard keys (time, seg_id, etc.)
    2. Fuzzy matches to standard keys (case-insensitive, 40% similarity cutoff)
    3. Exact matches to feature display names/value_names (including position z/y/x)
    4. Fuzzy matches to feature display names (case-insensitive, 40% cutoff)
    5. Remaining properties map to themselves (custom properties)

    Position attributes (z, y, x) are matched via Position feature's value_names,
    resulting in a composite mapping like {"pos": ["z", "y", "x"]}.

    Args:
        importable_node_properties: List of property names available in the source
        required_features: List of required feature names (e.g., ["time"])
        available_computed_features: Dict of feature_key -> feature metadata
            (should have "feature_type" and "display_name" for each feature)
            Contains both node and edge features - will be filtered to node features only

    Returns:
        Inferred node_name_map (standard_key -> source_property). May be incomplete
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
    standard_fields = build_standard_fields(required_features)
    display_name_to_key = build_display_name_mapping(node_features)

    # Initialize state
    mapping: dict[str, str | list[str]] = {}
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
) -> dict[str, str | list[str]]:
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
    mapping: dict[str, str | list[str]] = {}
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
