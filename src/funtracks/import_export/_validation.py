from __future__ import annotations

from typing import TYPE_CHECKING
from warnings import warn

import networkx as nx
from geff._typing import InMemoryGeff
from geff.validate.graph import (
    validate_no_repeated_edges,
    validate_no_self_edges,
    validate_nodes_for_edges,
    validate_unique_node_ids,
)
from geff.validate.segmentation import has_seg_ids_at_coords
from geff.validate.tracks import validate_lineages, validate_tracklets

if TYPE_CHECKING:
    import dask.array as da

# Constants from import_from_geff
SEG_KEY = "seg_id"


def validate_graph_seg_match(
    graph: nx.DiGraph,
    segmentation: da.Array,
    scale: list[float],
    position_attr: list[str],
) -> bool:
    """Validate if the graph matches the provided segmentation data.

    Checks if the seg_id value of the last node matches the pixel value at the
    (scaled) node coordinates. Returns a boolean indicating whether relabeling
    of the segmentation to match node id values is required.

    Args:
        graph: NetworkX graph with standard keys
        segmentation: Segmentation data (dask array)
        scale: Scaling information (pixel to world coordinates)
        position_attr: Position keys (e.g., ["y", "x"] or ["z", "y", "x"])

    Returns:
        bool: True if relabeling from seg_id to node_id is required.
    """
    # Check segmentation dimensions match graph dimensionality
    ndim = len(position_attr) + 1  # +1 for time
    if len(segmentation.shape) != ndim:
        raise ValueError(
            f"Segmentation has {len(segmentation.shape)} dimensions but graph has "
            f"{ndim} dimensions (time + {len(position_attr)} spatial dims)"
        )

    # Get the last node for validation
    node_ids = list(graph.nodes())
    if not node_ids:
        raise ValueError("Graph has no nodes")

    last_node_id = node_ids[-1]
    last_node_data = graph.nodes[last_node_id]

    # Check if seg_id exists; if not, assume it matches node_id
    seg_id = last_node_data.get(SEG_KEY, last_node_id)

    # Get the coordinates for the last node (using standard keys)
    # Position may be stored as composite "pos" attribute or separate y/x/z attributes
    coord = [int(last_node_data["time"])]
    if "pos" in last_node_data:
        # Composite position: [z, y, x] or [y, x]
        pos = last_node_data["pos"]
        coord.extend(pos)
    else:
        # Separate position attributes (legacy)
        if "z" in position_attr:
            coord.append(last_node_data["z"])
        coord.append(last_node_data["y"])
        coord.append(last_node_data["x"])

    # Check bounds
    for i, (c, s) in enumerate(zip(coord, segmentation.shape, strict=False)):
        pixel_coord = int(c / scale[i])
        if not (0 <= pixel_coord < s):
            raise ValueError(
                f"Coordinate {i} ({c}) is out of bounds for segmentation shape {s} "
                f"(pixel coord: {pixel_coord})"
            )

    # Check if the segmentation pixel value at the coordinates matches the seg id
    seg_id_at_coord, errors = has_seg_ids_at_coords(
        segmentation, [coord], [seg_id], tuple(1 / s for s in scale)
    )
    if not seg_id_at_coord:
        error_msg = "Error testing seg id:\n" + "\n".join(f"- {e}" for e in errors)
        raise ValueError(error_msg)

    # TODO: The relabeling check (seg_id != node_id) is duplicated in
    # TracksBuilder.handle_segmentation. Consider deduplicating by either:
    # 1. Using this return value in the caller, or
    # 2. Removing the return value and making this purely a validation function
    # Return True if relabeling is needed (seg_id != node_id)
    return last_node_id != seg_id


def validate_node_name_map(
    name_map: dict[str, str | list[str]],
    importable_node_props: list[str],
    required_features: list[str],
    available_features: dict | None = None,
    ndim: int | None = None,
    has_segmentation: bool = False,
) -> None:
    """Validate node name_map contains all required mappings.

    Checks:
    - No None values in required mappings
    - All required_features are mapped
    - Position ("pos") is mapped to coordinate columns (unless segmentation provided)
    - All mapped properties exist in importable_node_props
    - Features with spatial_dims=True have correct number of list elements

    Args:
        name_map: Mapping from standard keys to source property names
        importable_node_props: List of property names available in the source
        required_features: List of required feature names (e.g., ["time"])
        available_features: Optional dict of feature_key -> feature metadata
            for spatial_dims validation
        ndim: Optional number of dimensions including time for spatial_dims validation
        has_segmentation: If True, position can be computed from segmentation
            and is not required in name_map

    Raises:
        ValueError: If validation fails
    """
    # Check for None values in required features
    none_mappings = [key for key in required_features if name_map.get(key) is None]
    if none_mappings:
        raise ValueError(
            f"The name_map cannot contain None values. "
            f"Fields with None values: {none_mappings}"
        )

    # Check required features are mapped
    missing_features = [f for f in required_features if f not in name_map]
    if missing_features:
        raise ValueError(
            f"name_map is missing required mappings: {missing_features}. "
            f"Required features: {required_features}."
        )

    # Check position mapping if provided
    if "pos" in name_map:
        pos_mapping = name_map["pos"]
        # Position can be either:
        # - A list of coordinate columns to combine (e.g., ["y", "x"])
        # - A single string for an already-stacked position attribute (e.g., "pos")
        if isinstance(pos_mapping, list) and len(pos_mapping) < 2:
            raise ValueError(
                f"Position mapping as list must have at least 2 coordinate columns. "
                f"Got: {pos_mapping}"
            )
    elif not has_segmentation:
        # Position is required if no segmentation to compute it from
        raise ValueError(
            "name_map must contain 'pos' mapping for position coordinates "
            "(or provide segmentation to compute position). "
            "Expected format: {'pos': ['y', 'x']} or {'pos': 'pos'}"
        )

    # Fail if mapped properties don't exist in importable_node_props
    if importable_node_props:
        invalid_mappings = []
        for std_key, source_prop in name_map.items():
            # Handle multi-value features (list of column names)
            if isinstance(source_prop, list):
                for prop in source_prop:
                    if prop not in importable_node_props:
                        invalid_mappings.append(f"{std_key} -> '{prop}'")
            elif source_prop not in importable_node_props:
                invalid_mappings.append(f"{std_key} -> '{source_prop}'")

        if invalid_mappings:
            raise ValueError(
                f"name_map contains mappings to non-existent properties: "
                f"{invalid_mappings}. "
                f"Importable node properties: {importable_node_props}"
            )

    # Validate spatial_dims features have correct number of list elements
    if available_features is not None:
        validate_spatial_dims_in_name_map(name_map, available_features, ndim)


def validate_spatial_dims_in_name_map(
    name_map: dict[str, str | list[str]],
    available_features: dict,
    ndim: int | None = None,
) -> None:
    """Validate that spatial_dims features have correct number of list elements.

    For features with spatial_dims=True, validates that list mappings have the
    correct number of elements matching the expected spatial dimensions.

    Args:
        name_map: Mapping from standard keys to source property names
        available_features: Dict of feature_key -> feature metadata
            (should have "spatial_dims" for features that need validation)
        ndim: Number of dimensions including time. If None, inferred from
            position mapping length.

    Raises:
        ValueError: If any spatial_dims feature has wrong number of list elements
    """
    # Determine expected spatial dimensions
    expect_spatial_dims: int | None = None
    ndim_from_pos = False
    if ndim is not None:
        expect_spatial_dims = ndim - 1  # ndim includes time
    elif "pos" in name_map:
        pos_mapping = name_map["pos"]
        if isinstance(pos_mapping, list):
            expect_spatial_dims = len(pos_mapping)
            ndim_from_pos = True

    if expect_spatial_dims is None:
        return  # Cannot validate without knowing dimensions

    for key, mapping in name_map.items():
        if key == "pos" and ndim_from_pos:
            # Don't validate pos against itself when it defines dimensions
            continue
        feature = available_features.get(key)
        if feature is None or not isinstance(feature, dict):
            continue
        if not feature.get("spatial_dims", False):
            continue
        # Only validate list mappings here; array shapes are validated
        # after loading via validate_spatial_dims()
        if isinstance(mapping, list) and len(mapping) != expect_spatial_dims:
            display_name = feature.get("display_name", key)
            raise ValueError(
                f"Feature '{key}' ({display_name}) has {len(mapping)} values "
                f"but expected {expect_spatial_dims} spatial dimensions. "
                f"Mapping: {mapping}"
            )


def validate_edge_name_map(
    edge_name_map: dict[str, str | list[str]],
    importable_edge_props: list[str],
    available_features: dict | None = None,
    ndim: int | None = None,
) -> None:
    """Validate edge name_map mappings exist in source.

    Checks:
    - All mapped edge properties exist in importable_edge_props
    - Features with spatial_dims=True have correct number of list elements

    Args:
        edge_name_map: Mapping from standard keys to edge property names
        importable_edge_props: List of edge property names available in the source
        available_features: Optional dict of feature_key -> feature metadata
            for spatial_dims validation
        ndim: Optional number of dimensions including time for spatial_dims validation

    Raises:
        ValueError: If validation fails
    """
    if importable_edge_props:
        invalid_mappings = []
        for std_key, source_prop in edge_name_map.items():
            # Handle multi-value features (list of column names)
            if isinstance(source_prop, list):
                for prop in source_prop:
                    if prop not in importable_edge_props:
                        invalid_mappings.append(f"{std_key} -> '{prop}'")
            elif source_prop not in importable_edge_props:
                invalid_mappings.append(f"{std_key} -> '{source_prop}'")

        if invalid_mappings:
            raise ValueError(
                f"edge_name_map contains mappings to non-existent properties: "
                f"{invalid_mappings}. "
                f"Importable edge properties: {importable_edge_props}"
            )

    # Validate spatial_dims features have correct number of list elements
    if available_features is not None:
        validate_spatial_dims_in_name_map(edge_name_map, available_features, ndim)


def validate_feature_key_collisions(
    name_map: dict[str, str | list[str]],
    edge_name_map: dict[str, str | list[str]] | None,
) -> None:
    """Validate that node and edge feature keys don't overlap.

    Feature keys must be unique across both node and edge features because
    they share the same namespace in FeatureDict.

    Args:
        name_map: Mapping from standard keys to node property names
        edge_name_map: Optional mapping from standard keys to edge property names

    Raises:
        ValueError: If any keys appear in both name_map and edge_name_map
    """
    if edge_name_map is None:
        return

    # Get the standard keys (not the source property names) from both maps
    node_keys = set(name_map.keys())
    edge_keys = set(edge_name_map.keys())

    # Find overlapping keys
    colliding_keys = node_keys & edge_keys

    if colliding_keys:
        raise ValueError(
            f"Feature keys cannot be shared between nodes and edges. "
            f"Colliding keys: {sorted(colliding_keys)}. "
            f"Please use unique keys for node and edge features."
        )


def validate_spatial_dims(
    in_memory_geff: InMemoryGeff,
    available_features: dict,
    ndim: int | None = None,
) -> None:
    """Validate that spatial_dims features have correct number of values.

    Validates that all features with spatial_dims=True have array shapes
    matching the expected number of spatial dimensions.

    Args:
        in_memory_geff: Loaded InMemoryGeff data with node properties
        available_features: Dict of feature_key -> feature metadata
            (should have "spatial_dims" for features that need validation)
        ndim: Number of dimensions including time. If None, validation is skipped.

    Raises:
        ValueError: If any spatial_dims feature has wrong number of values
    """
    if ndim is None:
        return  # Cannot validate without knowing dimensions

    node_props = in_memory_geff["node_props"]
    expect_spatial_dims = ndim - 1  # ndim includes time

    # Validate each feature with spatial_dims=True
    for key, prop_data in node_props.items():
        feature = available_features.get(key)
        if feature is None or not isinstance(feature, dict):
            continue
        if not feature.get("spatial_dims", False):
            continue

        # Check the actual array shape
        values = prop_data["values"]
        actual_dims = values.shape[1] if values.ndim == 2 else 1

        if actual_dims != expect_spatial_dims:
            display_name = feature.get("display_name", key)
            raise ValueError(
                f"Feature '{key}' ({display_name}) has {actual_dims} values "
                f"but expected {expect_spatial_dims} spatial dimensions."
            )


def validate_in_memory_geff(in_memory_geff: InMemoryGeff) -> None:
    """Validate the loaded InMemoryGeff data.

    Validates graph structure (required - raises on failure):
    - validate_unique_node_ids: No duplicate node IDs
    - validate_nodes_for_edges: All edges reference existing nodes
    - validate_no_self_edges: No self-loops
    - validate_no_repeated_edges: No duplicate edges

    Validates optional properties (warns and removes if invalid):
    - validate_tracklets: track_id must form valid tracklets
    - validate_lineages: lineage_id must form valid lineages

    Args:
        in_memory_geff: InMemoryGeff data structure to validate

    Raises:
        ValueError: If required validation (graph structure) fails
    """
    node_ids = in_memory_geff["node_ids"]
    edge_ids = in_memory_geff["edge_ids"]
    node_props = in_memory_geff["node_props"]

    # Validate graph structure (required - always fails if invalid)
    valid, nonunique_nodes = validate_unique_node_ids(node_ids)
    if not valid:
        raise ValueError(f"Some node ids are not unique:\n{nonunique_nodes}")

    valid, invalid_edges = validate_nodes_for_edges(node_ids, edge_ids)
    if not valid:
        raise ValueError(f"Some edges are missing nodes:\n{invalid_edges}")

    valid, invalid_edges = validate_no_self_edges(edge_ids)
    if not valid:
        raise ValueError(f"Self edges found in data:\n{invalid_edges}")

    valid, invalid_edges = validate_no_repeated_edges(edge_ids)
    if not valid:
        raise ValueError(f"Repeated edges found in data:\n{invalid_edges}")

    # Validate tracklet_id if present (optional - remove if invalid)
    if "track_id" in node_props:
        tracklet_ids = node_props["track_id"]["values"]
        valid, errors = validate_tracklets(node_ids, edge_ids, tracklet_ids)
        if not valid:
            warn(
                f"track_id validation failed:\n{chr(10).join(errors)}\n"
                "Removing track_id from data.",
                stacklevel=2,
            )
            del node_props["track_id"]

    # Validate lineage_id if present (optional - remove if invalid)
    if "lineage_id" in node_props:
        lineage_ids = node_props["lineage_id"]["values"]
        valid, errors = validate_lineages(node_ids, edge_ids, lineage_ids)
        if not valid:
            warn(
                f"lineage_id validation failed:\n{chr(10).join(errors)}\n"
                "Removing lineage_id from data.",
                stacklevel=2,
            )
            del node_props["lineage_id"]
