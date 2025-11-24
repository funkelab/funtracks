from __future__ import annotations

from typing import TYPE_CHECKING
from warnings import warn

import networkx as nx
import numpy as np
import pandas as pd
from geff._typing import InMemoryGeff
from geff.validate.graph import (
    validate_no_repeated_edges,
    validate_no_self_edges,
    validate_nodes_for_edges,
    validate_unique_node_ids,
)
from geff.validate.segmentation import has_seg_ids_at_coords
from geff.validate.tracks import validate_lineages, validate_tracklets

from funtracks.data_model.graph_attributes import NodeAttr

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
    coord = [int(last_node_data["time"])]
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

    # Return True if relabeling is needed (seg_id != node_id)
    return last_node_id != seg_id


def ensure_integer_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure that the 'id' column in the dataframe contains integer values

    Args:
        df (pd.DataFrame): A pandas dataframe with a columns named "id" and "parent_id"

    Returns:
        pd.DataFrame: The same dataframe with the ids remapped to be unique integers.
            Parent id column is also remapped.
    """
    if not pd.api.types.is_integer_dtype(df["id"]):
        unique_ids = df["id"].unique()
        id_mapping = {
            original_id: new_id for new_id, original_id in enumerate(unique_ids, start=1)
        }
        df["id"] = df["id"].map(id_mapping)
        df["parent_id"] = df["parent_id"].map(id_mapping).astype(pd.Int64Dtype())

    return df


def ensure_correct_labels(df: pd.DataFrame, segmentation: np.ndarray) -> np.ndarray:
    """Create a new segmentation where the values from the column df['seg_id'] are
    replaced by those in df['id']

    Args:
        df (pd.DataFrame): A pandas dataframe with columns "seg_id" and "id" where
            the "id" column contains unique integers
        segmentation (np.ndarray): A numpy array where segmentation label values
            are recorded in the "seg_id" column of the dataframe

    Returns:
        np.ndarray: A numpy array similar to the input segmentation of dtype uint64
            where each segmentation now has a unique label across time that corresponds
            to the ID of each node
    """

    # Create a new segmentation image
    new_segmentation = np.zeros_like(segmentation).astype(np.uint64)

    # Loop through each time point
    for t in df[NodeAttr.TIME.value].unique():
        # Filter the dataframe for the current time point
        df_t = df[df[NodeAttr.TIME.value] == t]

        # Create a mapping from seg_id to id for the current time point
        seg_id_to_id = dict(zip(df_t["seg_id"], df_t["id"], strict=True))

        # Apply the mapping to the segmentation image for the current time point
        for seg_id, new_id in seg_id_to_id.items():
            new_segmentation[t][segmentation[t] == seg_id] = new_id

    return new_segmentation


def _test_valid(
    df: pd.DataFrame, segmentation: np.ndarray, scale: list[float] | None
) -> bool:
    """Test if the provided segmentation, dataframe, and scale values are valid together.
    Tests the following requirements:
      - The scale, if provided, has same dimensions as the segmentation
      - The location coordinates have the same dimensions as the segmentation
      - The segmentation pixel value for the coordinates of first node corresponds
    with the provided seg_id as a basic sanity check that the csv file matches with the
    segmentation file

    Args:
        df (pd.DataFrame): the pandas dataframe to turn into tracks, with standardized
            column names
        segmentation (np.ndarray): The segmentation, a 3D or 4D array of integer labels
        scale (list[float] | None): A list of floats representing the relationship between
            the point coordinates and the pixels in the segmentation

    Returns:
        bool: True if the combination of segmentation, dataframe, and scale
            pass all validity tests and can likely be loaded, and False otherwise
    """
    if scale is not None:
        if segmentation.ndim != len(scale):
            warn(
                f"Dimensions of the segmentation image ({segmentation.ndim}) "
                f"do not match the number of scale values given ({len(scale)})",
                stacklevel=2,
            )
            return False
    else:
        scale = [
            1,
        ] * segmentation.ndim

    row = df.iloc[0]
    pos = (
        [row[NodeAttr.TIME.value], row["z"], row["y"], row["x"]]
        if "z" in df.columns
        else [row[NodeAttr.TIME.value], row["y"], row["x"]]
    )

    if segmentation.ndim != len(pos):
        warn(
            f"Dimensions of the segmentation ({segmentation.ndim}) do not match the "
            f"number of positional dimensions ({len(pos)})",
            stacklevel=2,
        )
        return False

    seg_id = row["seg_id"]
    coordinates = [
        int(coord / scale_value) for coord, scale_value in zip(pos, scale, strict=True)
    ]

    try:
        value = segmentation[tuple(coordinates)]
    except IndexError:
        warn(f"Could not get the segmentation value at index {coordinates}", stacklevel=2)
        return False

    return value == seg_id


def _validate_node_name_map(
    name_map: dict[str, str],
    importable_node_props: list[str],
    required_features: list[str],
    position_attr: list[str],
    ndim: int,
) -> None:
    """Validate node name_map contains all required mappings.

    Checks:
    - No None values in required mappings
    - No duplicate values in required mappings
    - All required_features are mapped
    - All position_attr are mapped (based on ndim)
    - All mapped properties exist in importable_node_props

    Args:
        name_map: Mapping from standard keys to source property names
        importable_node_props: List of property names available in the source
        required_features: List of required feature names (e.g., ["time"])
        position_attr: List of position attributes (e.g., ["z", "y", "x"])
        ndim: Number of dimensions (3 for 2D+time, 4 for 3D+time)

    Raises:
        ValueError: If validation fails
    """
    # Build list of required fields
    required_pos_attrs = position_attr[-(ndim - 1) :]
    required_fields = required_features + required_pos_attrs

    # Check for None values in required fields
    none_mappings = [key for key in required_fields if name_map.get(key) is None]
    if none_mappings:
        raise ValueError(
            f"The name_map cannot contain None values. "
            f"Fields with None values: {none_mappings}"
        )

    # Check for duplicate values in required fields
    required_values = [name_map[key] for key in required_fields if key in name_map]
    if len(set(required_values)) != len(required_values):
        raise ValueError(
            "The name_map cannot contain duplicate values. "
            "Please provide a unique mapping for each required field."
        )

    missing_features = []

    # Check required features
    for feature in required_features:
        if feature not in name_map:
            missing_features.append(feature)

    # Check position attributes based on ndim
    # For 3D (ndim=3): require y, x
    # For 4D (ndim=4): require z, y, x
    required_pos_attrs = position_attr[-(ndim - 1) :]
    for pos_attr in required_pos_attrs:
        if pos_attr not in name_map:
            missing_features.append(pos_attr)

    if missing_features:
        raise ValueError(
            f"name_map is missing required mappings: {missing_features}. "
            f"Required features: {required_features}. "
            f"Required position attrs: {required_pos_attrs}"
        )

    # Fail if mapped properties don't exist in importable_node_props
    if importable_node_props:
        invalid_mappings = []
        for std_key, source_prop in name_map.items():
            if source_prop not in importable_node_props:
                invalid_mappings.append(f"{std_key} -> '{source_prop}'")

        if invalid_mappings:
            raise ValueError(
                f"name_map contains mappings to non-existent properties: "
                f"{invalid_mappings}. "
                f"Importable node properties: {importable_node_props}"
            )


def _validate_edge_name_map(
    edge_name_map: dict[str, str],
    importable_edge_props: list[str],
) -> None:
    """Validate edge name_map mappings exist in source.

    Checks:
    - All mapped edge properties exist in importable_edge_props

    Args:
        edge_name_map: Mapping from standard keys to edge property names
        importable_edge_props: List of edge property names available in the source

    Raises:
        ValueError: If validation fails
    """
    if not importable_edge_props:
        return

    invalid_mappings = []
    for std_key, source_prop in edge_name_map.items():
        if source_prop not in importable_edge_props:
            invalid_mappings.append(f"{std_key} -> '{source_prop}'")

    if invalid_mappings:
        raise ValueError(
            f"edge_name_map contains mappings to non-existent properties: "
            f"{invalid_mappings}. "
            f"Importable edge properties: {importable_edge_props}"
        )


def _validate_feature_key_collisions(
    name_map: dict[str, str],
    edge_name_map: dict[str, str] | None,
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


def validate_name_map(
    name_map: dict[str, str],
    importable_node_props: list[str],
    required_features: list[str],
    position_attr: list[str],
    ndim: int,
    edge_name_map: dict[str, str] | None = None,
    importable_edge_props: list[str] | None = None,
) -> None:
    """Validate that name_map and edge_name_map contain valid mappings.

    Orchestrates validation by calling:
    1. _validate_node_name_map() - validates node mappings
    2. _validate_edge_name_map() - validates edge mappings (if provided)
    3. _validate_feature_key_collisions() - checks for key conflicts

    Args:
        name_map: Mapping from standard keys to source property names
        importable_node_props: List of property names available in the source
        required_features: List of required feature names (e.g., ["time"])
        position_attr: List of position attributes (e.g., ["z", "y", "x"])
        ndim: Number of dimensions (3 for 2D+time, 4 for 3D+time)
        edge_name_map: Optional mapping from standard keys to edge property names
        importable_edge_props: Optional list of edge property names available

    Raises:
        ValueError: If validation fails
    """
    # Validate node name_map
    _validate_node_name_map(
        name_map, importable_node_props, required_features, position_attr, ndim
    )

    # Validate edge name_map if provided
    if edge_name_map is not None:
        if importable_edge_props is None:
            importable_edge_props = []
        _validate_edge_name_map(edge_name_map, importable_edge_props)

    # Check for feature key collisions between nodes and edges
    _validate_feature_key_collisions(name_map, edge_name_map)


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
