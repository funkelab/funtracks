from collections.abc import Sequence
from typing import Any

import numpy as np
import polars as pl
import tracksdata as td
from polars.testing import assert_frame_equal
from skimage import measure
from tracksdata.nodes._mask import Mask

from .graph_attributes import EdgeAttr, NodeAttr


def td_get_single_attr_from_edge(graph, edge: tuple[int, int], attrs: Sequence[str]):
    """Get a single attribute from a edge in a tracksdata graph."""

    item = graph.filter(node_ids=[edge[0], edge[1]]).edge_attrs()[attrs].item()
    return item


def convert_np_types(data):
    """Recursively convert numpy and polars types to native Python types."""
    if isinstance(data, dict):
        return {key: convert_np_types(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_np_types(item) for item in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()  # Convert numpy arrays to Python lists
    elif isinstance(data, np.integer):
        return int(data)  # Convert numpy integers to Python int
    elif isinstance(data, np.floating):
        return float(data)  # Convert numpy floats to Python float
    elif isinstance(data, pl.Series):
        return data.to_list()  # Convert polars Series to Python list
    else:
        return data  # Return the data as-is if it's already a native Python type


def td_to_dict(graph) -> dict:
    """Convert the tracks graph to a dictionary format similar to
    networkx.node_link_data.

    This is used within Tracks.save to save the graph to a json file.
    """
    node_attr_names = graph.node_attr_keys.copy()
    node_attr_names.insert(0, "node_id")
    node_data_all = graph.node_attrs()
    nodes = []
    for i, node in enumerate(graph.node_ids()):
        node_data = node_data_all[i]
        node_data_dict = {
            node_attr_names[i]: convert_np_types(node_data[node_attr_names[i]].item())
            for i in range(len(node_attr_names))
        }
        node_dict = {"id": node}
        node_dict.update(node_data_dict)  # Add all attributes to the dictionary
        node_dict.pop("id")
        nodes.append(node_dict)

    edge_attr_names = graph.edge_attr_keys.copy()
    edge_attr_names.insert(0, "edge_id")
    edge_attr_names.insert(1, "source_id")
    edge_attr_names.insert(2, "target_id")
    edges = []
    edge_data_all = graph.edge_attrs()
    for i, _ in enumerate(graph.edge_ids()):
        edge_data = edge_data_all[i]
        edge_data_dict = {
            edge_attr_names[i]: convert_np_types(edge_data[edge_attr_names[i]].item())
            for i in range(len(edge_attr_names))
        }
        edge_dict = {
            "source": edge_data_dict["source_id"],
            "target": edge_data_dict["target_id"],
        }
        edge_data_dict.pop("source_id")
        edge_data_dict.pop("target_id")
        edge_dict.update(edge_data_dict)  # Add all attributes to the dictionary
        edges.append(edge_dict)

    edges = sorted(edges, key=lambda edge: edge["edge_id"])

    return {
        "directed": True,  # all TracksData graphs are directed
        "multigraph": False,  # all TracksData garphs are not multigraphs
        "graph": {},  # Add any graph-level attributes if needed
        "nodes": nodes,
        "edges": edges,
    }


def td_from_dict(graph_dict) -> td.graph.GraphView:
    """Convert a dictionary to a tracksdata SQL graph."""

    # Get edge attribute keys and data
    node_attr_keys = list(graph_dict["nodes"][0].keys())
    node_attr_keys.remove("node_id")  # node_id is handled separately
    node_data_list = [
        {k: node[k] for k in node_attr_keys} for node in graph_dict["nodes"]
    ]
    node_ids = [node["node_id"] for node in graph_dict["nodes"]]

    # convert pos to numpy arrays
    if "pos" in node_attr_keys:
        for i in range(len(node_data_list)):
            node_data_list[i]["pos"] = np.array(node_data_list[i]["pos"])

    # Get edge attribute keys and data
    edge_attr_keys = list(graph_dict["edges"][0].keys())
    edge_data_list = [
        {k: edge[k] for k in edge_attr_keys} for edge in graph_dict["edges"]
    ]

    # rename 'source' and 'target' to 'source_id' and 'target_id'
    if "source" in edge_attr_keys:
        edge_attr_keys.remove("source")
        edge_attr_keys.append("source_id")
    if "target" in edge_attr_keys:
        edge_attr_keys.remove("target")
        edge_attr_keys.append("target_id")
    for edge in edge_data_list:
        edge["source_id"] = edge["source"]
        edge["target_id"] = edge["target"]
        edge.pop("source")
        edge.pop("target")

    kwargs = {
        "drivername": "sqlite",
        "database": ":memory:",
        "overwrite": True,
    }
    graph_td = td.graph.SQLGraph(**kwargs)

    # add node/edge attributes to graph, including default values
    for key in node_attr_keys:
        if key not in ["t"]:
            first_value = node_data_list[0][key]
            # if "pos" is an array, default_value should be None
            if key == "pos" and len(first_value) > 1:
                first_value = None
            graph_td.add_node_attr_key(key, default_value=first_value)
    for key in edge_attr_keys:
        if key not in ["edge_id", "source_id", "target_id"]:
            first_value = edge_data_list[0][key]
            graph_td.add_edge_attr_key(key, default_value=first_value)

    graph_td.bulk_add_nodes(node_data_list, indices=node_ids)
    graph_td.bulk_add_edges(edge_data_list)

    graph_td_sub = graph_td.filter(
        td.NodeAttr(td.DEFAULT_ATTR_KEYS.SOLUTION) == 1,
        td.EdgeAttr(td.DEFAULT_ATTR_KEYS.SOLUTION) == 1,
    ).subgraph()

    return graph_td_sub


def td_graph_edge_list(graph):
    """Get list of edges from a tracksdata graph.

    Args:
        graph: A tracksdata graph

    Returns:
        list: List of edges: [[source_id, target_id], ...]
    """
    existing_edges = (
        graph.edge_attrs().select(["source_id", "target_id"]).to_numpy().tolist()
    )
    return existing_edges


def td_get_node_ids_from_df(df):
    """Get list of node_ids from a polars DataFrame, handling empty case.

    Args:
        df: A polars DataFrame that may contain a 'node_id' column

    Returns:
        list: List of node_ids if DataFrame has rows, empty list otherwise
    """
    return list(df["node_id"]) if len(df) > 0 else []


def td_get_predecessors(graph, node):
    """Get list of predecessor node IDs for a given node.

    Args:
        graph: A tracksdata graph
        node: Node ID to get predecessors for

    Returns:
        list: List of predecessor node IDs
    """
    predecessors_df = graph.predecessors(node)
    return td_get_node_ids_from_df(predecessors_df)


def td_get_successors(graph, node):
    """Get list of successor node IDs for a given node.

    Args:
        graph: A tracksdata graph
        node: Node ID to get successors for

    Returns:
        list: List of successor node IDs
    """
    successors_df = graph.successors(node)
    return td_get_node_ids_from_df(successors_df)


def values_are_equal(val1: Any, val2: Any) -> bool:
    """
    Compare two values that could be of any type (arrays, lists, scalars, etc.)

    Args:
        val1: First value to compare
        val2: Second value to compare

    Returns:
        bool: True if values are equal, False otherwise
    """
    # If both are None, they're equal
    if val1 is None and val2 is None:
        return True

    # If only one is None, they're not equal
    if val1 is None or val2 is None:
        return False

    # Handle numpy arrays
    if isinstance(val1, np.ndarray) or isinstance(val2, np.ndarray):
        try:
            return np.array_equal(np.asarray(val1), np.asarray(val2), equal_nan=True)
        except (ValueError, TypeError):
            # Return False if arrays cannot be compared (incompatible shapes or types)
            return False

    # Handle lists that might need to be compared as arrays
    if isinstance(val1, list) and isinstance(val2, list):
        try:
            return np.array_equal(np.asarray(val1), np.asarray(val2), equal_nan=True)
        except (ValueError, TypeError):
            # Return False if arrays cannot be compared (incompatible shapes or types)
            # If can't convert to numpy arrays, fall back to regular comparison
            return val1 == val2

    # Default comparison for other types
    return val1 == val2


def validate_and_merge_node_attrs(attrs_of_root_node: dict, node_dict: dict) -> dict:
    """
    Compare and validate two node attribute dictionaries.

    Args:
        attrs_of_root_node: Dictionary containing the root node attributes (reference)
        node_dict: Dictionary containing the node attributes to compare/merge

    Returns:
        Updated dictionary with merged values

    Raises:
        ValueError: If node_dict contains fields not present in attrs_of_root_node
    """
    # Check for invalid fields in node_dict
    invalid_fields = set(node_dict.keys()) - set(attrs_of_root_node.keys())
    if invalid_fields:
        raise ValueError(
            f"Node dictionary contains fields not present in root: {invalid_fields}"
        )

    # Create a new dict starting with root values
    merged_attrs = attrs_of_root_node.copy()

    # Compare and update values
    for field, value in node_dict.items():
        # Skip None values from node_dict to keep root values
        if value is not None and not values_are_equal(value, attrs_of_root_node[field]):
            merged_attrs[field] = value

    return merged_attrs


def assert_node_attrs_equal_with_masks(
    object1, object2, check_column_order: bool = False, check_row_order: bool = False
):
    """
    Fully compare the content of two graphs (node attributes and Masks)
    """

    if isinstance(object1, td.graph.GraphView) and (
        isinstance(object2, td.graph.GraphView)
    ):
        node_attrs1 = object1.node_attrs()
        node_attrs2 = object2.node_attrs()
    elif isinstance(object1, pl.DataFrame) and isinstance(object2, pl.DataFrame):
        node_attrs1 = object1
        node_attrs2 = object2
    else:
        raise ValueError(
            "Both objects must be either tracksdata graphs or polars DataFrames"
        )

    assert_frame_equal(
        node_attrs1.drop("mask"),
        node_attrs2.drop("mask"),
        check_column_order=check_column_order,
        check_row_order=check_row_order,
    )
    for node in node_attrs1["node_id"]:
        mask1 = node_attrs1.filter(pl.col("node_id") == node)["mask"].item()
        mask2 = node_attrs2.filter(pl.col("node_id") == node)["mask"].item()
        assert np.array_equal(mask1.bbox, mask2.bbox)
        assert np.array_equal(mask1.mask, mask2.mask)


def pixels_to_td_mask(
    pix: tuple[np.ndarray, ...], ndim: int, scale: list[float] | None
) -> tuple[Mask, float]:
    """
    Convert pixel coordinates to tracksdata mask format.

    Args:
        pix: Pixel coordinates for 1 node!
        ndim: Number of dimensions (2D or 3D).
        scale: Scale factors for each dimension, used for area calculation

    Returns:
        Tuple[td.Mask, np.ndarray]: A tuple containing the
            tracksdata mask and the mask array.
    """

    spatial_dims = ndim - 1  # Handle both 2D and 3D

    # Calculate position and bounding box more efficiently
    bbox = np.zeros(2 * spatial_dims, dtype=int)

    # Calculate bbox and shape in one pass
    for dim in range(spatial_dims):
        pix_dim = dim + 1
        min_val = np.min(pix[pix_dim])
        max_val = np.max(pix[pix_dim])
        bbox[dim] = min_val
        bbox[dim + spatial_dims] = max_val + 1

    # Calculate mask shape from bbox
    mask_shape = bbox[spatial_dims:] - bbox[:spatial_dims]

    # Convert coordinates to mask-local coordinates
    local_coords = [pix[dim + 1] - bbox[dim] for dim in range(spatial_dims)]
    mask_array = np.zeros(mask_shape, dtype=bool)
    mask_array[tuple(local_coords)] = True

    area = np.sum(mask_array)
    if scale is not None:
        area *= np.prod(scale[1:])

    mask = Mask(mask_array, bbox=bbox)
    return mask, area


def combine_td_masks(
    mask1: Mask, mask2: Mask, scale: list[float] | None
) -> tuple[Mask, float]:
    """
    Combine two tracksdata mask objects into a single mask object.
    The resulting mask will encompass both input masks.

    Args:
        mask1: First Mask object with .mask and .bbox attributes
        mask2: Second Mask object with .mask and .bbox attributes
        scale: Scale factors for each dimension, used for area calculation

    Returns:
        Mask: A new Mask object containing the union of both masks
    """
    # Get spatial dimensions from first bbox
    spatial_dims = len(mask1.bbox) // 2

    # Calculate the combined bounding box
    combined_bbox = np.zeros(2 * spatial_dims, dtype=int)

    # Find the minimum and maximum coordinates for the new bbox
    for dim in range(spatial_dims):
        combined_bbox[dim] = min(mask1.bbox[dim], mask2.bbox[dim])
        combined_bbox[dim + spatial_dims] = max(
            mask1.bbox[dim + spatial_dims], mask2.bbox[dim + spatial_dims]
        )

    # Calculate the shape of the combined mask
    combined_shape = combined_bbox[spatial_dims:] - combined_bbox[:spatial_dims]
    combined_mask = np.zeros(combined_shape, dtype=bool)

    # Create slicing for first mask
    slices1 = tuple(
        slice(offset1_start, offset1_end)
        for offset1_start, offset1_end in zip(
            [mask1.bbox[d] - combined_bbox[d] for d in range(spatial_dims)],
            [
                mask1.bbox[d] - combined_bbox[d] + mask1.mask.shape[d]
                for d in range(spatial_dims)
            ],
            strict=True,
        )
    )

    # Place second mask in the combined mask
    slices2 = tuple(
        slice(offset2_start, offset2_end)
        for offset2_start, offset2_end in zip(
            [mask2.bbox[d] - combined_bbox[d] for d in range(spatial_dims)],
            [
                mask2.bbox[d] - combined_bbox[d] + mask2.mask.shape[d]
                for d in range(spatial_dims)
            ],
            strict=True,
        )
    )

    # Combine the masks using logical OR
    combined_mask[slices1] |= mask1.mask
    combined_mask[slices2] |= mask2.mask

    area = np.sum(combined_mask)
    if scale is not None:
        area *= np.prod(scale[1:])

    return Mask(combined_mask, bbox=combined_bbox), float(area)


def subtract_td_masks(
    mask_old: Mask, mask_new: Mask, scale: list[float] | None
) -> tuple[Mask, float]:
    """
    Subtract mask_new from mask_old, creating a new mask with the difference.
    Will throw an error if mask_new contains True pixels that are not True in mask_old.

    Args:
        mask_old: Original Mask object that pixels will be removed from
        mask_new: Mask object containing pixels to remove
        scale: Scale factors for each dimension, used for area calculation

    Returns:
        Tuple[Mask, float]: A new Mask object containing the result of
            mask_old - mask_new, and the new area after subtraction
    """
    # Get spatial dimensions from first bbox
    spatial_dims = len(mask_old.bbox) // 2

    # First verify that all True pixels in mask_new are also True in mask_old
    # We do this by placing both masks in a common coordinate system

    # Calculate the combined bounding box
    combined_bbox = np.zeros(2 * spatial_dims, dtype=int)
    for dim in range(spatial_dims):
        combined_bbox[dim] = min(mask_old.bbox[dim], mask_new.bbox[dim])
        combined_bbox[dim + spatial_dims] = max(
            mask_old.bbox[dim + spatial_dims], mask_new.bbox[dim + spatial_dims]
        )

    # Place both masks in the combined coordinate system
    combined_shape = combined_bbox[spatial_dims:] - combined_bbox[:spatial_dims]
    old_mask_full = np.zeros(combined_shape, dtype=bool)
    new_mask_full = np.zeros(combined_shape, dtype=bool)

    # Create slicing for old mask
    slices_old = tuple(
        slice(offset_start, offset_end)
        for offset_start, offset_end in zip(
            [mask_old.bbox[d] - combined_bbox[d] for d in range(spatial_dims)],
            [
                mask_old.bbox[d] - combined_bbox[d] + mask_old.mask.shape[d]
                for d in range(spatial_dims)
            ],
            strict=True,
        )
    )

    # Create slicing for new mask
    slices_new = tuple(
        slice(offset_start, offset_end)
        for offset_start, offset_end in zip(
            [mask_new.bbox[d] - combined_bbox[d] for d in range(spatial_dims)],
            [
                mask_new.bbox[d] - combined_bbox[d] + mask_new.mask.shape[d]
                for d in range(spatial_dims)
            ],
            strict=True,
        )
    )

    old_mask_full[slices_old] = mask_old.mask
    new_mask_full[slices_new] = mask_new.mask

    # Check if all True pixels in mask_new are also True in mask_old
    if not np.all(new_mask_full <= old_mask_full):
        raise ValueError("mask_new contains True pixels that are not True in mask_old")

    # Perform the subtraction
    result_mask = old_mask_full & ~new_mask_full

    # Find the new bounding box based on remaining True pixels
    if not np.any(result_mask):
        # If no pixels remain, return minimal empty mask
        result_bbox = np.zeros(2 * spatial_dims, dtype=int)
        return Mask(np.zeros((1,) * spatial_dims, dtype=bool), bbox=result_bbox), 0.0

    true_indices = np.nonzero(result_mask)
    result_bbox = np.zeros(2 * spatial_dims, dtype=int)

    for dim in range(spatial_dims):
        result_bbox[dim] = np.min(true_indices[dim]) + combined_bbox[dim]
        result_bbox[dim + spatial_dims] = (
            np.max(true_indices[dim]) + combined_bbox[dim] + 1
        )

    # Extract the final mask within the new bbox
    final_shape = result_bbox[spatial_dims:] - result_bbox[:spatial_dims]
    final_mask = np.zeros(final_shape, dtype=bool)

    # Create slicing from result_mask to final_mask space
    slices_final = tuple(
        slice(
            result_bbox[dim] - combined_bbox[dim],
            result_bbox[dim] - combined_bbox[dim] + final_shape[dim],
        )
        for dim in range(spatial_dims)
    )

    # Copy the relevant portion of the result_mask to final_mask
    final_mask[:] = result_mask[slices_final]

    # Calculate area
    area = np.sum(final_mask)
    if scale is not None:
        area *= np.prod(scale[1:])

    return Mask(final_mask, bbox=result_bbox), float(area)


def compute_node_attrs_from_masks(
    masks: list[Mask], ndim: int, scale: list[float] | None
) -> dict[str, list[Any]]:
    """
    Compute node attributes (area and pos) from a tracksdata Mask object.

    Parameters
    ----------
    masks : list[Mask]
        A list of tracksdata Mask objects containing the mask and bounding box.
    ndim : int
        Number of dimensions (2D or 3D).
    scale : list[float] | None
        Scale factors for each dimension.

    Returns
    -------
    dict[str, Any]
        A dictionary containing the computed node attributes ('area' and 'pos').
    """
    if not masks:
        return {}

    area_list = []
    pos_list = []
    for mask in masks:
        seg_crop = mask.mask
        seg_bbox = mask.bbox

        pos_scale = scale[1:] if scale is not None else np.ones(ndim - 1)
        area = np.sum(seg_crop)
        if pos_scale is not None:
            area *= np.prod(pos_scale)
        area_list.append(float(area))

        # Calculate position - use centroid if area > 0, otherwise use bbox center
        if area > 0:
            pos = measure.centroid(seg_crop, spacing=pos_scale)  # type: ignore
            pos += seg_bbox[: ndim - 1] * (pos_scale if pos_scale is not None else 1)
        else:
            # Use bbox center when area is 0
            pos = np.array(
                [(seg_bbox[d] + seg_bbox[d + ndim - 1]) / 2 for d in range(ndim - 1)]
            )
        pos_list.append(pos)

    return {NodeAttr.AREA.value: area_list, NodeAttr.POS.value: pos_list}


def compute_node_attrs_from_pixels(
    pixels: list[tuple[np.ndarray, ...]] | None, ndim: int, scale: list[float] | None
) -> dict[str, list[Any]]:
    """
    Compute node attributes (area and pos) from pixel coordinates.
    Parameters
    ----------
    pixels : list[tuple[np.ndarray, ...]]
        List of pixel coordinates for each node.
    ndim : int
        Number of dimensions (2D or 3D).
    scale : list[float] | None
        Scale factors for each dimension.

    Returns
    -------
    dict[str, list[Any]]
        A dictionary containing the computed node attributes ('area' and 'pos').
    """
    if pixels is None:
        return {}

    # Convert pixels to masks first
    masks = []
    for pix in pixels:
        mask, _ = pixels_to_td_mask(pix, ndim, scale)
        masks.append(mask)

    # Reuse the from_masks function to compute attributes
    return compute_node_attrs_from_masks(masks, ndim, scale)


def create_empty_sql_graph(database: str, position_attrs: list[str]) -> td.graph.SQLGraph:
    """
    Create an empty tracksdata SQL graph with standard node and edge attributes.
    Parameters
    ----------
    database : str
        Path to the SQLite database file, e.g. ':memory:' for in-memory database.
    position_attrs : list[str]
        List of position attribute names, e.g. ['pos'] or ['x', 'y', 'z'].

    Returns
    -------
    td.graph.SQLGraph
        An empty tracksdata SQL graph with standard node and edge attributes.
    """
    kwargs = {
        "drivername": "sqlite",
        "database": database,
        "overwrite": True,
    }
    graph_sql = td.graph.SQLGraph(**kwargs)

    if "pos" in position_attrs:
        graph_sql.add_node_attr_key(NodeAttr.POS.value, default_value=None)
    else:
        if "x" in position_attrs:
            graph_sql.add_node_attr_key("x", default_value=0)
        if "y" in position_attrs:
            graph_sql.add_node_attr_key("y", default_value=0)
        if "z" in position_attrs:
            graph_sql.add_node_attr_key("z", default_value=0)
    graph_sql.add_node_attr_key(NodeAttr.AREA.value, default_value=0.0)
    graph_sql.add_node_attr_key(NodeAttr.TRACK_ID.value, default_value=0)
    graph_sql.add_node_attr_key(td.DEFAULT_ATTR_KEYS.SOLUTION, default_value=1)
    graph_sql.add_node_attr_key(td.DEFAULT_ATTR_KEYS.MASK, default_value=None)
    graph_sql.add_node_attr_key(td.DEFAULT_ATTR_KEYS.BBOX, default_value=None)
    graph_sql.add_edge_attr_key(EdgeAttr.IOU.value, default_value=0)
    graph_sql.add_edge_attr_key(td.DEFAULT_ATTR_KEYS.SOLUTION, default_value=1)

    return graph_sql


def create_empty_graphview_graph(
    database: str, position_attrs: list[str]
) -> td.graph.GraphView:
    """
    Create an empty tracksdata GraphView with standard node and edge attributes.
    Parameters
    ----------
    database : str
        Path to the SQLite database file, e.g. ':memory:' for in-memory database.
    position_attrs : list[str]
        List of position attribute names, e.g. ['pos'] or ['x', 'y', 'z'].

    Returns
    -------
    td.graph.GraphView
        An empty tracksdata GraphView with standard node and edge attributes.
    """
    graph_sql = create_empty_sql_graph(database, position_attrs)

    graph_td_sub = graph_sql.filter(
        td.NodeAttr(td.DEFAULT_ATTR_KEYS.SOLUTION) == 1,
        td.EdgeAttr(td.DEFAULT_ATTR_KEYS.SOLUTION) == 1,
    ).subgraph()

    return graph_td_sub
