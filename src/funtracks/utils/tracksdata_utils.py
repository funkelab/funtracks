import tempfile
import uuid
from collections.abc import Sequence
from typing import Any

import networkx as nx
import numpy as np
import polars as pl
import scipy.ndimage as ndi
import tracksdata as td
from polars.testing import assert_frame_equal
from skimage import measure
from tracksdata.nodes._mask import Mask


def create_empty_graphview_graph(
    node_attributes: list[str] | None = None,
    edge_attributes: list[str] | None = None,
    node_default_values: list[Any] | None = None,
    edge_default_values: list[Any] | None = None,
    database: str | None = None,
    position_attrs: list[str] | None = None,
) -> td.graph.GraphView:
    """
    Create an empty tracksdata GraphView with standard node and edge attributes.
    Parameters
    ----------
    node_attributes : list[str] | None
        List of node attribute names to include. (providing time attribute not necessary)
    edge_attributes : list[str] | None
        List of edge attribute names to include.
    node_default_values : list[Any] | None
        List of default values for each node attribute.
        Must match length of node_attributes.
    edge_default_values : list[Any] | None
        List of default values for each edge attribute.
        Must match length of edge_attributes.
    database : str | None
        Path to the SQLite database file. If None, creates a unique temporary file.
        Use ':memory:' for in-memory database (may cause issues with pickling in pytest).
    position_attrs : list[str] | None
        List of position attribute names, e.g. ['pos'] or ['x', 'y', 'z'].
        Defaults to ['pos'] if None.

    Returns
    -------
    td.graph.GraphView
        An empty tracksdata GraphView with standard node and edge attributes.
    """
    if position_attrs is None:
        position_attrs = ["pos"]

    # Generate unique database path if not specified
    if database is None:
        temp_dir = tempfile.gettempdir()
        unique_id = uuid.uuid4().hex[:8]
        database = f"{temp_dir}/funtracks_test_{unique_id}.db"

    if node_default_values is not None:
        assert len(node_default_values) == len(node_attributes or []), (
            "Length of node_default_values must match length of node_attributes"
        )
    else:
        node_default_values = [0.0] * len(node_attributes or [])

    if edge_default_values is not None:
        assert len(edge_default_values) == len(edge_attributes or []), (
            "Length of edge_default_values must match length of edge_attributes"
        )
    else:
        edge_default_values = [0.0] * len(edge_attributes or [])

    kwargs = {
        "drivername": "sqlite",
        "database": database,
        "overwrite": True,
    }
    graph_sql = td.graph.SQLGraph(**kwargs)

    # Add standard node and edge attributes
    if "pos" in (node_attributes or []) or any(
        attr in (node_attributes or []) for attr in position_attrs
    ):
        if "pos" in position_attrs:
            graph_sql.add_node_attr_key("pos", default_value=None)
        else:
            if "x" in position_attrs:
                graph_sql.add_node_attr_key("x", default_value=0.0)
            if "y" in position_attrs:
                graph_sql.add_node_attr_key("y", default_value=0.0)
            if "z" in position_attrs:
                graph_sql.add_node_attr_key("z", default_value=0.0)

    if "mask" in (node_attributes or []):
        graph_sql.add_node_attr_key("mask", default_value=None)
    if "bbox" in (node_attributes or []):
        graph_sql.add_node_attr_key("bbox", default_value=None)

    for attr in node_attributes or []:
        if attr not in graph_sql.node_attr_keys():
            graph_sql.add_node_attr_key(
                attr,
                default_value=node_default_values[(node_attributes or []).index(attr)],
            )

    for attr in edge_attributes or []:
        if attr not in graph_sql.edge_attr_keys():
            graph_sql.add_edge_attr_key(
                attr,
                default_value=edge_default_values[(edge_attributes or []).index(attr)],
            )
    graph_sql.add_node_attr_key(td.DEFAULT_ATTR_KEYS.SOLUTION, default_value=1)
    graph_sql.add_edge_attr_key(td.DEFAULT_ATTR_KEYS.SOLUTION, default_value=1)

    graph_td_sub = graph_sql.filter(
        td.NodeAttr(td.DEFAULT_ATTR_KEYS.SOLUTION) == 1,
        td.EdgeAttr(td.DEFAULT_ATTR_KEYS.SOLUTION) == 1,
    ).subgraph()

    return graph_td_sub


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

    # Check all fields, except masks
    assert_frame_equal(
        node_attrs1.drop("mask"),
        node_attrs2.drop("mask"),
        check_column_order=check_column_order,
        check_row_order=check_row_order,
    )
    # Check masks separately
    for node in node_attrs1["node_id"]:
        mask1 = node_attrs1.filter(pl.col("node_id") == node)["mask"].item()
        mask2 = node_attrs2.filter(pl.col("node_id") == node)["mask"].item()
        assert np.array_equal(mask1.bbox, mask2.bbox)
        assert np.array_equal(mask1.mask, mask2.mask)


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

    return {"area": area_list, "pos": pos_list}


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


def td_mask_to_pixels(mask: Mask, time: int, ndim: int) -> tuple[np.ndarray, ...]:
    """
    Convert tracksdata mask to pixel coordinates.

    This is the inverse of pixels_to_td_mask.

    Args:
        mask: Tracksdata Mask object with .mask (boolean array) and .bbox attributes
        time: Time point for this mask
        ndim: Number of dimensions (3 for 2D+time, 4 for 3D+time)

    Returns:
        Tuple of numpy arrays: (time_array, *spatial_coords)
        For 2D: (t, y, x) where each is a 1D array of pixel coordinates
        For 3D: (t, z, y, x) where each is a 1D array of pixel coordinates

    Example:
        >>> mask = Mask(np.array([[True, False], [False, True]]),
        ...             bbox=np.array([10, 20, 12, 22]))
        >>> pixels = td_mask_to_pixels(mask, time=5, ndim=3)
        >>> # Returns: (array([5, 5]), array([10, 11]), array([20, 21]))
    """
    spatial_dims = ndim - 1

    # Find all True pixels in the local mask
    local_coords = np.nonzero(mask.mask)

    # Convert local coordinates to global coordinates by adding bbox offset
    global_coords = []
    for dim in range(spatial_dims):
        global_coords.append(local_coords[dim] + mask.bbox[dim])

    # Create time array with same length as spatial coordinates
    num_pixels = len(local_coords[0])
    time_array = np.full(num_pixels, time, dtype=int)

    # Return as tuple: (time, spatial_dim_0, spatial_dim_1, ...)
    return (time_array, *global_coords)


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
        # result_bbox = np.zeros(2 * spatial_dims, dtype=int)
        result_bbox = np.array([0] * spatial_dims + [1] * spatial_dims)
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


def segmentation_to_masks_and_bboxes(
    segmentation: np.ndarray,
) -> list[tuple[int, int, Mask]]:
    """Convert a segmentation array to individual masks and bounding boxes.

    Parameters
    ----------
    segmentation : np.ndarray
        Segmentation array of shape (T, Z, Y, X) or (T, Y, X)
        Each unique value represents a different segment/object.

    Returns
    -------
    list[tuple[int, int, Mask]]
        List of tuples, one per segment, containing:
        - label (int): original label ID
        - time (int): time point
        - mask (Mask): tracksdata Mask object with boolean mask and bbox
    """
    results = []

    # Process each time point
    for t in range(segmentation.shape[0]):
        time_slice = segmentation[t]

        # Get unique labels
        labels = np.unique(time_slice)
        labels = labels[labels != 0]

        # Find objects for each label
        for label in labels:
            # Create binary mask for this label
            binary_mask = time_slice == label

            # Find bounding box using scipy (same as Ultrack uses)
            slices = ndi.find_objects(binary_mask.astype(int))[0]

            if slices is None:
                continue

            # Extract the cropped mask and ensure C-contiguous for blosc2 serialization
            cropped_mask = np.ascontiguousarray(binary_mask[slices])

            # Convert slices to bbox format (min_*, max_*)
            ndim = len(slices)
            bbox = np.array(
                [slices[i].start for i in range(ndim)]  # min coordinates
                + [slices[i].stop for i in range(ndim)]  # max coordinates
            )

            # Create Mask object
            mask = Mask(cropped_mask, bbox=bbox)

            results.append((int(label), t, mask))

    return results


def add_masks_and_bboxes_to_graph(
    graph: td.graph.GraphView,
    segmentation: np.ndarray,
) -> td.graph.GraphView:
    """Add mask and bbox attributes to graph nodes from segmentation.

    Parameters
    ----------
    graph : td.graph.GraphView
        Graph to add attributes to
    segmentation : np.ndarray
        Segmentation array of shape (T, Z, Y, X) or (T, Y, X)

    Returns
    -------
    td.graph.GraphView
        Graph with 'mask' and 'bbox' attributes added to nodes
    """

    # Convert segmentation to masks and bounding boxes
    list_of_masks = segmentation_to_masks_and_bboxes(segmentation)

    # Add 'mask' and 'bbox' attributes to graph nodes
    graph.add_node_attr_key("mask", default_value=None)
    graph.add_node_attr_key("bbox", default_value=None)

    for label, _, mask in list_of_masks:
        if graph.has_node(label):
            graph[label]["mask"] = [mask]
            graph[label]["bbox"] = [mask.bbox]

    return graph


def td_get_single_attr_from_edge(graph, edge: tuple[int, int], attrs: Sequence[str]):
    """Get a single attribute from a edge in a tracksdata graph."""

    # TODO Teun: later opdate to:
    # edge_id = graph.edge_id(edge[0], edge[1])
    # item = graph.edge_attrs(attr_keys=attrs).filter(pl.col("edge_id") == edge_id)
    # .select(attrs).item()once tracksdata supports default values. Right now, polars
    # crashes when the edge attributes have different types. We either need a
    # df = pl.DataFrame(data, strict=False).with_columns(... in line 171 in
    # tracksdata/graph/rx, or tracksdata needs to support default values for missing
    # attributes. The implementation below can be slow for large graphs.
    item = graph.filter(node_ids=[edge[0], edge[1]]).edge_attrs()[attrs].item()
    return item


def td_relabel_nodes(graph, mapping: dict[int, int]) -> td.graph.SQLGraph:
    """Relabel nodes in a tracksdata graph according to a mapping.

    Args:
        graph: A tracksdata graph
        mapping: Dictionary mapping old node IDs to new node IDs

    Returns:
        A new SQLGraph with relabeled nodes
    """

    # For IndexedRXGraph or SQLGraph
    old_graph = graph

    kwargs = {
        "drivername": "sqlite",
        "database": ":memory:",
        "overwrite": True,
    }
    new_graph = td.graph.SQLGraph(**kwargs)

    # Copy attribute key registrations with defaults
    node_defaults = get_node_attr_defaults(graph)
    for key, default_val in node_defaults.items():
        new_graph.add_node_attr_key(key, default_value=default_val)

    edge_defaults = get_edge_attr_defaults(graph)
    for key, default_val in edge_defaults.items():
        new_graph.add_edge_attr_key(key, default_value=default_val)

    # Get all data
    old_nodes = old_graph.node_attrs()
    old_edges = old_graph.edge_attrs()

    # Use the provided mapping
    id_mapping = mapping

    # Add nodes with new IDs
    for row in old_nodes.iter_rows(named=True):
        old_id = row.pop("node_id")
        new_id = id_mapping[old_id]
        new_graph.add_node(row, index=new_id)

    # Add edges with remapped IDs
    for row in old_edges.iter_rows(named=True):
        source_id = id_mapping[row["source_id"]]
        target_id = id_mapping[row["target_id"]]
        attrs = {
            k: v for k, v in row.items() if k not in ["edge_id", "source_id", "target_id"]
        }
        new_graph.add_edge(source_id, target_id, attrs)

    new_graph_sub = new_graph.filter(
        td.NodeAttr(td.DEFAULT_ATTR_KEYS.SOLUTION) == 1,
        td.EdgeAttr(td.DEFAULT_ATTR_KEYS.SOLUTION) == 1,
    ).subgraph()
    return new_graph_sub


def get_node_attr_defaults(graph) -> dict[str, Any]:
    """Get node attribute keys and their default values from SQLGraph."""
    # Unwrap GraphView if needed
    actual_graph = graph._root if hasattr(graph, "_root") else graph

    defaults = {}
    for col in actual_graph.Node.__table__.columns:
        col_name = col.name
        # Skip system columns
        if col_name in ["node_id", "t"]:
            continue

        # Extract default value from SQLAlchemy column
        default_val = None
        if (
            hasattr(col, "default")
            and col.default is not None
            and hasattr(col.default, "arg")
        ):
            default_val = col.default.arg

        defaults[col_name] = default_val
    return defaults


def get_edge_attr_defaults(graph) -> dict[str, Any]:
    """Get edge attribute keys and their default values from SQLGraph."""
    # Unwrap GraphView if needed
    actual_graph = graph._root if hasattr(graph, "_root") else graph

    defaults = {}
    for col in actual_graph.Edge.__table__.columns:
        col_name = col.name
        # Skip system columns
        if col_name in ["edge_id", "source_id", "target_id"]:
            continue

        # Extract default value
        default_val = None
        if (
            hasattr(col, "default")
            and col.default is not None
            and hasattr(col.default, "arg")
        ):
            default_val = col.default.arg

        defaults[col_name] = default_val
    return defaults


def convert_graph_nx_to_td(graph_nx: nx.DiGraph) -> td.graph.GraphView:
    """Convert a NetworkX DiGraph to a tracksdata SQLGraph.

    Args:
        graph_nx: The NetworkX DiGraph to convert.

    Returns:
        A tracksdata SQLGraph representing the same graph.
    """

    # Initialize an empty tracksdata SQLGraph
    kwargs = {
        "drivername": "sqlite",
        "database": ":memory:",
        "overwrite": True,
    }
    graph_td = td.graph.SQLGraph(**kwargs)

    # Get all nodes and edges with attributes
    all_nodes = list(graph_nx.nodes(data=True))
    all_edges = list(graph_nx.edges(data=True))

    # Add node attribute keys to tracksdata graph
    for attr, value in all_nodes[0][1].items():
        if attr not in graph_td.node_attr_keys():
            default_value = None if isinstance(value, list) else 0.0
            graph_td.add_node_attr_key(attr, default_value=default_value)
        else:
            if attr != "t":
                raise Warning(
                    f"Node attribute '{attr}' already exists in "
                    f"tracksdata graph. Skipping addition."
                )
    graph_td.add_node_attr_key(td.DEFAULT_ATTR_KEYS.SOLUTION, default_value=1)

    # Add edge attribute keys to tracksdata graph
    for attr, value in all_edges[0][2].items():
        if attr not in graph_td.edge_attr_keys():
            default_value = None if isinstance(value, list) else 0.0
            graph_td.add_edge_attr_key(attr, default_value=default_value)
        else:
            raise Warning(
                f"Edge attribute '{attr}' already exists in tracksdata graph. "
                f"Skipping addition."
            )
    graph_td.add_edge_attr_key(td.DEFAULT_ATTR_KEYS.SOLUTION, default_value=1)

    # Add node attributes
    for node_id, attrs in all_nodes:
        attrs[td.DEFAULT_ATTR_KEYS.SOLUTION] = 1
        graph_td.add_node(attrs, index=node_id)

    # Add edges
    for source_id, target_id, attrs in all_edges:
        attrs[td.DEFAULT_ATTR_KEYS.SOLUTION] = 1
        graph_td.add_edge(source_id, target_id, attrs)

    # Create subgraph (GraphView) with only solution nodes and edges
    graph_td_sub = graph_td.filter(
        td.NodeAttr(td.DEFAULT_ATTR_KEYS.SOLUTION) == 1,
        td.EdgeAttr(td.DEFAULT_ATTR_KEYS.SOLUTION) == 1,
    ).subgraph()

    return graph_td_sub
