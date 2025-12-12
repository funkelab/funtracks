import tempfile
import uuid
from collections.abc import Sequence
from typing import Any

import numpy as np
import polars as pl
import tracksdata as td
from polars.testing import assert_frame_equal
from skimage import measure
from tracksdata.nodes._mask import Mask


def create_empty_graphview_graph(
    with_pos: bool = False,
    with_track_id: bool = False,
    with_area: bool = False,
    with_iou: bool = False,
    database: str | None = None,
    position_attrs: list[str] | None = None,
) -> td.graph.GraphView:
    """
    Create an empty tracksdata GraphView with standard node and edge attributes.
    Parameters
    ----------
    with_pos : bool
        Whether to include position attributes.
    with_track_id : bool
        Whether to include track ID attribute.
    with_area : bool
        Whether to include area attribute.
    with_iou : bool
        Whether to include IOU attribute.
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

    kwargs = {
        "drivername": "sqlite",
        "database": database,
        "overwrite": True,
    }
    graph_sql = td.graph.SQLGraph(**kwargs)

    if with_pos:
        if "pos" in position_attrs:
            graph_sql.add_node_attr_key("pos", default_value=None)
        else:
            if "x" in position_attrs:
                graph_sql.add_node_attr_key("x", default_value=0)
            if "y" in position_attrs:
                graph_sql.add_node_attr_key("y", default_value=0)
            if "z" in position_attrs:
                graph_sql.add_node_attr_key("z", default_value=0)
    if with_area:
        graph_sql.add_node_attr_key("area", default_value=0.0)
    if with_track_id:
        graph_sql.add_node_attr_key("track_id", default_value=0)
    graph_sql.add_node_attr_key(td.DEFAULT_ATTR_KEYS.SOLUTION, default_value=1)
    # TODO Teun: segmentation
    # graph_sql.add_node_attr_key(td.DEFAULT_ATTR_KEYS.MASK, default_value=None)
    # graph_sql.add_node_attr_key(td.DEFAULT_ATTR_KEYS.BBOX, default_value=None)
    if with_iou:
        graph_sql.add_edge_attr_key("iou", default_value=0)
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

    # TODO Teun: enable this when segmentation/masks are part of node_attrs
    # assert_frame_equal(
    #     node_attrs1.drop("mask"),
    #     node_attrs2.drop("mask"),
    #     check_column_order=check_column_order,
    #     check_row_order=check_row_order,
    # )
    # for node in node_attrs1["node_id"]:
    #     mask1 = node_attrs1.filter(pl.col("node_id") == node)["mask"].item()
    #     mask2 = node_attrs2.filter(pl.col("node_id") == node)["mask"].item()
    #     assert np.array_equal(mask1.bbox, mask2.bbox)
    #     assert np.array_equal(mask1.mask, mask2.mask)
    assert_frame_equal(
        node_attrs1,
        node_attrs2,
        check_column_order=check_column_order,
        check_row_order=check_row_order,
    )


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


def td_get_ancestors(graph, node_id):
    """Get ancestors of a node in a tracksdata graph.

    Args:
        graph: A tracksdata graph
        node_id: Node ID to get ancestors for
    """

    ancestors = set()
    to_visit = [node_id]

    while to_visit:
        current_node = to_visit.pop()
        predecessors = graph.predecessors(current_node)
        for pred in predecessors:
            if pred not in ancestors:
                ancestors.add(pred)
                to_visit.append(pred)

    return ancestors


def td_get_single_attr_from_edge(graph, edge: tuple[int, int], attrs: Sequence[str]):
    """Get a single attribute from a edge in a tracksdata graph."""

    # TODO Teun: do graph.edge_id()
    # TODO Teun: AND do edge_attrs(key) directly to prevent loading all attributes
    item = graph.filter(node_ids=[edge[0], edge[1]]).edge_attrs()[attrs].item()
    return item
