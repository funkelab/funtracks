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

    # TODO Teun: segmentation
    # graph_sql.add_node_attr_key(td.DEFAULT_ATTR_KEYS.MASK, default_value=None)
    # graph_sql.add_node_attr_key(td.DEFAULT_ATTR_KEYS.BBOX, default_value=None)

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

    return new_graph


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
