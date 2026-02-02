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
from tracksdata.nodes._mask import Mask


def to_polars_dtype(dtype_or_value: str | Any) -> pl.DataType:
    """Convert a type string or value to polars dtype.

    Args:
        dtype_or_value: Either a type string ("int", "float", "str", "bool")
                       or a value whose type should be inferred

    Returns:
        Corresponding polars dtype

    Raises:
        ValueError: If the type is not supported

    Examples:
        >>> to_polars_dtype("int")
        Int64
        >>> to_polars_dtype(5)
        Int64
        >>> to_polars_dtype(np.int64(5))
        Int64
        >>> to_polars_dtype("")  # String value
        String
    """
    # Check if it's a known type string first
    type_string_mapping = {
        "str": pl.String,
        "int": pl.Int64,
        "float": pl.Float64,
        "bool": pl.Boolean,
        "datetime": pl.Datetime,
        "date": pl.Date,
    }

    if dtype_or_value in type_string_mapping:
        return type_string_mapping[dtype_or_value]

    # If it's a string but not a type name, try as polars type name (backward compat)
    if isinstance(dtype_or_value, str):
        try:
            return getattr(pl, dtype_or_value)
        except AttributeError:
            # It's a string value, not a type name - return String dtype
            return pl.String

    # Otherwise, infer from the value's type
    if isinstance(dtype_or_value, (bool, np.bool_)):
        return pl.Boolean
    elif isinstance(dtype_or_value, (int, np.integer)):
        return pl.Int64
    elif isinstance(dtype_or_value, (float, np.floating)):
        return pl.Float64
    else:
        raise ValueError(f"Unsupported type: {type(dtype_or_value)}")


def create_empty_graphview_graph(
    node_attributes: list[str] | None = None,
    edge_attributes: list[str] | None = None,
    node_default_values: list[Any] | None = None,
    edge_default_values: list[Any] | None = None,
    database: str | None = None,
    position_attrs: list[str] | None = None,
    ndim: int = 3,
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
    ndim : int
        Number of dimensions including time, so 2D+T dataset has ndim = 3.
        Defaults to 3 (2D+time).

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
            graph_sql.add_node_attr_key("pos", pl.Array(pl.Float64, ndim - 1))
        else:
            if "x" in position_attrs:
                graph_sql.add_node_attr_key("x", default_value=0.0, dtype=pl.Float64)
            if "y" in position_attrs:
                graph_sql.add_node_attr_key("y", default_value=0.0, dtype=pl.Float64)
            if "z" in position_attrs:
                graph_sql.add_node_attr_key("z", default_value=0.0, dtype=pl.Float64)
    if "mask" in (node_attributes or []):
        graph_sql.add_node_attr_key("mask", pl.Object)
    if "bbox" in (node_attributes or []):
        graph_sql.add_node_attr_key("bbox", pl.Array(pl.Int64, 2 * (ndim - 1)))
    if "track_id" in (node_attributes or []):
        graph_sql.add_node_attr_key("track_id", default_value=-1, dtype=pl.Int64)

    for attr in node_attributes or []:
        if attr not in graph_sql.node_attr_keys():
            default_value = node_default_values[(node_attributes or []).index(attr)]
            graph_sql.add_node_attr_key(
                attr,
                default_value=default_value
                if not isinstance(default_value, np.ndarray)
                else None,
                dtype=to_polars_dtype(default_value)
                if not isinstance(default_value, np.ndarray)
                else pl.Array(pl.Float64, len(default_value)),  # type: ignore
            )

    for attr in edge_attributes or []:
        if attr not in graph_sql.edge_attr_keys():
            default_value = edge_default_values[(edge_attributes or []).index(attr)]
            graph_sql.add_edge_attr_key(
                attr,
                default_value=default_value,
                dtype=to_polars_dtype(default_value),
            )
    graph_sql.add_node_attr_key(
        td.DEFAULT_ATTR_KEYS.SOLUTION, default_value=1, dtype=pl.Int64
    )
    graph_sql.add_edge_attr_key(
        td.DEFAULT_ATTR_KEYS.SOLUTION, default_value=1, dtype=pl.Int64
    )

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
        check_dtypes=False,
    )
    # Check masks separately
    for node in node_attrs1["node_id"]:
        mask1 = node_attrs1.filter(pl.col("node_id") == node)["mask"].item()
        mask2 = node_attrs2.filter(pl.col("node_id") == node)["mask"].item()
        assert np.array_equal(mask1.bbox, mask2.bbox)
        assert np.array_equal(mask1.mask, mask2.mask)


def pixels_to_td_mask(
    pix: tuple[np.ndarray, ...],
    ndim: int,
    scale: list[float] | None = None,
    include_area: bool = False,
) -> Mask | tuple[Mask, float]:
    """
    Convert pixel coordinates to tracksdata mask format.

    Args:
        pix: Pixel coordinates for 1 node!
        ndim: Number of dimensions (2D or 3D).
        scale: Scale factors for each dimension, used for area calculation
        include_area: Whether to compute and return the area.

    Returns:
        Mask | tuple[Mask, float]: A tuple containing the
            tracksdata mask and the area if include_area is True.
            Otherwise, just the tracksdata mask.
    """

    if include_area and scale is None:
        raise ValueError("Scale must be provided when include_area is True.")

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
    mask = Mask(mask_array, bbox=bbox)

    if include_area:
        area = np.sum(mask_array)
        if scale is not None:
            area *= np.prod(scale[1:])
        return mask, area
    else:
        return mask


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


def segmentation_to_masks(
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
    list_of_masks = segmentation_to_masks(segmentation)

    # Add 'mask' and 'bbox' attributes to graph nodes
    graph.add_node_attr_key("mask", pl.Object)
    graph.add_node_attr_key("bbox", pl.Array(pl.Int64, 2 * (segmentation.ndim - 1)))

    for label, _, mask in list_of_masks:
        if graph.has_node(label):
            graph[label]["mask"] = [mask]
            graph[label]["bbox"] = [mask.bbox]

    return graph


def td_get_single_attr_from_edge(graph, edge: tuple[int, int], attrs: Sequence[str]):
    """Get a single attribute from a edge in a tracksdata graph."""

    # TODO Teun: later opdate to: graph.edges[edge_id][attr] (after td update)
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

    # Copy attribute key registrations with defaults and dtypes
    node_schemas = graph._node_attr_schemas()
    for key, schema in node_schemas.items():
        if key not in ["node_id", "t"]:  # Skip system columns
            new_graph.add_node_attr_key(
                key, default_value=schema.default_value, dtype=schema.dtype
            )

    edge_schemas = graph._edge_attr_schemas()
    for key, schema in edge_schemas.items():
        if key not in ["edge_id", "source_id", "target_id"]:  # Skip system columns
            new_graph.add_edge_attr_key(
                key, default_value=schema.default_value, dtype=schema.dtype
            )

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
            default_value: Any  # mypy necessities
            dtype: pl.DataType
            if isinstance(value, list):
                # Array type - always use Float64 for numeric arrays from NetworkX
                # since NetworkX doesn't enforce type consistency across nodes
                default_value = None
                dtype = pl.Array(pl.Float64, len(value))
            else:
                # Scalar type - always use Float64 for numeric types from NetworkX
                # since NetworkX doesn't enforce type consistency across nodes
                if isinstance(value, (int, float, np.integer, np.floating)):
                    default_value = 0.0
                    dtype = pl.Float64
                else:
                    default_value = ""
                    dtype = pl.String
            graph_td.add_node_attr_key(attr, default_value=default_value, dtype=dtype)
        else:
            if attr != "t":
                raise Warning(
                    f"Node attribute '{attr}' already exists in "
                    f"tracksdata graph. Skipping addition."
                )
    graph_td.add_node_attr_key(
        td.DEFAULT_ATTR_KEYS.SOLUTION, default_value=1, dtype=pl.Int64
    )

    # Add edge attribute keys to tracksdata graph
    for attr, value in all_edges[0][2].items():
        if attr not in graph_td.edge_attr_keys():
            if isinstance(value, list):
                # Array type - always use Float64 for numeric arrays from NetworkX
                default_value = None
                dtype = pl.Array(pl.Float64, len(value))
            else:
                # Scalar type - always use Float64 for numeric types from NetworkX
                if isinstance(value, (int, float, np.integer, np.floating)):
                    default_value = 0.0
                    dtype = pl.Float64
                else:
                    default_value = ""
                    dtype = pl.String
            graph_td.add_edge_attr_key(attr, default_value=default_value, dtype=dtype)
        else:
            raise Warning(
                f"Edge attribute '{attr}' already exists in tracksdata graph. "
                f"Skipping addition."
            )
    graph_td.add_edge_attr_key(
        td.DEFAULT_ATTR_KEYS.SOLUTION, default_value=1, dtype=pl.Int64
    )

    # Add node attributes
    for node_id, attrs in all_nodes:
        attrs_copy = dict(attrs)
        # Convert lists to numpy arrays to work around tracksdata SQLGraph bug
        # where Python lists with floats get truncated
        for key, value in attrs_copy.items():
            if isinstance(value, list):
                attrs_copy[key] = np.array(value, dtype=np.float64)
        attrs_copy[td.DEFAULT_ATTR_KEYS.SOLUTION] = 1
        graph_td.add_node(attrs_copy, index=node_id)

    # Add edges
    for source_id, target_id, attrs in all_edges:
        attrs_copy = dict(attrs)
        # Convert lists to numpy arrays to work around tracksdata SQLGraph bug
        for key, value in attrs_copy.items():
            if isinstance(value, list):
                attrs_copy[key] = np.array(value, dtype=np.float64)
        attrs_copy[td.DEFAULT_ATTR_KEYS.SOLUTION] = 1
        graph_td.add_edge(source_id, target_id, attrs_copy)

    # Create subgraph (GraphView) with only solution nodes and edges
    graph_td_sub = graph_td.filter(
        td.NodeAttr(td.DEFAULT_ATTR_KEYS.SOLUTION) == 1,
        td.EdgeAttr(td.DEFAULT_ATTR_KEYS.SOLUTION) == 1,
    ).subgraph()

    return graph_td_sub
