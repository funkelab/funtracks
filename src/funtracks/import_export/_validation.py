from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx
from geff.validate.segmentation import has_seg_ids_at_coords

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

    # Check if seg_id exists
    if SEG_KEY in last_node_data:
        seg_id = int(last_node_data[SEG_KEY])
    else:
        # If no seg_id, assume it matches node_id
        seg_id = int(last_node_id)

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
