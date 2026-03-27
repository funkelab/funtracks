import logging
from typing import Any

import numpy as np
import polars as pl
import tracksdata as td
from scipy.spatial import KDTree
from skimage.measure import regionprops
from tqdm import tqdm
from tracksdata.nodes._mask import Mask

from ..utils.tracksdata_utils import create_empty_graphview_graph

logger = logging.getLogger(__name__)


def nodes_from_segmentation(
    segmentation: np.ndarray,
    scale: list[float] | None = None,
    mask: bool = False,
) -> tuple[td.graph.GraphView, dict[int, list[Any]]]:
    """Extract candidate nodes from a segmentation. Returns a tracksdata graph
    with only nodes, and also a dictionary from frames to node_ids for
    efficient edge adding.

    Each node will have the following attributes:
        - t
        - pos
        - area
        - mask (if mask=True): cropped boolean mask (tracksdata Mask object)
        - bbox (if mask=True): bounding box as [min_0, ..., max_0, ...] int array

    Args:
        segmentation (np.ndarray): A numpy array with integer labels and dimensions
            (t, [z], y, x). Labels must be unique across time, and the label
            will be used as the node id. If the labels are not unique, preprocess
            with funtracks.utils.ensure_unique_labels before calling this function.
        scale (list[float] | None, optional): The scale of the segmentation data in all
            dimensions (including time, which should have a dummy 1 value).
            Will be used to rescale the point locations and attribute computations.
            Defaults to None, which implies the data is isotropic.
        mask (bool, optional): Whether to include mask and bbox attributes for each
            node. Uses regionprop.image (already computed by regionprops at no extra
            cost) and regionprop.bbox. Including them here avoids a separate write
            pass over all nodes later. Defaults to False.

    Returns:
        tuple[td.graph.GraphView, dict[int, list[Any]]]: A candidate graph with only
            nodes, and a mapping from time frames to node ids.
    """
    logger.debug("Extracting nodes from segmentation")

    if scale is None:
        scale = [1] * segmentation.ndim
    else:
        assert len(scale) == segmentation.ndim, (
            f"Scale {scale} should have {segmentation.ndim} dims"
        )

    node_attributes = ["pos", "area", "mask", "bbox"] if mask else ["pos", "area"]
    cand_graph = create_empty_graphview_graph(
        node_attributes=node_attributes,
        position_attrs=["pos"],
        ndim=segmentation.ndim,
    )

    node_frame_dict: dict[int, list[Any]] = {}
    seen_ids: set[int] = set()
    nodes_attrs_list: list[dict] = []
    nodes_id_list: list[int] = []

    for t in tqdm(range(len(segmentation)), desc="Extracting nodes from segmentation"):
        segs = segmentation[t]
        nodes_in_frame: list[int] = []
        props = regionprops(segs, spacing=tuple(scale[1:]))
        for regionprop in props:
            node_id = regionprop.label
            if node_id in seen_ids:
                raise ValueError("Duplicate values found among nodes")
            seen_ids.add(node_id)
            attrs: dict[str, Any] = {
                "t": t,
                "pos": list(regionprop.centroid),
                "area": float(regionprop.area),
            }
            if mask:
                # regionprop.image is the cropped boolean mask within the bbox,
                # already computed by regionprops at no extra cost.
                # regionprop.bbox format: (min_0, ..., min_n, max_0, ..., max_n),
                # which matches the Mask bbox convention.
                bbox = np.array(regionprop.bbox, dtype=np.int64)
                attrs["mask"] = Mask(np.ascontiguousarray(regionprop.image), bbox=bbox)
                attrs["bbox"] = bbox
            nodes_attrs_list.append(attrs)
            nodes_id_list.append(node_id)
            nodes_in_frame.append(node_id)

        if nodes_in_frame:
            node_frame_dict[t] = nodes_in_frame

    if nodes_id_list:
        cand_graph.bulk_add_nodes(nodes=nodes_attrs_list, indices=nodes_id_list)

    return cand_graph, node_frame_dict


def nodes_from_points_list(
    points_list: np.ndarray,
    scale: list[float] | None = None,
) -> tuple[td.graph.GraphView, dict[int, list[Any]]]:
    """Extract candidate nodes from a list of points. Uses the index of the
    point in the list as its unique id.
    Returns a tracksdata graph with only nodes, and also a dictionary from frames to
    node_ids for efficient edge adding.

    Args:
        points_list (np.ndarray): An NxD numpy array with N points and D
            (3 or 4) dimensions. Dimensions should be in order (t, [z], y, x).
        scale (list[float] | None, optional): Amount to scale the points in each
            dimension (including time). Only needed if the provided points are in
            "voxel" coordinates instead of world coordinates. Defaults to None, which
            implies the data is isotropic.

    Returns:
        tuple[td.graph.GraphView, dict[int, list[Any]]]: A candidate graph with only
            nodes, and a mapping from time frames to node ids.
    """
    logger.info("Extracting nodes from points list")

    ndim = points_list.shape[1]

    if scale is not None:
        assert len(scale) == ndim, (
            f"Cannot scale points with {ndim} dims by factor {scale}"
        )
        points_list = points_list * np.array(scale)

    cand_graph = create_empty_graphview_graph(
        node_attributes=["pos"],
        position_attrs=["pos"],
        ndim=ndim,
    )

    node_frame_dict: dict[int, list[Any]] = {}
    nodes_attrs_list: list[dict] = []
    nodes_id_list: list[int] = []

    for i, point in enumerate(points_list):
        t = int(point[0])
        pos = list(point[1:])
        nodes_attrs_list.append({"t": t, "pos": pos})
        nodes_id_list.append(i)
        if t not in node_frame_dict:
            node_frame_dict[t] = []
        node_frame_dict[t].append(i)

    if nodes_id_list:
        cand_graph.bulk_add_nodes(nodes=nodes_attrs_list, indices=nodes_id_list)

    return cand_graph, node_frame_dict


def _compute_node_frame_dict(cand_graph: td.graph.GraphView) -> dict[int, list[Any]]:
    """Compute dictionary from time frames to node ids for candidate graph.

    Args:
        cand_graph (td.graph.GraphView): A tracksdata graph

    Returns:
        dict[int, list[Any]]: A mapping from time frames to lists of node ids.
    """
    node_frame_dict: dict[int, list[Any]] = {}
    df = cand_graph.node_attrs(attr_keys=["node_id", "t"])
    for node_id, t in zip(df["node_id"].to_list(), df["t"].to_list(), strict=True):
        t_int = int(t)
        if t_int not in node_frame_dict:
            node_frame_dict[t_int] = []
        node_frame_dict[t_int].append(node_id)
    return node_frame_dict


def create_kdtree(cand_graph: td.graph.GraphView, node_ids: list[Any]) -> KDTree:
    """Create a kdtree with the given nodes from the candidate graph.
    Will fail if provided node ids are not in the candidate graph.

    Args:
        cand_graph (td.graph.GraphView): A candidate graph
        node_ids (list[Any]): The nodes within the candidate graph to
            include in the KDTree. Useful for limiting to one time frame.
            Must be a list (not a generic iterable) to preserve order for
            correct mapping of query_ball_tree results back to node ids.

    Returns:
        KDTree: A KDTree containing the positions of the given nodes,
            in the same order as node_ids.
    """
    # Fetch all positions in one batch query instead of one SQL lookup per node.
    # Per-node access via cand_graph.nodes[node]["pos"] deserializes the full
    # graph attribute schema on every call, which dominates runtime at scale.
    df = cand_graph.filter(node_ids=node_ids).node_attrs(attr_keys=["node_id", "pos"])
    pos_by_id = dict(zip(df["node_id"].to_list(), df["pos"].to_list(), strict=True))
    positions = [pos_by_id[nid] for nid in node_ids]
    return KDTree(positions)


def add_cand_edges(
    cand_graph: td.graph.GraphView,
    max_edge_distance: float,
    node_frame_dict: None | dict[int, list[Any]] = None,
    iou_dict: dict[int, dict[int, float]] | None = None,
) -> None:
    """Add candidate edges to a candidate graph by connecting all nodes in adjacent
    frames that are closer than max_edge_distance.

    Args:
        cand_graph (td.graph.GraphView): Candidate graph with only nodes populated.
            Will be modified in-place to add edges.
        max_edge_distance (float): Maximum distance that objects can travel between
            frames. All nodes within this distance in adjacent frames will by connected
            with a candidate edge.
        node_frame_dict (dict[int, list[Any]] | None, optional): A mapping from frames
            to node ids. If not provided, it will be computed from cand_graph. Defaults
            to None.
        iou_dict (dict[int, dict[int, float]] | None, optional): Pre-computed IOU
            values as a map from source node_id -> {target node_id -> iou}. When
            provided, the iou value is included directly in each edge dict passed to
            bulk_add_edges, avoiding a separate per-edge update pass after insertion.
            Defaults to None (no iou attribute added).
    """
    logger.info("Extracting candidate edges")
    if not node_frame_dict:
        node_frame_dict = _compute_node_frame_dict(cand_graph)

    if iou_dict is not None:
        cand_graph.add_edge_attr_key("iou", default_value=0.0, dtype=pl.Float64)

    frames = sorted(node_frame_dict.keys())
    prev_node_ids = node_frame_dict[frames[0]]
    prev_kdtree = create_kdtree(cand_graph, prev_node_ids)
    for frame in tqdm(frames, desc="Adding candidate edges"):
        if frame + 1 not in node_frame_dict:
            continue
        next_node_ids = node_frame_dict[frame + 1]
        next_kdtree = create_kdtree(cand_graph, next_node_ids)

        matched_indices = prev_kdtree.query_ball_tree(next_kdtree, max_edge_distance)

        new_edges = []
        for prev_node_id, next_node_indices in zip(
            prev_node_ids, matched_indices, strict=False
        ):
            for next_node_index in next_node_indices:
                next_node_id = next_node_ids[next_node_index]
                edge: dict[str, Any] = {
                    "source_id": prev_node_id,
                    "target_id": next_node_id,
                }
                if iou_dict is not None:
                    edge["iou"] = iou_dict.get(prev_node_id, {}).get(next_node_id, 0.0)
                new_edges.append(edge)

        if new_edges:
            cand_graph.bulk_add_edges(new_edges)

        prev_node_ids = next_node_ids
        prev_kdtree = next_kdtree
