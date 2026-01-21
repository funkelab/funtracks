import logging
from collections.abc import Iterable
from typing import Any

import networkx as nx
import numpy as np
from scipy.spatial import KDTree
from skimage.measure import regionprops
from tqdm import tqdm

from funtracks.data_model.graph_attributes import NodeAttr

logger = logging.getLogger(__name__)


def nodes_from_segmentation(
    segmentation: np.ndarray,
    scale: list[float] | None = None,
) -> tuple[nx.DiGraph, dict[int, list[Any]]]:
    """Extract candidate nodes from a segmentation. Returns a networkx graph
    with only nodes, and also a dictionary from frames to node_ids for
    efficient edge adding.

    Each node will have the following attributes (named as in NodeAttrs):
        - time
        - position
        - segmentation id
        - area

    Args:
        segmentation (np.ndarray): A numpy array with integer labels and dimensions
            (t, [z], y, x). Labels must be unique across time, and the label
            will be used as the node id. If the labels are not unique, preprocess
            with funtracks.utils.ensure_ensure_unique_labels before calling this function.
        scale (list[float] | None, optional): The scale of the segmentation data in all
            dimensions (including time, which should have a dummy 1 value).
            Will be used to rescale the point locations and attribute computations.
            Defaults to None, which implies the data is isotropic.

    Returns:
        tuple[nx.DiGraph, dict[int, list[Any]]]: A candidate graph with only nodes,
            and a mapping from time frames to node ids.
    """
    logger.debug("Extracting nodes from segmentation")
    cand_graph = nx.DiGraph()
    # also construct a dictionary from time frame to node_id for efficiency
    node_frame_dict: dict[int, list[Any]] = {}

    if scale is None:
        scale = [
            1,
        ] * segmentation.ndim
    else:
        assert len(scale) == segmentation.ndim, (
            f"Scale {scale} should have {segmentation.ndim} dims"
        )

    for t in tqdm(range(len(segmentation))):
        segs = segmentation[t]
        nodes_in_frame = []
        props = regionprops(segs, spacing=tuple(scale[1:]))
        for regionprop in props:
            node_id = regionprop.label
            attrs = {NodeAttr.TIME.value: t, NodeAttr.AREA.value: regionprop.area}
            attrs[NodeAttr.SEG_ID.value] = regionprop.label
            centroid = regionprop.centroid  # [z,] y, x
            attrs[NodeAttr.POS.value] = centroid
            cand_graph.add_node(node_id, **attrs)
            nodes_in_frame.append(node_id)
        if nodes_in_frame:
            if t not in node_frame_dict:
                node_frame_dict[t] = []
            node_frame_dict[t].extend(nodes_in_frame)
    return cand_graph, node_frame_dict


def nodes_from_points_list(
    points_list: np.ndarray,
    scale: list[float] | None = None,
) -> tuple[nx.DiGraph, dict[int, list[Any]]]:
    """Extract candidate nodes from a list of points. Uses the index of the
    point in the list as its unique id.
    Returns a networkx graph with only nodes, and also a dictionary from frames to
    node_ids for efficient edge adding.

    Args:
        points_list (np.ndarray): An NxD numpy array with N points and D
            (3 or 4) dimensions. Dimensions should be in order (t, [z], y, x).
        scale (list[float] | None, optional): Amount to scale the points in each
            dimension (including time). Only needed if the provided points are in
            "voxel" coordinates instead of world coordinates. Defaults to None, which
            implies the data is isotropic.

    Returns:
        tuple[nx.DiGraph, dict[int, list[Any]]]: A candidate graph with only nodes,
            and a mapping from time frames to node ids.
    """
    cand_graph = nx.DiGraph()
    # also construct a dictionary from time frame to node_id for efficiency
    node_frame_dict: dict[int, list[Any]] = {}
    logger.info("Extracting nodes from points list")

    # scale points
    if scale is not None:
        assert len(scale) == points_list.shape[1], (
            f"Cannot scale points with {points_list.shape[1]} dims by factor {scale}"
        )
        points_list = points_list * np.array(scale)

    # add points to graph
    for i, point in enumerate(points_list):
        # assume t, [z], y, x
        t = point[0]
        pos = list(point[1:])
        node_id = i
        attrs = {
            NodeAttr.TIME.value: t,
            NodeAttr.POS.value: pos,
        }
        cand_graph.add_node(node_id, **attrs)
        if t not in node_frame_dict:
            node_frame_dict[t] = []
        node_frame_dict[t].append(node_id)
    return cand_graph, node_frame_dict


def _compute_node_frame_dict(cand_graph: nx.DiGraph) -> dict[int, list[Any]]:
    """Compute dictionary from time frames to node ids for candidate graph.

    Args:
        cand_graph (nx.DiGraph): A networkx graph

    Returns:
        dict[int, list[Any]]: A mapping from time frames to lists of node ids.
    """
    node_frame_dict: dict[int, list[Any]] = {}
    for node, data in cand_graph.nodes(data=True):
        t = data[NodeAttr.TIME.value]
        if t not in node_frame_dict:
            node_frame_dict[t] = []
        node_frame_dict[t].append(node)
    return node_frame_dict


def create_kdtree(cand_graph: nx.DiGraph, node_ids: Iterable[Any]) -> KDTree:
    """Create a kdtree with the given nodes from the candidate graph.
    Will fail if provided node ids are not in the candidate graph.

    Args:
        cand_graph (nx.DiGraph): A candidate graph
        node_ids (Iterable[Any]): The nodes within the candidate graph to
            include in the KDTree. Useful for limiting to one time frame.

    Returns:
        KDTree: A KDTree containing the positions of the given nodes.
    """
    positions = [cand_graph.nodes[node][NodeAttr.POS.value] for node in node_ids]
    return KDTree(positions)


def add_cand_edges(
    cand_graph: nx.DiGraph,
    max_edge_distance: float,
    node_frame_dict: None | dict[int, list[Any]] = None,
) -> None:
    """Add candidate edges to a candidate graph by connecting all nodes in adjacent
    frames that are closer than max_edge_distance. Also adds attributes to the edges.

    Args:
        cand_graph (nx.DiGraph): Candidate graph with only nodes populated. Will
            be modified in-place to add edges.
        max_edge_distance (float): Maximum distance that objects can travel between
            frames. All nodes within this distance in adjacent frames will by connected
            with a candidate edge.
        node_frame_dict (dict[int, list[Any]] | None, optional): A mapping from frames
            to node ids. If not provided, it will be computed from cand_graph. Defaults
            to None.
    """
    logger.info("Extracting candidate edges")
    if not node_frame_dict:
        node_frame_dict = _compute_node_frame_dict(cand_graph)

    frames = sorted(node_frame_dict.keys())
    prev_node_ids = node_frame_dict[frames[0]]
    prev_kdtree = create_kdtree(cand_graph, prev_node_ids)
    for frame in tqdm(frames):
        if frame + 1 not in node_frame_dict:
            continue
        next_node_ids = node_frame_dict[frame + 1]
        next_kdtree = create_kdtree(cand_graph, next_node_ids)

        matched_indices = prev_kdtree.query_ball_tree(next_kdtree, max_edge_distance)

        for prev_node_id, next_node_indices in zip(
            prev_node_ids, matched_indices, strict=False
        ):
            for next_node_index in next_node_indices:
                next_node_id = next_node_ids[next_node_index]
                cand_graph.add_edge(prev_node_id, next_node_id)

        prev_node_ids = next_node_ids
        prev_kdtree = next_kdtree
