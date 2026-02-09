import logging

import networkx as nx
import numpy as np

from .iou import add_iou
from .utils import add_cand_edges, nodes_from_points_list, nodes_from_segmentation

logger = logging.getLogger(__name__)


def compute_graph_from_seg(
    segmentation: np.ndarray,
    max_edge_distance: float,
    iou: bool = False,
    scale: list[float] | None = None,
) -> nx.DiGraph:
    """Construct a candidate graph from a segmentation array. Nodes are placed at the
    centroid of each segmentation and edges are added for all nodes in adjacent frames
    within max_edge_distance.

    Args:
        segmentation (np.ndarray): A numpy array with integer labels and dimensions
            (t, [z], y, x).
        max_edge_distance (float): Maximum distance that objects can travel between
            frames. All nodes with centroids within this distance in adjacent frames
            will by connected with a candidate edge.
        iou (bool, optional): Whether to include IOU on the candidate graph.
            Defaults to False.
        scale (list[float] | None, optional): The scale of the segmentation data.
            Will be used to rescale the point locations and attribute computations.
            Defaults to None, which implies the data is isotropic.

    Returns:
        nx.DiGraph: A candidate graph that can be passed to the motile solver
    """
    # add nodes
    cand_graph, node_frame_dict = nodes_from_segmentation(segmentation, scale=scale)
    # safety check for duplicate nodes with early exit for efficiency
    seen = set()
    for values in node_frame_dict.values():
        for x in set(values):
            if x in seen:
                logger.info(
                    "Duplicate values are found among nodes, segmentation will be "
                    "relabeled"
                )
                raise ValueError("Duplicate values found among nodes")
            seen.add(x)

    logger.info("Candidate nodes: %d", cand_graph.number_of_nodes())

    # add edges
    add_cand_edges(
        cand_graph,
        max_edge_distance=max_edge_distance,
        node_frame_dict=node_frame_dict,
    )
    if iou:
        # Scale does not matter to IOU, because both numerator and denominator
        # are scaled by the anisotropy.
        add_iou(cand_graph, segmentation, node_frame_dict)

    logger.info("Candidate edges: %d", cand_graph.number_of_edges())

    return cand_graph


def compute_graph_from_points_list(
    points_list: np.ndarray,
    max_edge_distance: float,
    scale: list[float] | None = None,
) -> nx.DiGraph:
    """Construct a candidate graph from a points list.

    Args:
        points_list (np.ndarray): An NxD numpy array with N points and D
            (3 or 4) dimensions. Dimensions should be in order  (t, [z], y, x).
        max_edge_distance (float): Maximum distance that objects can travel between
            frames. All nodes with centroids within this distance in adjacent frames
            will by connected with a candidate edge.
        scale (list[float] | None, optional): Amount to scale the points in each
            dimension. Only needed if the provided points are in "voxel" coordinates
            instead of world coordinates. Defaults to None, which implies the data is
            isotropic.

    Returns:
        nx.DiGraph: A candidate graph that can be passed to the motile solver.
    """
    # add nodes
    cand_graph, node_frame_dict = nodes_from_points_list(points_list, scale=scale)
    logger.info("Candidate nodes: %d", cand_graph.number_of_nodes())
    # add edges
    add_cand_edges(
        cand_graph,
        max_edge_distance=max_edge_distance,
        node_frame_dict=node_frame_dict,
    )
    return cand_graph
