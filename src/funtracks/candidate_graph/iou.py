from itertools import product

import networkx as nx
import numpy as np
from tqdm import tqdm

from .utils import _compute_node_frame_dict


def _compute_ious(frame1: np.ndarray, frame2: np.ndarray) -> list[tuple[int, int, float]]:
    """Compute label IOUs between two label arrays of the same shape. Ignores background
    (label 0).

    Args:
        frame1 (np.ndarray): Array with integer labels
        frame2 (np.ndarray): Array with integer labels

    Returns:
        list[tuple[int, int, float]]: List of tuples of label in frame 1, label in
            frame 2, and iou values. Labels that have no overlap are not included.
    """
    frame1 = frame1.flatten()
    frame2 = frame2.flatten()
    # get indices where both are not zero (ignore background)
    # this speeds up computation significantly
    non_zero_indices = np.logical_and(frame1, frame2)
    flattened_stacked = np.array([frame1[non_zero_indices], frame2[non_zero_indices]])

    values, counts = np.unique(flattened_stacked, axis=1, return_counts=True)
    frame1_values, frame1_counts = np.unique(frame1, return_counts=True)
    frame1_label_sizes = dict(zip(frame1_values, frame1_counts, strict=True))
    frame2_values, frame2_counts = np.unique(frame2, return_counts=True)
    frame2_label_sizes = dict(zip(frame2_values, frame2_counts, strict=True))
    ious: list[tuple[int, int, float]] = []
    for index in range(values.shape[1]):
        pair = values[:, index]
        intersection = counts[index]
        id1, id2 = pair
        union = frame1_label_sizes[id1] + frame2_label_sizes[id2] - intersection
        ious.append((id1, id2, intersection / union))
    return ious


def _get_iou_dict(segmentation, multiseg=False) -> dict[int, dict[int, float]]:
    """Get all ious values for the provided segmentations (all frames).
    Will return as map from node_id -> dict[node_id] -> iou for easy
    navigation when adding to candidate graph.

    Args:
        segmentation (np.ndarray): Segmentations that were used to create cand_graph.
            Has shape ([h], t, [z], y, x), where h is the number of hypotheses
            if multiseg is True.
        multiseg (bool): Flag indicating if the provided segmentation contains
            multiple hypothesis segmentations. Defaults to False.

    Returns:
        dict[int, dict[int, float]]: A map from node id to another dictionary, which
            contains node_ids to iou values.
    """
    iou_dict: dict[int, dict[int, float]] = {}
    hypo_pairs: list[tuple[int, ...]] = [(0, 0)]
    if multiseg:
        num_hypotheses = segmentation.shape[0]
        if num_hypotheses > 1:
            hypo_pairs = list(product(range(num_hypotheses), repeat=2))
    else:
        segmentation = np.expand_dims(segmentation, 0)

    for frame in range(segmentation.shape[1] - 1):
        for hypo1, hypo2 in hypo_pairs:
            seg1 = segmentation[hypo1][frame]
            seg2 = segmentation[hypo2][frame + 1]
            ious = _compute_ious(seg1, seg2)
            for label1, label2, iou in ious:
                if label1 not in iou_dict:
                    iou_dict[label1] = {}
                iou_dict[label1][label2] = iou
    return iou_dict


def add_iou(
    cand_graph: nx.DiGraph,
    segmentation: np.ndarray,
    node_frame_dict: dict[int, list[int]] | None = None,
    multiseg=False,
) -> None:
    """Add IOU to the candidate graph.

    Args:
        cand_graph (nx.DiGraph): Candidate graph with nodes and edges already populated
        segmentation (np.ndarray): segmentation that was used to create cand_graph.
            Has shape ([h], t, [z], y, x), where h is the number of hypotheses if
            multiseg is True.
        node_frame_dict(dict[int, list[Any]] | None, optional): A mapping from
            time frames to nodes in that frame. Will be computed if not provided,
            but can be provided for efficiency (e.g. after running
            nodes_from_segmentation). Defaults to None.
        multiseg (bool): Flag indicating if the given segmentation is actually multiple
            stacked segmentations. Defaults to False.
    """
    if node_frame_dict is None:
        node_frame_dict = _compute_node_frame_dict(cand_graph)
    frames = sorted(node_frame_dict.keys())
    ious = _get_iou_dict(segmentation, multiseg=multiseg)
    for frame in tqdm(frames):
        if frame + 1 not in node_frame_dict:
            continue
        next_nodes = node_frame_dict[frame + 1]
        for node_id in node_frame_dict[frame]:
            for next_id in next_nodes:
                iou = ious.get(node_id, {}).get(next_id, 0)
                if (node_id, next_id) in cand_graph.edges:
                    cand_graph.edges[(node_id, next_id)]["iou"] = iou
