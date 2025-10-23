from __future__ import annotations

import warnings
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

from funtracks.features import IoU

from ._compute_ious import _compute_ious
from ._graph_annotator import GraphAnnotator

if TYPE_CHECKING:
    from funtracks.data_model import Tracks


class EdgeAnnotator(GraphAnnotator):
    """Manages edge features computed from segmentations or endpoint positions.

    The possible features include:
    - Intersection over Union (IoU)

    Args:
        tracks (Tracks): The tracks to manage the edge features on
    """

    def __init__(self, tracks: Tracks) -> None:
        self.iou_key = "iou"
        feats = {} if tracks.segmentation is None else {self.iou_key: IoU()}
        super().__init__(tracks, feats)

    def compute(self, feature_keys: list[str] | None = None) -> None:
        """Compute the currently included features and add them to the tracks.

        Args:
            feature_keys: Optional list of specific feature keys to compute.
                If None, computes all currently active features. Keys not in
                self.features (not enabled) are ignored.

        Raises:
            ValueError: If the segmentation is missing from the tracks.
        """
        if self.tracks.segmentation is None:
            raise ValueError("Cannot compute edge features without segmentation.")

        keys_to_compute = self._filter_feature_keys(feature_keys)
        if not keys_to_compute:
            return

        seg = self.tracks.segmentation
        # TODO: add skip edges
        if self.iou_key in keys_to_compute:
            nodes_by_frame = defaultdict(list)
            for n in self.tracks.nodes():
                nodes_by_frame[self.tracks.get_time(n)].append(n)

            for t in range(seg.shape[0] - 1):
                nodes_in_t = nodes_by_frame[t]
                edges = list(self.tracks.graph.out_edges(nodes_in_t))
                self._iou_update(edges, seg[t], seg[t + 1])

    def _iou_update(
        self,
        edges: list[tuple[int, int]],
        seg_frame: np.ndarray,
        seg_next_frame: np.ndarray,
    ) -> None:
        """Perform the IoU computation and update all feature values for a
        single pair of frames of segmentation data.

        Args:
            edges (list[tuple[int, int]]): A list of edges between two frames
            seg_frame (np.ndarray): A 2D or 3D numpy array representing the seg for the
                starting time of the edges
            seg_next_frame (np.ndarray): A 2D or 3D numpy array representing the seg for
                the ending time of the edges
        """
        ious = _compute_ious(seg_frame, seg_next_frame)  # list of (id1, id2, iou)
        for id1, id2, iou in ious:
            edge = (id1, id2)
            if edge in edges:
                self.tracks._set_edge_attr(edge, self.iou_key, iou)
                edges.remove(edge)

        # anything left has IOU of 0
        for edge in edges:
            self.tracks._set_edge_attr(edge, self.iou_key, 0)

    def update(self, element: int | tuple[int, int]):
        """Update the regionprops features for the given node.

        Args:
            element (int | tuple[int, int]): The edge to update. Should be an edge
                and not a node, but has possible edge type to match generic signature.

        Raises:
            ValueError: If the tracks do not have a segmentation
            ValueError: If a node element is passed instead of an edge.
        """
        if self.tracks.segmentation is None:
            raise ValueError("Cannot update edge features without segmentation.")

        if isinstance(element, int):
            raise ValueError(f"EdgeAnnotator update expected an edge, got node {element}")
        if self.iou_key in self.features:
            source, target = element
            start_time = self.tracks.get_time(source)
            end_time = self.tracks.get_time(target)
            start_seg = self.tracks.segmentation[start_time]
            end_seg = self.tracks.segmentation[end_time]
            masked_start = np.where(start_seg == source, source, 0)
            masked_end = np.where(end_seg == target, target, 0)
            if np.max(masked_start) == 0 or np.max(masked_end) == 0:
                warnings.warn(
                    f"Cannot find label {source} in frame {start_time} or label {target} "
                    "in frame {end_time}: updating edge IOU value to 0",
                    stacklevel=2,
                )
                self.tracks._set_edge_attr(element, self.iou_key, 0)

            iou_list = _compute_ious(masked_start, masked_end)
            iou = 0 if len(iou_list) == 0 else iou_list[0][2]
            self.tracks._set_edge_attr(element, self.iou_key, iou)
