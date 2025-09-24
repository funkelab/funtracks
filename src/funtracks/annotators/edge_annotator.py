from __future__ import annotations

import warnings
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

from ..features.compute_ious import _compute_ious
from ..features.feature import Feature, FeatureType
from .graph_annotator import GraphAnnotator

if TYPE_CHECKING:
    from funtracks.data_model import Tracks


class IOU(Feature):
    def __init__(self):
        super().__init__(
            key="IoU",
            feature_type=FeatureType.EDGE,
            value_type=float,
            valid_ndim=(3, 4),
            recompute=True,
        )


class EdgeAnnotator(GraphAnnotator):
    """A graph annotator extract edge features from segmentations or endpoint positions.

    The possible features include:
    - IoU
    """

    def __init__(self, tracks: Tracks):
        iou_feat = IOU()
        feats = [] if tracks.segmentation is None else [iou_feat]
        super().__init__(tracks, feats)
        self.iou_feat = iou_feat

    def compute(self, add_to_set=False) -> None:
        """Compute the currently included features and add them to the tracks.

        Args:
            add_to_set (bool, optional): Whether to add the Features to the Tracks
            FeatureSet. Defaults to False. Should usually be set to True on the initial
            computation, but False on subsequent re-computations.

        Raises:
            ValueError: If the segmentation is missing from the tracks.
        """
        if self.tracks.segmentation is None:
            raise ValueError("Cannot compute edge features without segmentation.")
        if add_to_set:
            self.add_features_to_set()

        seg = self.tracks.segmentation
        # TODO: add skip edges
        if self.iou_feat in self.features:
            nodes_by_frame = defaultdict(list)
            for n in self.tracks.nodes():
                nodes_by_frame[self.tracks.get_time(n)].append(n)

            for t in range(seg.shape[0] - 1):
                nodes_in_t = nodes_by_frame[t]
                edges = list(self.tracks.graph.out_edges(nodes_in_t))
                self._iou_update(edges, seg[t], seg[t + 1], self.iou_feat)

    def _iou_update(
        self,
        edges: list[tuple[int, int]],
        seg_frame: np.ndarray,
        seg_next_frame: np.ndarray,
        iou_feat: Feature,
    ) -> None:
        """Perform the IOU computation and update all feature values for a
        single pair of frames of segmentation data.

        Args:
            edges (list[tuple[int, int]]): A list of edges between two frames
            seg_frame (np.ndarray): A 2D or 3D numpy array representing the seg for the
                starting time of the edges
            seg_next_frame (np.ndarray): A 2D or 3D numpy array representing the seg for
                the ending time of the edges
            iou_feat (Feature): The feature represeting IOU
        """
        ious = _compute_ious(seg_frame, seg_next_frame)  # list of (id1, id2, iou)
        for id1, id2, iou in ious:
            edge = (id1, id2)
            if edge in edges:
                self.tracks._set_edge_attr(edge, iou_feat.key, iou)
                edges.remove(edge)

        # anything left has IOU of 0
        for edge in edges:
            self.tracks._set_edge_attr(edge, iou_feat.key, 0)

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
        if self.iou_feat in self.features:
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
                self.tracks._set_edge_attr(element, self.iou_feat.key, 0)

            iou_list = _compute_ious(masked_start, masked_end)
            iou = 0 if len(iou_list) == 0 else iou_list[0][2]
            self.tracks._set_edge_attr(element, self.iou_feat.key, iou)
