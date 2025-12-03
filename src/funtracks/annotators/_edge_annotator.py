from __future__ import annotations

import warnings
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

from funtracks.actions.add_delete_edge import AddEdge
from funtracks.actions.update_segmentation import UpdateNodeSeg
from funtracks.features import Feature, IoU

from ._compute_ious import _compute_ious
from ._graph_annotator import GraphAnnotator

if TYPE_CHECKING:
    from funtracks.actions import BasicAction
    from funtracks.data_model import Tracks

DEFAULT_IOU_KEY = "iou"


class EdgeAnnotator(GraphAnnotator):
    """Manages edge features computed from segmentations or endpoint positions.

    The possible features include:
    - Intersection over Union (IoU)

    Args:
        tracks (Tracks): The tracks to manage the edge features on
    """

    @classmethod
    def can_annotate(cls, tracks) -> bool:
        """Check if this annotator can annotate the given tracks.

        Requires segmentation data to be present.

        Args:
            tracks: The tracks to check compatibility with

        Returns:
            True if tracks have segmentation, False otherwise
        """
        return tracks.segmentation is not None

    @classmethod
    def get_available_features(cls, ndim: int = 3) -> dict[str, Feature]:
        """Get all features that can be computed by this annotator.

        Returns features with default keys. Custom keys can be specified at
        initialization time.

        Args:
            ndim: Total number of dimensions including time (unused for this annotator,
                kept for API consistency). Defaults to 3.

        Returns:
            Dictionary mapping feature keys to Feature definitions.
        """
        return {DEFAULT_IOU_KEY: IoU()}

    def __init__(self, tracks: Tracks) -> None:
        self.iou_key = DEFAULT_IOU_KEY
        # Build features dict with custom key
        feats = {} if tracks.segmentation is None else {DEFAULT_IOU_KEY: IoU()}
        super().__init__(tracks, feats)

    def compute(self, feature_keys: list[str] | None = None) -> None:
        """Compute the currently included features and add them to the tracks.

        Args:
            feature_keys: Optional list of specific feature keys to compute.
                If None, computes all currently active features. Keys not in
                self.features (not enabled) are ignored.
        """
        # Can only compute features if segmentation is present
        if self.tracks.segmentation is None:
            return

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

    def update(self, action: BasicAction):
        """Update the edge features based on the action.

        Only responds to AddEdge and UpdateNodeSeg actions that affect edge IoU.

        Args:
            action (BasicAction): The action that triggered this update
        """
        # Only update for actions that change edges or segmentation
        if not isinstance(action, (AddEdge, UpdateNodeSeg)):
            return

        # Can only compute features if segmentation is present
        if self.tracks.segmentation is None:
            return

        if self.iou_key not in self.features:
            return

        # Get edges to update based on action type
        if isinstance(action, AddEdge):
            edges_to_update = [action.edge]
        else:  # UpdateNodeSeg
            # Get all incident edges to the modified node
            node = action.node
            edges_to_update = list(self.tracks.graph.in_edges(node)) + list(
                self.tracks.graph.out_edges(node)
            )

        # Update IoU for each edge
        for edge in edges_to_update:
            source, target = edge
            start_time = self.tracks.get_time(source)
            end_time = self.tracks.get_time(target)
            start_seg = self.tracks.segmentation[start_time]
            end_seg = self.tracks.segmentation[end_time]
            masked_start = np.where(start_seg == source, source, 0)
            masked_end = np.where(end_seg == target, target, 0)
            if np.max(masked_start) == 0 or np.max(masked_end) == 0:
                warnings.warn(
                    f"Cannot find label {source} in frame {start_time} or label {target} "
                    f"in frame {end_time}: updating edge IOU value to 0",
                    stacklevel=2,
                )
                self.tracks._set_edge_attr(edge, self.iou_key, 0)
            else:
                iou_list = _compute_ious(masked_start, masked_end)
                iou = 0 if len(iou_list) == 0 else iou_list[0][2]
                self.tracks._set_edge_attr(edge, self.iou_key, iou)

    def change_key(self, old_key: str, new_key: str) -> None:
        """Rename a feature key in this annotator.

        Overrides base implementation to also update the iou_key instance variable.

        Args:
            old_key: Existing key to rename.
            new_key: New key to replace it with.

        Raises:
            KeyError: If old_key does not exist.
        """
        # Call base implementation to update all_features
        super().change_key(old_key, new_key)

        # Update iou_key if it matches
        if self.iou_key == old_key:
            self.iou_key = new_key
