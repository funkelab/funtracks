from __future__ import annotations

import warnings
from collections import defaultdict
from typing import TYPE_CHECKING

from funtracks.actions.add_delete_edge import AddEdge
from funtracks.actions.update_segmentation import UpdateNodeSeg
from funtracks.features import Feature, IoU

from ._graph_annotator import GraphAnnotator, _derive_mask_prefix

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
        return tracks.segmentation_shape is not None

    @classmethod
    def create_annotators(cls, tracks) -> list[GraphAnnotator]:
        """Create one EdgeAnnotator per mask feature in tracks.features.

        Each instance gets a mask_attr and iou_key derived from the mask key.

        Args:
            tracks: The tracks to create annotators for

        Returns:
            List of EdgeAnnotator instances (one per mask feature)
        """
        if not cls.can_annotate(tracks):
            return []

        mask_features = [
            key
            for key, feat in tracks.features.items()
            if feat.get("value_type") == "mask"
        ]

        if not mask_features:
            # No mask features in FeatureDict — fall back to single default instance
            return [cls(tracks)]

        annotators: list[GraphAnnotator] = []
        for mask_key in mask_features:
            prefix = _derive_mask_prefix(mask_key)
            iou_key = f"{prefix}{DEFAULT_IOU_KEY}"
            annotators.append(cls(tracks, mask_attr=mask_key, iou_key=iou_key))

        return annotators

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

    def __init__(
        self,
        tracks: Tracks,
        mask_attr: str = "mask",
        iou_key: str = DEFAULT_IOU_KEY,
    ) -> None:
        self.mask_attr = mask_attr
        self.iou_key = iou_key
        # Build features dict with custom key
        feats = {} if tracks.segmentation_shape is None else {self.iou_key: IoU()}
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

        # TODO: add skip edges
        if self.iou_key in keys_to_compute:
            nodes_by_frame = defaultdict(list)
            for n in self.tracks.graph.node_ids():
                nodes_by_frame[self.tracks.get_time(n)].append(n)

            for t in range(self.tracks.segmentation.shape[0] - 1):
                nodes_in_t = nodes_by_frame[t]
                edges = []
                for node in nodes_in_t:
                    for succ in self.tracks.graph.successors(node):
                        edges.append((node, succ))
                self._iou_update(edges)

    def _iou_update(
        self,
        edges: list[tuple[int, int]],
    ) -> None:
        """Perform the IoU computation and update all feature values for a
        list of edges.

        Args:
            edges (list[tuple[int, int]]): A list of edges between two frames
        """
        for edge in edges:
            source, target = edge
            mask1 = self.tracks.graph.nodes[source][self.mask_attr]
            mask2 = self.tracks.graph.nodes[target][self.mask_attr]
            iou = mask1.iou(mask2)
            self.tracks._set_edge_attr(edge, self.iou_key, iou)

    def update(self, action: BasicAction):
        """Update the edge features based on the action.

        Only responds to AddEdge and UpdateNodeSeg actions that affect edge IoU.

        Args:
            action (BasicAction): The action that triggered this update
        """
        # Only update for actions that change edges or segmentation
        if not isinstance(action, (AddEdge, UpdateNodeSeg)):
            return

        # Check if the action affects this annotator's mask attribute
        if isinstance(action, UpdateNodeSeg) and action.mask_key != self.mask_attr:
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
            modified_node = action.node
            edges_to_update = []
            for pred in self.tracks.graph.predecessors(modified_node):
                edges_to_update.append((pred, modified_node))
            for succ in self.tracks.graph.successors(modified_node):
                edges_to_update.append((modified_node, succ))

        # Update IoU for each edge
        for edge in edges_to_update:
            source, target = edge
            mask1 = self.tracks.graph.nodes[source][self.mask_attr]
            mask2 = self.tracks.graph.nodes[target][self.mask_attr]
            if mask1.mask.sum() == 0 or mask2.mask.sum() == 0:
                empty_node = source if mask1.mask.sum() == 0 else target
                frame = self.tracks.get_time(empty_node)
                warnings.warn(
                    f"Cannot find label {empty_node} in frame {frame}"
                    f": updating edge IOU value to 0",
                    stacklevel=2,
                )
                self.tracks._set_edge_attr(edge, self.iou_key, 0.0)
            else:
                iou = mask1.iou(mask2)
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
