from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import networkx as nx

from funtracks.features import FeatureDict

from .graph_attributes import NodeAttr
from .tracks import Tracks

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np

    from .tracks import Node


class SolutionTracks(Tracks):
    """Difference from Tracks: every node must have a track_id"""

    def __init__(
        self,
        graph: nx.DiGraph,
        segmentation: np.ndarray | None = None,
        time_attr: str | None = NodeAttr.TIME.value,
        pos_attr: str | tuple[str] | list[str] | None = NodeAttr.POS.value,
        scale: list[float] | None = None,
        ndim: int | None = None,
        features: FeatureDict | None = None,
        existing_features: list[str] | None = None,
    ):
        super().__init__(
            graph,
            segmentation=segmentation,
            time_attr=time_attr,
            pos_attr=pos_attr,
            scale=scale,
            ndim=ndim,
            features=features,
            existing_features=existing_features,
        )
        self.track_annotator = self._get_track_annotator()

        # If track_id is not in existing_features, we need to enable it
        if (
            existing_features is None
            or self.track_annotator.tracklet_key not in existing_features
        ):
            self.enable_features([self.track_annotator.tracklet_key])

    def _get_track_annotator(self):
        """Get the TrackAnnotator instance from the annotator registry.

        Returns:
            TrackAnnotator: The track annotator instance

        Raises:
            RuntimeError: If no TrackAnnotator is registered
        """
        from funtracks.annotators import TrackAnnotator

        for annotator in self.annotators.annotators:
            if isinstance(annotator, TrackAnnotator):
                return annotator
        raise RuntimeError(
            "No TrackAnnotator registered for this SolutionTracks instance"
        )

    @classmethod
    def from_tracks(cls, tracks: Tracks):
        # Get existing features from tracks
        existing_features = list(tracks.features.keys())

        # Check if all nodes have track_id before trusting existing track IDs
        # Short circuit on first missing track_id
        all_nodes_have_track_id = True
        for node in tracks.graph.nodes():
            if tracks.get_node_attr(node, NodeAttr.TRACK_ID.value) is None:
                all_nodes_have_track_id = False
                break

        # Only add track_id to existing_features if ALL nodes have it
        if all_nodes_have_track_id:
            existing_features.append(NodeAttr.TRACK_ID.value)

        return cls(
            tracks.graph,
            segmentation=tracks.segmentation,
            time_attr=None,
            pos_attr=None,
            scale=tracks.scale,
            ndim=tracks.ndim,
            features=tracks.features,
            existing_features=existing_features,
        )

    @property
    def node_id_to_track_id(self) -> dict[Node, int]:
        warnings.warn(
            "node_id_to_track_id property will be removed in funtracks v2. "
            "Use `get_track_id` instead for better performance.",
            DeprecationWarning,
            stacklevel=2,
        )
        return nx.get_node_attributes(self.graph, NodeAttr.TRACK_ID.value)

    def get_next_track_id(self) -> int:
        """Return the next available track_id and update max_tracklet_id in TrackAnnotator

        # TODO: I don't think we need to update the max here, it will get updated if we
        actually add a node automatically.
        """
        annotator = self.track_annotator
        annotator.max_tracklet_id = annotator.max_tracklet_id + 1
        return annotator.max_tracklet_id

    def get_track_id(self, node) -> int:
        track_id = self.get_node_attr(node, NodeAttr.TRACK_ID.value, required=True)
        return track_id

    def export_tracks(self, outfile: Path | str):
        """Export the tracks from this run to a csv with the following columns:
        t,[z],y,x,id,parent_id,track_id
        Cells without a parent_id will have an empty string for the parent_id.
        Whether or not to include z is inferred from self.ndim
        """
        header = ["t", "z", "y", "x", "id", "parent_id", "track_id"]
        if self.ndim == 3:
            header = [header[0]] + header[2:]  # remove z
        with open(outfile, "w") as f:
            f.write(",".join(header))
            for node_id in self.graph.nodes():
                parents = list(self.graph.predecessors(node_id))
                parent_id = "" if len(parents) == 0 else parents[0]
                track_id = self.get_track_id(node_id)
                time = self.get_time(node_id)
                position = self.get_position(node_id)
                row = [
                    time,
                    *position,
                    node_id,
                    parent_id,
                    track_id,
                ]
                f.write("\n")
                f.write(",".join(map(str, row)))

    def get_track_neighbors(
        self, track_id: int, time: int
    ) -> tuple[Node | None, Node | None]:
        """Get the last node with the given track id before time, and the first node
        with the track id after time, if any. Does not assume that a node with
        the given track_id and time is already in tracks, but it can be.

        Args:
            track_id (int): The track id to search for
            time (int): The time point to find the immediate predecessor and successor
                for

        Returns:
            tuple[Node | None, Node | None]: The last node before time with the given
            track id, and the first node after time with the given track id,
            or Nones if there are no such nodes.
        """
        annotator = self.track_annotator
        if (
            track_id not in annotator.tracklet_id_to_nodes
            or len(annotator.tracklet_id_to_nodes[track_id]) == 0
        ):
            return None, None
        candidates = annotator.tracklet_id_to_nodes[track_id]
        candidates.sort(key=lambda n: self.get_time(n))

        pred = None
        succ = None
        for cand in candidates:
            if self.get_time(cand) < time:
                pred = cand
            elif self.get_time(cand) > time:
                succ = cand
                break
        return pred, succ
