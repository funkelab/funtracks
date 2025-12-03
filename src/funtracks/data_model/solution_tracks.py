from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np

from funtracks.features import FeatureDict

from .tracks import Tracks

if TYPE_CHECKING:
    from pathlib import Path

    from funtracks.annotators import TrackAnnotator

    from .tracks import Node


class SolutionTracks(Tracks):
    """Difference from Tracks: every node must have a track_id"""

    def __init__(
        self,
        graph: nx.DiGraph,
        segmentation: np.ndarray | None = None,
        time_attr: str | None = None,
        pos_attr: str | tuple[str] | list[str] | None = None,
        tracklet_attr: str | None = None,
        scale: list[float] | None = None,
        ndim: int | None = None,
        features: FeatureDict | None = None,
    ):
        """Initialize a SolutionTracks object.

        SolutionTracks extends Tracks to ensure every node has a track_id. A
        TrackAnnotator is automatically added to manage track IDs.

        Args:
            graph (nx.DiGraph): NetworkX directed graph with nodes as detections and
                edges as links.
            segmentation (np.ndarray | None): Optional segmentation array where labels
                match node IDs. Required for computing region properties (area, etc.).
            time_attr (str | None): Graph attribute name for time. Defaults to "time"
                if None.
            pos_attr (str | tuple[str, ...] | list[str] | None): Graph attribute
                name(s) for position. Can be:
                - Single string for one attribute containing position array
                - List/tuple of strings for multi-axis (one attribute per axis)
                Defaults to "pos" if None.
            tracklet_attr (str | None): Graph attribute name for tracklet/track IDs.
                Defaults to "track_id" if None.
            scale (list[float] | None): Scaling factors for each dimension (including
                time). If None, all dimensions scaled by 1.0.
            ndim (int | None): Number of dimensions (3 for 2D+time, 4 for 3D+time).
                If None, inferred from segmentation or scale.
            features (FeatureDict | None): Pre-built FeatureDict with feature
                definitions. If provided, time_attr/pos_attr/tracklet_attr are ignored.
                Assumes that all features in the dict already exist on the graph (will
                be activated but not recomputed). If None, core computed features (pos,
                area, track_id) are auto-detected by checking if they exist on the graph.
        """
        super().__init__(
            graph,
            segmentation=segmentation,
            time_attr=time_attr,
            pos_attr=pos_attr,
            tracklet_attr=tracklet_attr,
            scale=scale,
            ndim=ndim,
            features=features,
        )

        self.track_annotator = self._get_track_annotator()

    def _initialize_track_ids(self) -> None:
        """Initialize track IDs for all nodes.

        Deprecated:
            This method is deprecated and will be removed in funtracks v2.0.
            Track IDs are now auto-computed during SolutionTracks initialization.
        """
        warnings.warn(
            "`_initialize_track_ids` is deprecated and will be removed in funtracks v2.0."
            " Track IDs are now auto-computed during SolutionTracks initialization.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.enable_features([self.features.tracklet_key])  # type: ignore

    def _get_track_annotator(self) -> TrackAnnotator:
        """Get the TrackAnnotator instance from the annotator registry.

        Returns:
            TrackAnnotator: The track annotator instance

        Raises:
            RuntimeError: If no TrackAnnotator is registered
        """
        from funtracks.annotators import TrackAnnotator

        for annotator in self.annotators:
            if isinstance(annotator, TrackAnnotator):
                return annotator
        raise RuntimeError(
            "No TrackAnnotator registered for this SolutionTracks instance"
        )

    @classmethod
    def from_tracks(cls, tracks: Tracks):
        force_recompute = False
        if (tracklet_key := tracks.features.tracklet_key) is not None:
            # Check if all nodes have track_id before trusting existing track IDs
            # Short circuit on first missing track_id
            for node in tracks.graph.nodes():
                if tracks.get_node_attr(node, tracklet_key) is None:
                    force_recompute = True
                    break
        soln_tracks = cls(
            tracks.graph,
            segmentation=tracks.segmentation,
            scale=tracks.scale,
            ndim=tracks.ndim,
            features=tracks.features,
        )
        if force_recompute:
            soln_tracks.enable_features([soln_tracks.features.tracklet_key])  # type: ignore
        return soln_tracks

    @property
    def max_track_id(self) -> int:
        return self.track_annotator.max_tracklet_id

    @property
    def track_id_to_node(self) -> dict[int, list[int]]:
        return self.track_annotator.tracklet_id_to_nodes

    @property
    def node_id_to_track_id(self) -> dict[Node, int]:
        warnings.warn(
            "node_id_to_track_id property will be removed in funtracks v2. "
            "Use `get_track_id` instead for better performance.",
            DeprecationWarning,
            stacklevel=2,
        )
        return nx.get_node_attributes(self.graph, self.features.tracklet_key)

    def get_next_track_id(self) -> int:
        """Return the next available track_id and update max_tracklet_id in TrackAnnotator

        # TODO: I don't think we need to update the max here, it will get updated if we
        actually add a node automatically.
        """
        annotator = self.track_annotator
        annotator.max_tracklet_id = annotator.max_tracklet_id + 1
        return annotator.max_tracklet_id

    def get_track_id(self, node) -> int:
        if self.features.tracklet_key is None:
            raise ValueError("Tracklet key not initialized in features")
        track_id = self.get_node_attr(node, self.features.tracklet_key, required=True)
        return track_id

    def export_tracks(
        self, outfile: Path | str, node_ids: set[int] | None = None
    ) -> None:
        """Export the tracks from this run to a csv with the following columns:
        t,[z],y,x,id,parent_id,track_id
        Cells without a parent_id will have an empty string for the parent_id.
        Whether or not to include z is inferred from self.ndim

        Args:
            outfile (Path): path to output csv file
            node_ids (set[int], optional): nodes to be included. If provided, only these
            nodes and their ancestors will be included in the output.

        .. deprecated:: 1.0
            `SolutionTracks.export_tracks()` is deprecated and will be removed in v2.0.
            Use :func:`funtracks.import_export.export_to_csv` instead.
        """
        warnings.warn(
            "SolutionTracks.export_tracks() is deprecated and will be removed in v2.0. "
            "Use funtracks.import_export.export_to_csv() instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        # Import here to avoid circular imports
        from funtracks.import_export.csv._export import export_to_csv

        export_to_csv(self, outfile, node_ids)

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

    def has_track_id_at_time(self, track_id: int, time: int) -> bool:
        """Function to check if a node with given track id exists at given time point.

        Args:
            track_id (int): The track id to search for.
            time (int): The time point to check.

        Returns:
            True if a node with given track id exists at given time point.
        """

        nodes = self.track_id_to_node.get(track_id)
        if not nodes:
            return False

        return time in self.get_times(nodes)
