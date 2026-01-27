from __future__ import annotations

from typing import TYPE_CHECKING

import tracksdata as td

from funtracks.features import FeatureDict

from .tracks import Tracks

if TYPE_CHECKING:
    from funtracks.annotators import TrackAnnotator

    from .tracks import Node


class SolutionTracks(Tracks):
    """Difference from Tracks: every node must have a track_id"""

    def __init__(
        self,
        graph: td.graph.GraphView,
        time_attr: str | None = None,
        pos_attr: str | tuple[str] | list[str] | None = None,
        tracklet_attr: str | None = None,
        scale: list[float] | None = None,
        ndim: int | None = None,
        features: FeatureDict | None = None,
        _segmentation: td.array.GraphArrayView | None = None,
    ):
        """Initialize a SolutionTracks object.

        SolutionTracks extends Tracks to ensure every node has a track_id. A
        TrackAnnotator is automatically added to manage track IDs.

        Args:
            graph (td.graph.GraphView): NetworkX directed graph with nodes as detections
                and edges as links.
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
            _segmentation (GraphArrayView | None): Internal parameter for reusing an
                existing GraphArrayView instance. Not intended for public use.
        """
        super().__init__(
            graph,
            time_attr=time_attr,
            pos_attr=pos_attr,
            tracklet_attr=tracklet_attr,
            scale=scale,
            ndim=ndim,
            features=features,
            _segmentation=_segmentation,
        )

        self.track_annotator = self._get_track_annotator()

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
        # Check if all nodes have track_id before trusting existing track IDs
        if (
            tracks.features.tracklet_key is not None
            and (
                tracks.graph.node_attrs(attr_keys=tracks.features.tracklet_key)[
                    tracks.features.tracklet_key
                ]
                == -1
            ).any()
            # Attributes are no longer None, so 0 now means non-computed
        ):
            force_recompute = True

        soln_tracks = cls(
            tracks.graph,
            scale=tracks.scale,
            ndim=tracks.ndim,
            features=tracks.features,
            _segmentation=tracks.segmentation,
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
        return (
            int(pred) if pred is not None else None,
            int(succ) if succ is not None else None,
        )

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
