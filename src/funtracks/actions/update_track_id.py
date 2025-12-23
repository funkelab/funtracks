from __future__ import annotations

from typing import TYPE_CHECKING

from ._base import BasicAction

if TYPE_CHECKING:
    from funtracks.data_model import SolutionTracks
    from funtracks.data_model.tracks import Node


class UpdateTrackIDs(BasicAction):
    """Update tracklet ID and optionally lineage ID starting from a node.

    This action captures the old IDs before updating, enabling proper undo/redo.
    The actual ID updates are performed by TrackAnnotator via notify_annotators().

    The two updates have different scopes:
    - Tracklet ID: Updated for the linear track segment starting at start_node,
      following successors until a division is encountered or the track ends.
    - Lineage ID: Updated for the entire weakly connected component containing
      start_node, including all downstream branches after divisions.

    Args:
        tracks: The tracks to update
        start_node: The node ID of the first node to update.
        tracklet_id: The new tracklet id to assign to the track segment.
        lineage_id: The new lineage id to assign to the connected component.
            If None, lineage ID is not updated. Defaults to None.
    """

    def __init__(
        self,
        tracks: SolutionTracks,
        start_node: Node,
        tracklet_id: int,
        lineage_id: int | None = None,
    ):
        super().__init__(tracks)
        self.tracks: SolutionTracks  # Narrow type from base class
        self.start_node = start_node

        # Capture old tracklet ID
        self.old_tracklet_id = self.tracks.get_track_id(start_node)
        self.new_tracklet_id = tracklet_id

        # Capture old lineage ID (before any changes)
        self.new_lineage_id = lineage_id
        self.old_lineage_id = self.tracks.get_lineage_id(start_node)

        self._apply()

    def inverse(self) -> BasicAction:
        """Restore the previous tracklet_id and lineage_id."""
        return UpdateTrackIDs(
            self.tracks,
            self.start_node,
            self.old_tracklet_id,
            self.old_lineage_id,
        )

    def _apply(self) -> None:
        """Assign new track IDs to the track starting with start_node.

        Delegates to TrackAnnotator via notify_annotators(), which performs the
        actual track ID walking and updates.
        """
        self.tracks.notify_annotators(self)


def __getattr__(name: str):
    """Provide backwards compatibility for UpdateTrackID."""
    if name == "UpdateTrackID":
        import warnings

        warnings.warn(
            "UpdateTrackID is deprecated and will be removed in funtracks v2.0. "
            "Use UpdateTrackIDs instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return UpdateTrackIDs
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
