"""Backward-compatible ``SolutionTracks`` shim.

In funtracks v2, ``SolutionTracks`` was a subclass of ``Tracks`` that added
track-id management (``TrackAnnotator``).  Since the persistent-graph rework,
every ``Tracks`` instance has track IDs, so ``SolutionTracks`` is no longer
needed.  This module keeps the class importable so that downstream code
(e.g. ``motile_tracker.MotileRun(SolutionTracks)``) continues to work.

New code should use ``Tracks`` directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from warnings import warn

import tracksdata as td

from .tracks import Tracks

if TYPE_CHECKING:
    from tracksdata.array import GraphArrayView

    from funtracks.features import FeatureDict


class SolutionTracks(Tracks):
    """Backward-compatible alias for :class:`Tracks`.

    Accepts a :class:`~tracksdata.graph.GraphView` (the v2 constructor
    signature) or a :class:`~tracksdata.graph.BaseGraph` (the new signature).
    When a ``GraphView`` is passed the root ``BaseGraph`` is extracted
    automatically so that ``Tracks.__init__`` can build its own solution view.

    .. deprecated::
        Use :class:`Tracks` directly for new code.
    """

    def __init__(
        self,
        graph: td.graph.GraphView | td.graph.BaseGraph,
        time_attr: str | None = None,
        pos_attr: str | tuple[str] | list[str] | None = None,
        tracklet_attr: str | None = None,
        lineage_attr: str | None = None,
        scale: list[float] | None = None,
        ndim: int | None = None,
        features: FeatureDict | None = None,
        _segmentation: GraphArrayView | None = None,
    ):
        warn(
            "SolutionTracks is deprecated — use Tracks directly. "
            "All Tracks instances now have track IDs.",
            DeprecationWarning,
            stacklevel=2,
        )
        # v2 callers pass a GraphView; Tracks now requires the root BaseGraph.
        if isinstance(graph, td.graph.GraphView):
            graph = graph._root  # type: ignore[attr-defined]
        super().__init__(
            graph,
            time_attr=time_attr,
            pos_attr=pos_attr,
            tracklet_attr=tracklet_attr,
            lineage_attr=lineage_attr,
            scale=scale,
            ndim=ndim,
            features=features,
            _segmentation=_segmentation,
        )

    @classmethod
    def from_tracks(cls, tracks: Tracks) -> SolutionTracks:
        """Create a ``SolutionTracks`` from an existing ``Tracks``.

        .. deprecated::
            No longer needed — every ``Tracks`` already has track IDs.
        """
        warn(
            "SolutionTracks.from_tracks() is deprecated — Tracks already has track IDs.",
            DeprecationWarning,
            stacklevel=2,
        )
        return cls(
            tracks.graph_full,
            scale=tracks.scale,
            ndim=tracks.ndim,
            features=tracks.features,
        )
