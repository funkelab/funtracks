from __future__ import annotations

import inspect
import sys
from typing import TYPE_CHECKING

import numpy as np

from ._base import (
    Feature,
    FeatureType,
)
from .compute_ious import compute_ious


if TYPE_CHECKING:
    from funtracks.tracking_graph import TrackingGraph
    import funlib.persistence as fp


class IoU(Feature):
    def __init__(self, attr_name: str | None = None):
        super().__init__(
            attr_name=attr_name if attr_name is not None else "iou",
            display_name="IoU",
            value_names="IoU",
            feature_type=FeatureType.EDGE,
            valid_ndim=(3, 4),
            computed=True,
        )
    
    def compute(self, graph: TrackingGraph, segmentation: fp.Array):
        for edge in graph.edges:
            source, target = edge
            source_seg = segmentation[graph.get_time(source)] == source
            target_seg = segmentation[graph.get_time(target)] == target
            ious = compute_ious(source_seg, target_seg)
            if len(ious) == 0:
                iou = 0.0
            else:
                assert len(ious) == 1
                _, _, iou = ious[0]
            graph.edges[edge][self.attr_name] = iou

    def update(self, graph: TrackingGraph, segmentation: fp.Array, edge: tuple[int, int]):
        source, target = edge
        source_seg = segmentation[graph.get_time(source)] == source
        target_seg = segmentation[graph.get_time(target)] == target
        ious = compute_ious(source_seg, target_seg)
        if len(ious) == 0:
            iou = 0.0
        else:
            assert len(ious) == 1
            _, _, iou = ious[0]
        graph.edges[edge][self.attr_name] = iou

class EdgeSelected(Feature):
    def __init__(self, attr_name: str | None = None):
        super().__init__(
            attr_name=attr_name if attr_name is not None else "selected",
            value_names="Edge Selected",
            feature_type=FeatureType.EDGE,
            valid_ndim=(3, 4),
            required=False,
            default_value=False,
        )


class EdgeSelectionPin(Feature):
    def __init__(self, attr_name: str | None = None):
        super().__init__(
            attr_name=attr_name if attr_name is not None else "pin",
            value_names="Edge Pinned",
            feature_type=FeatureType.EDGE,
            valid_ndim=(3, 4),
            required=False,
        )


class Distance(Feature):
    def __init__(self, attr_name: str | None = None):
        super().__init__(
            attr_name=attr_name if attr_name is not None else "distance",
            display_name="Distance",
            value_names="Distance",
            feature_type=FeatureType.EDGE,
            valid_ndim=(3, 4),
            computed=True,
        )
    
    def compute(self, graph: TrackingGraph, **kwargs):
        for edge in graph.edges:
            source, target = edge
            source_loc = np.array(graph.get_position(source))
            target_loc = np.array(graph.get_position(target))
            dist = np.linalg.norm(target_loc - source_loc)
            graph.edges[edge][self.attr_name] = dist

    def update(self, graph: TrackingGraph, edge: tuple[int, int]):
        source, target = edge
        source_loc = np.array(graph.get_position(source))
        target_loc = np.array(graph.get_position(target))
        dist = np.linalg.norm(target_loc - source_loc)
        graph.edges[edge][self.attr_name] = dist


class FrameSpan(Feature):
    def __init__(self, attr_name: str | None = None):
        super().__init__(
            attr_name=attr_name if attr_name is not None else "span",
            display_name="Frame span",
            value_names="Frame Span",
            feature_type=FeatureType.EDGE,
            valid_ndim=(3, 4),
            computed=True,
        )

    def compute(self, graph: TrackingGraph, **kwargs):
        for edge in graph.edges:
            source, target = edge
            source_time = graph.get_time(source)
            target_time = graph.get_time(target)
            graph.edges[edge][self.attr_name] = target_time - source_time
    
    def update(self, graph: TrackingGraph, edge: tuple[int, int]):
        source, target = edge
        source_time = graph.get_time(source)
        target_time = graph.get_time(target)
        graph.edges[edge][self.attr_name] = target_time - source_time
    
edge_features = [
    cls for name, cls in inspect.getmembers(sys.modules[__name__], inspect.isclass)
    if issubclass(cls, Feature) and cls is not Feature and cls.__module__ == __name__
]
