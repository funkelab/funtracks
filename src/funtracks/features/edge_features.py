from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._base import (
    Feature,
    FeatureType,
)
from .compute_ious import compute_ious

if TYPE_CHECKING:
    from ..project import Project


class IoU(Feature):
    def __init__(self, attr_name: str | None = None):
        super().__init__(
            attr_name=attr_name if attr_name is not None else "iou",
            value_names="IoU",
            feature_type=FeatureType.EDGE,
            valid_ndim=(3, 4),
            computed=True,
        )

    def update(self, project: Project, edge: tuple[int, int]) -> float:
        source, target = edge
        source_seg = project.segmentation[project.graph.get_time(source)] == source
        target_seg = project.segmentation[project.graph.get_time(target)] == target
        ious = compute_ious(source_seg, target_seg)
        if len(ious) == 0:
            iou = 0.0
        else:
            assert len(ious) == 1
            _, _, iou = ious[0]
        return iou


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
            value_names="Distance",
            feature_type=FeatureType.EDGE,
            valid_ndim=(3, 4),
            computed=True,
        )

    def update(self, project: Project, edge: tuple[int, int]) -> float:
        source, target = edge
        source_loc = np.array(project.graph.get_position(source))
        target_loc = np.array(project.graph.get_position(target))
        dist = np.linalg.norm(target_loc - source_loc)
        return dist


class FrameSpan(Feature):
    def __init__(self, attr_name: str | None = None):
        super().__init__(
            attr_name=attr_name if attr_name is not None else "span",
            value_names="Frame Span",
            feature_type=FeatureType.EDGE,
            valid_ndim=(3, 4),
            computed=True,
        )

    def update(self, project: Project, edge: tuple[int, int]) -> int:
        source, target = edge
        source_time = project.graph.get_time(source)
        target_time = project.graph.get_time(target)
        return target_time - source_time
