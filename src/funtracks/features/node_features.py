from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from skimage import measure

from ._base import (
    Feature,
    FeatureType,
)

if TYPE_CHECKING:
    from ..project import Project


class Time(Feature):
    def __init__(self, attr_name=None):
        super().__init__(
            attr_name=attr_name if attr_name is not None else "time",
            value_names="Time",
            feature_type=FeatureType.NODE,
            valid_ndim=(3, 4),
            required=True,
        )


class Position(Feature):
    def __init__(self, axes: tuple[str, ...], attr_name=None):
        super().__init__(
            attr_name=attr_name if attr_name is not None else "pos",
            value_names=axes,
            feature_type=FeatureType.NODE,
            valid_ndim=(3, 4),
        )


class ComputedPosition(Feature):
    def __init__(self, axes: tuple[str, ...], attr_name=None):
        super().__init__(
            attr_name=attr_name if attr_name is not None else "pos",
            value_names=axes,
            feature_type=FeatureType.NODE,
            valid_ndim=(3, 4),
            computed=True,
            regionprops_name="centroid",
        )

    def update(self, project: Project, node: int) -> list[float]:
        # Note: assumes the time is already on the graph
        time = project.cand_graph.get_time(node)
        seg = project.segmentation[time] == node
        voxel_size = project.segmentation.voxel_size
        pos_scale = voxel_size[1:] if voxel_size is not None else None
        pos = measure.centroid(seg, spacing=pos_scale).tolist()
        return pos


class Area(Feature):
    def __init__(self, ndim=3, attr_name=None):
        super().__init__(
            attr_name=attr_name if attr_name is not None else "area",
            value_names="Area" if ndim == 3 else "Volume",
            feature_type=FeatureType.NODE,
            valid_ndim=(3, 4),
            computed=True,
            regionprops_name="area",
        )

    def update(self, project: Project, node: int) -> int:
        time = project.cand_graph.get_time(node)
        seg = project.segmentation[time] == node
        voxel_size = project.segmentation.voxel_size
        pos_scale = voxel_size[1:] if voxel_size is not None else None
        area = np.sum(seg)
        if pos_scale is not None:
            area *= np.prod(pos_scale)
        return area.tolist()


class TrackID(Feature):
    def __init__(self, attr_name=None):
        super().__init__(
            attr_name=attr_name if attr_name is not None else "track_id",
            value_names="Track ID",
            feature_type=FeatureType.NODE,
            valid_ndim=(3, 4),
            required=False,
        )


class NodeSelected(Feature):
    def __init__(self, attr_name=None):
        super().__init__(
            attr_name=attr_name if attr_name is not None else "selected",
            value_names="Node Selected",
            feature_type=FeatureType.NODE,
            valid_ndim=(3, 4),
            required=False,
            default_value=False,
        )


class NodeSelectionPin(Feature):
    def __init__(self, attr_name=None):
        super().__init__(
            attr_name=attr_name if attr_name is not None else "pin",
            value_names="Node Pinned",
            feature_type=FeatureType.NODE,
            valid_ndim=(3, 4),
            required=False,
        )
