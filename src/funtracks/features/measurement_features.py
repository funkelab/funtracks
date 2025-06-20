from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from skimage import measure

import sys
import inspect

from ._base import (
    Feature,
    FeatureType,
)

if TYPE_CHECKING:
    from ..project import Project

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

class Intensity(Feature):
    def __init__(self, ndim=3, attr_name=None):
        super().__init__(
            attr_name=attr_name if attr_name is not None else "intensity",
            value_names="Intensity",
            feature_type=FeatureType.NODE,
            valid_ndim=(3, 4),
            computed=True,
            regionprops_name="intensity_mean",
        )

    def update(self, project: Project, node: int) -> int:
        time = project.cand_graph.get_time(node)
        seg = project.segmentation[time] == node
        intensity_image = project.raw[time] if project.raw is not None else None
        if intensity_image is not None:
            mean_intensity = np.mean(intensity_image[seg])
            return mean_intensity

featureset = [
    cls for name, cls in inspect.getmembers(sys.modules[__name__], inspect.isclass)
    if issubclass(cls, Feature) and cls is not Feature and cls.__module__ == __name__
]