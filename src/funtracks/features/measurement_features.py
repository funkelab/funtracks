from __future__ import annotations

import inspect
import sys
from typing import TYPE_CHECKING

import numpy as np

from funtracks.features.regionprops_extended import regionprops_extended

from ._base import (
    Feature,
    FeatureType,
)

if TYPE_CHECKING:
    from ..project import Project
    from funtracks.tracking_graph import TrackingGraph
    from typing import Any

import funlib.persistence as fp

class Area(Feature):
    def __init__(self, attr_name=None, ndim=3, **attrs):
        super().__init__(
            attr_name=attr_name if attr_name is not None else "area",
            display_name="Area" if ndim == 3 else "Volume",
            value_names="Area" if ndim == 3 else "Volume",
            feature_type=FeatureType.NODE,
            valid_ndim=(3, 4),
            computed=True,
            regionprops_name="area",
        )

    def update(self, graph: TrackingGraph, segmentation: fp.Array, node: int, **kwargs):
        time = graph.get_time(node)
        seg = segmentation[time] == node
        voxel_size = segmentation.voxel_size
        pos_scale = voxel_size[1:] if voxel_size is not None else None
        area = np.sum(seg)
        if pos_scale is not None:
            area *= np.prod(pos_scale)
        graph.nodes[node][self.attr_name] = area.tolist()
    
    def compute(self, graph: TrackingGraph, segmentation: fp.Array, **kwargs):
        for t in range(segmentation.shape[0]):
            props = regionprops_extended(segmentation[t], spacing=segmentation.voxel_size)
            if props:
                for prop in props: 
                    node = getattr(prop, 'label')
                    value = getattr(prop, self.region_props_name)
                    if isinstance(value, tuple):
                        value = list(value)
                    graph.nodes[node][self.attr_name] = value
    

class EllipsoidAxes(Feature):
    def __init__(self, attr_name=None, ndim=3, **attrs):
        super().__init__(
            attr_name=attr_name if attr_name is not None else "ellipse_axis_radii",
            display_name="Ellipse axis radii" if ndim == 3 else "Ellipsoid axis radii",
            value_names=("Major Axis", "Minor Axis") if ndim == 3 else ("Major Axis", "Semi-minor Axis", "Minor Axis"),
            feature_type=FeatureType.NODE,
            valid_ndim=(3, 4),
            computed=True,
            regionprops_name="axes",
        )

    def update(self, graph: TrackingGraph, segmentation: fp.Array, node: int, **kwargs):
        time = graph.get_time(node)
        seg = segmentation[time] == node
        voxel_size = segmentation.voxel_size
        pos_scale = voxel_size[1:] if voxel_size is not None else None
        props = regionprops_extended(seg, spacing=pos_scale)
        if props:
            regionprop = props[0]
            # tolist gives floats/ints in the case of single items
            value = getattr(regionprop, self.regionprops_name)
        else:
            value = None
        graph.nodes[node][self.attr_name] = value
    
    def compute(self, graph: TrackingGraph, segmentation: fp.Array, **kwargs):
        for t in range(segmentation.shape[0]):
            props = regionprops_extended(segmentation[t], spacing=segmentation.voxel_size)
            if props:
                for prop in props: 
                    node = getattr(prop, 'label')
                    value = getattr(prop, self.region_props_name)
                    if isinstance(value, tuple):
                        value = list(value)
                    graph.nodes[node][self.attr_name] = value

class Circularity(Feature):
    def __init__(self, attr_name=None, ndim=3, **attrs):
        super().__init__(
            attr_name=attr_name if attr_name is not None else "circularity",
            display_name="Circularity" if ndim == 3 else "Sphericity",
            value_names="Circularity" if ndim == 3 else "Sphericity",
            feature_type=FeatureType.NODE,
            valid_ndim=(3, 4),
            computed=True,
            regionprops_name="circularity" if ndim == 3 else "sphericity",
        )

    def update(self, graph: TrackingGraph, segmentation: fp.Array, node: int, **kwargs):
        time = graph.get_time(node)
        seg = segmentation[time] == node
        voxel_size = segmentation.voxel_size
        pos_scale = voxel_size[1:] if voxel_size is not None else None
        props = regionprops_extended(seg, spacing=pos_scale)
        if props:
            regionprop = props[0]
            # tolist gives floats/ints in the case of single items
            value = getattr(regionprop, self.regionprops_name)
        else:
            value = None
        graph.nodes[node][self.attr_name] = value
    
    def compute(self, graph: TrackingGraph, segmentation: fp.Array, **kwargs):
        for t in range(segmentation.shape[0]):
            props = regionprops_extended(segmentation[t], spacing=segmentation.voxel_size)
            if props:
                for prop in props: 
                    node = getattr(prop, 'label')
                    value = getattr(prop, self.region_props_name)
                    if isinstance(value, tuple):
                        value = list(value)
                    graph.nodes[node][self.attr_name] = value
class Perimeter(Feature):
    def __init__(self, attr_name=None, ndim=3, **attrs):
        super().__init__(
            attr_name=attr_name if attr_name is not None else "perimeter",
            display_name="Perimeter" if ndim == 3 else "Surface Area",
            value_names="Perimeter" if ndim == 3 else "Surface Area",
            feature_type=FeatureType.NODE,
            valid_ndim=(3, 4),
            computed=True,
            regionprops_name="perimeter" if ndim == 3 else "surface_area",
        )

    def update(self, graph: TrackingGraph, segmentation: fp.Array, node: int, **kwargs) -> int:
        time = graph.get_time(node)
        seg = segmentation[time] == node
        voxel_size = segmentation.voxel_size
        pos_scale = voxel_size[1:] if voxel_size is not None else None
        props = regionprops_extended(seg, spacing=pos_scale)
        if props:
            regionprop = props[0]
            # tolist gives floats/ints in the case of single items
            value = getattr(regionprop, self.regionprops_name)
        else:
            value = None
        graph.nodes[node][self.attr_name] = value
    
    def compute(self, graph: TrackingGraph, segmentation: fp.Array, **kwargs):
        for t in range(segmentation.shape[0]):
            props = regionprops_extended(segmentation[t], spacing=segmentation.voxel_size)
            if props:
                for prop in props: 
                    node = getattr(prop, 'label')
                    value = getattr(prop, self.region_props_name)
                    if isinstance(value, tuple):
                        value = list(value)
                    graph.nodes[node][self.attr_name] = value

class Intensity(Feature):
    def __init__(self, attr_name=None, n_channels: int=1, **attrs):
        super().__init__(
            attr_name=attr_name if attr_name is not None else "intensity",
            display_name="Intensity",
            value_names="Intensity" if n_channels == 1 else [f"Intensity_chan{chan}" for chan in range(n_channels)], # tuple of value names
            feature_type=FeatureType.NODE,
            valid_ndim=(3, 4),
            computed=True,
            regionprops_name="intensity_mean",
        )

    def update(self, graph: TrackingGraph, segmentation: fp.Array, raw: fp.Array, node: int, **kwargs):
        time = graph.get_time(node)
        seg = segmentation[time] == node
        intensity = []
        for chan in range(len(raw)):
            intensity_image = raw[chan][time] if raw is not None else None
            if intensity_image is not None:
                mean_intensity = np.mean(intensity_image[seg])
                intensity.append(mean_intensity)
        graph.nodes[node][self.attr_name] = intensity
    
    def compute(self, graph: TrackingGraph, segmentation: fp.Array, raw: fp.Array | None, **kwargs):
        for t in range(segmentation.shape[0]):          
            if raw is not None: 
                int_stack = []
                for chan in raw:
                    int_stack.append(chan[t])
                intensity = np.stack(int_stack, axis=-1)
            else: 
                intensity = None
            props = regionprops_extended(segmentation[t], intensity, spacing=segmentation.voxel_size)
            if props:
                for prop in props: 
                    node = getattr(prop, 'label')
                    value = getattr(prop, self.region_props_name)
                    if isinstance(value, tuple):
                        value = list(value)
                    graph.nodes[node][self.attr_name] = value


measurement_features = [
    cls for name, cls in inspect.getmembers(sys.modules[__name__], inspect.isclass)
    if issubclass(cls, Feature) and cls is not Feature and cls.__module__ == __name__
]
