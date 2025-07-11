from __future__ import annotations

from typing import TYPE_CHECKING

from skimage import measure

from ._base import (
    Feature,
    FeatureType,
)

if TYPE_CHECKING:
    from ..project import Project
    from funtracks.cand_graph import TrackingGraph
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
        assert project.segmentation is not None
        seg = project.segmentation[time] == node
        voxel_size = project.segmentation.voxel_size
        pos_scale = voxel_size[1:] if voxel_size is not None else None
        pos = measure.centroid(seg, spacing=pos_scale).tolist()
        return pos

class TrackID(Feature):
    def __init__(self, attr_name=None):
        super().__init__(
            attr_name=attr_name if attr_name is not None else "track_id",
            value_names="Track ID",
            feature_type=FeatureType.NODE,
            valid_ndim=(3, 4),
            required=False,
        )
    
    def compute(self, graph: TrackingGraph):

        # set track ids to None first, then update the values for the nodes in solution
        for node in graph.nodes:
            graph.nodes[node][self.attr_name] = None
        
        parent = {}

        # Union-Find functions
        def find(u):
            while parent.get(u, u) != u:
                parent[u] = parent.get(parent[u], parent[u])  # Path compression
                u = parent[u]
            return u

        def union(u, v):
            pu, pv = find(u), find(v)
            if pu != pv:
                parent[pu] = pv

        # Union nodes if both are part of a linear segment
        for u, v in graph.solution.edges():
            if graph.out_degree(u) == 1 and graph.in_degree(v) == 1:
                union(u, v)

        # Group nodes by root parent
        groups = {}
        for node in graph.solution.nodes:
            root = find(node)
            groups.setdefault(root, []).append(node)

        # Assign track_ids
        for track_id, nodes in enumerate(groups.values(), start=1):
            for node in nodes:
                graph.nodes[node][self.attr_name] = track_id


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
