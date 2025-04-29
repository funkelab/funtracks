from __future__ import annotations

import abc
from collections.abc import Collection
from typing import (
    Any,
)

from .features._base import Feature
from .features.feature_set import FeatureSet


class TrackingGraph(abc.ABC):
    """Abstract class for tracking graph with features on nodes and edges.

    Directed graph with edges going forward in time.
    Each node has time and position as required features.
    Should be optimized for accessing nodes by time frame, bulk accessing features,
    bulk initializing nodes/edges, and updating nodes/edges one by one (changing features
    and updating nodes and edges).

    Nodes must have positive integer ids, but they don't need to be sequential.

    Graph functions that are exposed:
     - predecessor and successor functions
     - in degree and out degree
     - connected components

    Assumed to be a candidate graph. Solutions saved in the selected attribute on
    edges.
    """

    def __init__(self, pos_attr: str, time_attr: str, ndim: int, seg: bool = False):
        self.features = FeatureSet(
            ndim=ndim, pos_attr=pos_attr, time_attr=time_attr, seg=seg
        )

    def get_positions(self, nodes):
        return self.get_feature_values(nodes, self.features.position)

    def get_times(self, nodes):
        return self.get_feature_values(nodes, self.features.time)

    def set_time(self, node, time: int):
        self.set_feature_value(node, self.features.time, time)

    def set_position(self, node, position):
        self.set_feature_value(node, self.features.position, position)

    @abc.abstractmethod
    def get_feature_values(self, ids, feature: Feature) -> Collection[Any]:
        raise NotImplementedError()

    @abc.abstractmethod
    def set_feature_value(self, id, feature: Feature, value):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_elements_with_feature(self, feature: Feature, value):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_solution(self) -> TrackingGraph:  # a view only, I think
        raise NotImplementedError()

    @abc.abstractmethod
    def predecessors(self, node):
        raise NotImplementedError()

    @abc.abstractmethod
    def successors(self, node):
        raise NotImplementedError()

    @abc.abstractmethod
    def add_node(self, node, features: dict[Feature, Any]):
        raise NotImplementedError
