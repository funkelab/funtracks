from __future__ import annotations

import abc
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable
    from typing import Any

    from .features._base import Feature


class GraphInterface(abc.ABC):
    def __init__(self, graph):
        self._graph = graph

    @property
    def nodes(self) -> Iterable[int]:
        raise NotImplementedError()

    @property
    def edges(self) -> Iterable[tuple[int, int]]:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_feature_values(self, ids, feature: Feature) -> Collection[Any]:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_feature_value(self, id, feature: Feature) -> Any:
        raise NotImplementedError()

    @abc.abstractmethod
    def set_feature_value(self, id, feature: Feature, value):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_elements_with_feature(self, feature: Feature, value):
        raise NotImplementedError()

    @abc.abstractmethod
    def subgraph(self, nodes, edges) -> GraphInterface:  # a view only, I think
        raise NotImplementedError()

    @abc.abstractmethod
    def predecessors(self, node):
        raise NotImplementedError()

    @abc.abstractmethod
    def successors(self, node):
        raise NotImplementedError()

    @abc.abstractmethod
    def add_node(self, node: int, features: dict[Feature, Any]):
        # At this point, it has to be all the static features in the FeatureSet
        raise NotImplementedError()

    @abc.abstractmethod
    def add_edge(self, edge: tuple[int, int], features: dict[Feature, Any]):
        raise NotImplementedError()

    @abc.abstractmethod
    def remove_node(self, node: int):
        raise NotImplementedError()

    @abc.abstractmethod
    def remove_edge(self, edge: tuple[int, int]):
        raise NotImplementedError()

    @abc.abstractmethod
    def remove_nodes(self, nodes: Iterable[int]):
        raise NotImplementedError()

    @abc.abstractmethod
    def remove_edges(self, edges: Iterable[tuple[int, int]]):
        raise NotImplementedError()

    @abc.abstractmethod
    def has_node(self, node: int) -> bool:
        raise NotImplementedError()

    @abc.abstractmethod
    def has_edge(self, edge: tuple[int, int]) -> bool:
        raise NotImplementedError()

    @abc.abstractmethod
    def out_degree(self, node: int) -> int:
        raise NotImplementedError()

    @abc.abstractmethod
    def in_degree(self, node: int) -> int:
        raise NotImplementedError()
