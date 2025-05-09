from __future__ import annotations

import logging
from typing import Any

import networkx as nx

from ._graph_interface import GraphInterface
from .features._base import Feature, FeatureType

logger = logging.getLogger(__name__)


class NxGraph(GraphInterface):
    def __init__(self, graph: nx.DiGraph):
        super().__init__(graph)

    @property
    def nodes(self):
        return self._graph.nodes()

    @property
    def edges(self):
        return self._graph.edges()

    def get_feature_values(self, ids, feature: Feature):
        if feature.feature_type == FeatureType.NODE:
            return [self._graph.nodes[node][feature.attr_name] for node in ids]
        elif feature.feature_type == FeatureType.EDGE:
            return [self._graph.edges[edge][feature.attr_name] for edge in ids]

    def get_feature_value(self, id, feature: Feature):
        if feature.feature_type == FeatureType.NODE:
            attr_dict = self._graph.nodes[id]
        elif feature.feature_type == FeatureType.EDGE:
            attr_dict = self._graph.edges[id]
        if not feature.required:
            return attr_dict.get(feature.attr_name, feature.default_value)
        return attr_dict[feature.attr_name]

    def set_feature_value(self, id, feature: Feature, value):
        if feature.feature_type == FeatureType.NODE:
            self._graph.nodes[id][feature.attr_name] = value
        elif feature.feature_type == FeatureType.EDGE:
            self._graph.edges[id][feature.attr_name] = value

    def get_elements_with_feature(self, feature: Feature, value):
        # could store the dictionary for time specifically like before if this is
        # a bottleneck
        if feature.feature_type == FeatureType.NODE:
            return [
                node
                for node, data in self._graph.nodes(data=True)
                if data[feature.attr_name] == value
            ]
        elif feature.feature_type == FeatureType.EDGE:
            return [
                edge
                for edge, data in self._graph.edges(data=True)
                if data[feature.attr_name] == value
            ]

    def predecessors(self, node):
        return list(self._graph.predecessors(node))

    def successors(self, node):
        return list(self._graph.successors(node))

    def add_node(self, node: int, features: dict[Feature, Any]):
        # TODO self.features.validate_new_node_features(features)
        self._graph.add_node(node)
        for feature, value in features.items():
            self.set_feature_value(node, feature, value)

    def add_edge(self, edge: tuple[int, int], features: dict[Feature, Any]):
        # TODO self.features.validate_new_edge_features(features)
        self._graph.add_edge(*edge)
        for feature, value in features.items():
            self.set_feature_value(edge, feature, value)

    def remove_node(self, node: int):
        self._graph.remove_node(node)

    def remove_nodes(self, nodes):
        self._graph.remove_nodes_from(nodes)

    def remove_edge(self, edge):
        self._graph.remove_edge(*edge)

    def remove_edges(self, edges):
        self._graph.remove_edges_from(edges)

    def has_node(self, node: int) -> bool:
        return self._graph.has_node(node)

    def has_edge(self, edge: tuple[int, int]) -> bool:
        return self._graph.has_edge(*edge)

    def out_degree(self, node: int) -> int:
        return self._graph.out_degree(node)

    def in_degree(self, node: int) -> int:
        return self._graph.in_degree(node)

    def subgraph(self, nodes, edges):
        return NxGraphView(self._graph, nodes, edges)


class NxGraphView(NxGraph):
    """Filters the view of the graph to only the provided nodes and edges."""

    def __init__(self, graph, nodes, edges):
        super().__init__(graph)
        self.nodes = set(nodes)
        self.edges = set(edges)

    def predecessors(self, node):
        return [
            pred
            for pred in super().predecessors(node)
            if pred in self.nodes and (pred, node) in self.edges
        ]

    def successors(self, node):
        return [
            succ
            for succ in super().successors(node)
            if succ in self.nodes and (node, succ) in self.edges
        ]

    def has_node(self, node):
        return node in self.nodes

    def has_edge(self, edge):
        return edge in self.edges

    def in_degree(self, node):
        return len(self.predecessors(node))

    def out_degree(self, node):
        return len(self.successors(node))

    def add_edge(self, edge):
        raise NotImplementedError()

    def add_node(self, node):
        raise NotImplementedError()

    def get_elements_with_feature(self, feature, value):
        all_elements = super().get_elements_with_feature(feature, value)
        if feature.feature_type == FeatureType.NODE:
            return [elt for elt in all_elements if elt in self.nodes]
        if feature.feature_type == FeatureType.EDGE:
            return [elt for elt in all_elements if elt in self.edges]

    def get_feature_value(self, id, feature):
        elements = self.nodes if feature.feature_type == FeatureType.NODE else self.edges
        if id not in elements:
            raise KeyError(f"{id} not in subgraph")
        return super().get_feature_value(id, feature)

    def get_feature_values(self, ids, feature):
        elements = self.nodes if feature.feature_type == FeatureType.NODE else self.edges
        if ids not in elements:
            raise KeyError(f"{ids} not in subgraph")
        return super().get_feature_values(ids, feature)
