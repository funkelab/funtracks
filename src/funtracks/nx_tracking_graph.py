from __future__ import annotations

from typing import Any

import networkx as nx

from .features._base import Feature, FeatureType
from .tracking_graph import TrackingGraph


class NxTrackingGraph(TrackingGraph):
    def __init__(self, graph: nx.DiGraph, pos_attr: str, time_attr: str, axes: list[str]):
        super().__init__(pos_attr=pos_attr, time_attr=time_attr, axes=axes)
        self._graph = graph

    def get_feature_values(self, ids, feature: Feature):
        if feature.feature_type == FeatureType.NODE:
            return [self._graph.nodes[node][feature.attr_name] for node in ids]
        elif feature.feature_type == FeatureType.EDGE:
            return [self._graph.edges[edge][feature.attr_name] for edge in ids]

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

    def get_solution(self) -> NxTrackingGraph:
        selected_nodes = self.get_elements_with_feature(self.node_selected_feature, True)
        # can't add or remove edges but can change attributes in a networkx subgraph
        subgraph = nx.subgraph(self._graph, selected_nodes)
        return NxTrackingGraph(
            subgraph,
            pos_attr=self.features.position.attr_name,
            time_attr=self.features.time.attr_name,
            axes=self.features.position.value_names,
        )

    def predecessors(self, node):
        return list(self._graph.predecessors(node))

    def successors(self, node):
        return list(self._graph.successors(node))

    def add_node(self, node, features: dict[Feature, Any]):
        self._graph.add_node(node)
        for feature, value in features.items():
            self.set_feature_value(node, feature, value)
