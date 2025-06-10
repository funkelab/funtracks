from __future__ import annotations

from typing import TYPE_CHECKING

from ._graph_interface import GraphInterface
from .features.feature_set import FeatureSet

if TYPE_CHECKING:
    from typing import Any

    from .features._base import Feature


class TrackingGraph(GraphInterface):
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

    """

    def __new__(cls, injected_cls, *args):
        # create a type that has name TrackingGraph and parent injected class
        GraphType = type(cls.__name__, (cls, injected_cls), {})
        # call Object.__new__ with the new class type (to avoid recursive calls to this
        # function we are writing)
        return super().__new__(GraphType)

    def __init__(self, injected_cls, graph, features: FeatureSet):
        super().__init__(graph)
        self.features = features
        track_ids = [self.get_track_id(node) for node in self.nodes]
        track_ids = [tid if tid is not None else 0 for tid in track_ids]
        self.max_track_id: int | None = max(track_ids)  # will be 0 if no track

    # Getters
    def get_positions(self, nodes):
        return self.get_feature_values(nodes, self.features.position)

    def get_position(self, node):
        return self.get_feature_value(node, self.features.position)

    def get_times(self, nodes):
        return self.get_feature_values(nodes, self.features.time)

    def get_time(self, node):
        return self.get_feature_value(node, self.features.time)

    def get_track_ids(self, nodes):
        return self.get_feature_values(nodes, self.features.track_id)

    def get_track_id(self, node):
        return self.get_feature_value(node, self.features.track_id)

    def get_distances(self, edges):
        return self.get_feature_values(edges, self.features.distance)

    def get_distance(self, edge):
        return self.get_feature_value(edge, self.features.distance)

    # setters
    def set_time(self, node, time: int):
        self.set_feature_value(node, self.features.time, time)

    def set_position(self, node, position):
        self.set_feature_value(node, self.features.position, position)

    def set_track_id(self, node: int, track_id: int):
        self.set_feature_value(node, self.features.track_id, track_id)

    def set_distance(self, edge, distance):
        self.set_feature_value(edge, self.features.distance, distance)

    def get_solution(self):
        selected_nodes = self.get_elements_with_feature(self.features.node_selected, True)
        selected_edges = self.get_elements_with_feature(self.features.edge_selected, True)
        return self.subgraph(selected_nodes, selected_edges)

    def get_track_neighbors(
        self, track_id: int, time: int
    ) -> tuple[int | None, int | None]:
        """Get the last node with the given track id before time, and the first node
        with the track id after time, if any. Does not assume that a node with
        the given track_id and time is already in tracks, but it can be.

        Args:
            track_id (int): The track id to search for
            time (int): The time point to find the immediate predecessor and successor
                for

        Returns:
            tuple[Node | None, Node | None]: The last node before time with the given
            track id, and the first node after time with the given track id,
            or Nones if there are no such nodes.
        """
        candidates = self.get_elements_with_feature(self.features.track_id, track_id)
        if len(candidates) == 0:
            return None, None
        candidates.sort(key=lambda n: self.get_time(n))

        pred = None
        succ = None
        for cand in candidates:
            if self.get_time(cand) < time:
                pred = cand
            elif self.get_time(cand) > time:
                succ = cand
                break
        return pred, succ

    def add_node(self, node, features: dict[Feature, Any]):
        # ignore computed features, but get default value for static features
        for feature in self.features.node_features:
            if not feature.computed and feature not in features and not feature.required:
                features[feature] = feature.default_value
        super().add_node(node, features)

    def get_next_track_id(self) -> int:
        """Return the next available track_id and update self.max_track_id"""
        self.max_track_id = self.max_track_id + 1
        return self.max_track_id
