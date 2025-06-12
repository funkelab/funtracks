from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import geff.networkx
import zarr
from scipy.spatial import KDTree
from tqdm import tqdm

from .features.feature_set import FeatureSet
from .nx_graph import NxGraph
from .params.cand_graph_params import CandGraphParams
from .tracking_graph import TrackingGraph

if TYPE_CHECKING:
    from pathlib import Path
logger = logging.getLogger(__name__)


class CandGraph(TrackingGraph):
    def __init__(self, injected_cls, graph, features, params: CandGraphParams):
        super().__init__(injected_cls, graph, features)
        self.params = params

    def update_max_move_distance(self, value: float):
        prev = self.params.max_move_distance
        self.params.max_move_distance = value
        if value < prev:
            self.remove_cand_edges()
        elif value > prev:
            self.initialize_cand_edges()

    def update_frame_span(self, value: int):
        prev = self.params.max_frame_span
        self.params.max_frame_span = value
        if value < prev:
            self.remove_cand_edges()
        elif value > prev:
            for span in range(prev + 1, value + 1):
                print(span)
                self.add_cand_edges(span)

    def remove_cand_edges(self):
        to_remove = [
            edge
            for edge in self.edges
            if (
                self.get_distance(edge) > self.params.max_move_distance
                or self.get_feature_value(edge, self.features.frame_span)
                > self.params.max_frame_span
            )
        ]
        to_remove = [
            edge
            for edge in to_remove
            if not self.get_feature_value(edge, self.features.edge_selection_pin)
        ]
        self.remove_edges(to_remove)

    def initialize_cand_edges(self):
        for span in range(1, self.params.max_frame_span + 1):
            self.add_cand_edges(span)

    def add_cand_edges(self, frame_span):
        # TODO: when we add a new node, need to add the candidate edges and compute features
        logger.info("Extracting candidate edges")
        node_frame_dict: dict[int, list[Any]] = defaultdict(list)
        for node in self.nodes:
            time = self.get_time(node)
            node_frame_dict[time].append(node)
        # mapping from time frame to kdtree (node ids same as in node frame dict)
        kdtree_dict: dict[int, KDTree] = {}

        frames = sorted(node_frame_dict.keys())
        for start_frame in tqdm(frames, desc="Adding candidate edges"):
            end_frame = start_frame + frame_span
            if end_frame not in node_frame_dict:
                continue

            start_ids = node_frame_dict[start_frame]
            end_ids = node_frame_dict[end_frame]
            if start_frame not in kdtree_dict:
                start_kdtree = KDTree(self.get_positions(start_ids))
                kdtree_dict[start_frame] = start_kdtree
            if end_frame not in kdtree_dict:
                end_kdtree = KDTree(self.get_positions(end_ids))
                kdtree_dict[end_frame] = end_kdtree

            distance_dict = start_kdtree.sparse_distance_matrix(
                end_kdtree, max_distance=self.params.max_move_distance
            )
            for pair, distance in distance_dict.items():
                prev_idx, next_idx = pair
                edge = (start_ids[prev_idx], end_ids[next_idx])
                if not self.has_edge(edge):
                    features = {
                        self.features.frame_span: frame_span,
                        self.features.distance: distance,
                    }
                    # TODO: computed features
                    self.add_edge(edge, features)

            start_ids = end_ids
            start_kdtree = end_kdtree
            del kdtree_dict[start_frame]

    def get_solution(self) -> TrackingGraph:
        selected_nodes = self.get_elements_with_feature(self.features.node_selected, True)
        selected_edges = self.get_elements_with_feature(self.features.edge_selected, True)
        # can't add or remove edges but can change attributes in a networkx subgraph
        subgraph = self.subgraph(selected_nodes, selected_edges)
        return TrackingGraph(self._injected_cls, subgraph, self.features)

    def save(self, path: Path):
        geff.networkx.write(
            self._graph, position_attr=self.features.position.attr_name, path=path
        )
        attrs = zarr.open(path).attrs
        attrs["cand_graph_params"] = self.params.model_dump(mode="json")
        attrs["features"] = self.features.dump_json()

    @classmethod
    def load(cls, path: Path) -> CandGraph:
        nx_graph = geff.networkx.read(path)
        attrs = zarr.open(path).attrs
        cand_graph_params_dict = attrs["cand_graph_params"]
        features_json = attrs["features"]
        features = FeatureSet.from_json(features_json)
        params = CandGraphParams(**cand_graph_params_dict)
        return CandGraph(NxGraph, nx_graph, features, params)

    @classmethod
    def from_tracking_graph(cls, tracking_graph: TrackingGraph, params: CandGraphParams):
        return CandGraph(
            tracking_graph._injected_cls,
            tracking_graph._graph,
            tracking_graph.features,
            params,
        )
