from __future__ import annotations

import networkx as nx

from funtracks.data_model import SolutionTracks

from ..features.feature import Feature, FeatureType
from .graph_annotator import GraphAnnotator


class TrackletID(Feature):
    def __init__(self, tracklet_id: str | None = None):
        super().__init__(
            key="tracklet_id" if tracklet_id is None else tracklet_id,
            feature_type=FeatureType.NODE,
            value_type=int,
            valid_ndim=(3, 4),
            recompute=False,
            required=True,
        )


class LineageID(Feature):
    def __init__(self, lineage_id: str | None = None):
        super().__init__(
            key="lineage_id" if lineage_id is None else lineage_id,
            feature_type=FeatureType.NODE,
            value_type=int,
            valid_ndim=(3, 4),
            recompute=False,
            required=True,
        )


class TrackAnnotator(GraphAnnotator):
    """A graph annotator to update tracklet and lineage IDs

    The possible features include:
    - TrackletID
    - LineageID
    """

    def __init__(
        self,
        tracks: SolutionTracks,
        tracklet_key: str | None = None,
        lineage_key: str | None = None,
    ):
        if not isinstance(tracks, SolutionTracks):
            raise ValueError("Currently the TrackAnnotator only works on SolutionTracks")
        tracklet = TrackletID(tracklet_key)
        lineage = LineageID(lineage_key)
        feats = [tracklet, lineage]
        super().__init__(tracks, feats)
        self.tracklet = tracklet
        self.lineage = lineage

        self.tracklet_id_to_node: dict[int, list[int]] = {}
        self.lineage_id_to_node: dict[int, list[int]] = {}
        self.max_tracklet_id = 0
        self.max_lineage_id = 0

        if tracklet_key is not None and tracks.graph.number_of_nodes() > 0:
            self.max_tracklet_id = max(
                tracks.get_nodes_attr(tracks.nodes(), self.tracklet.key)
            )
            for node in tracks.nodes():
                tra_id: int = tracks.get_node_attr(node, self.tracklet.key)
                if tra_id not in self.tracklet_id_to_node:
                    self.tracklet_id_to_node[tra_id] = []
                self.tracklet_id_to_node[tra_id].append(node)

        if lineage_key is not None and tracks.graph.number_of_nodes() > 0:
            self.max_lineage_id = max(
                tracks.get_nodes_attr(tracks.nodes(), self.lineage.key)
            )
            for node in tracks.nodes():
                lin_id: int = tracks.get_node_attr(node, self.lineage.key)
                if lin_id not in self.lineage_id_to_node:
                    self.lineage_id_to_node[lin_id] = []
                self.lineage_id_to_node[lin_id].append(node)

    def compute(self, add_to_set=False) -> None:
        """Compute the currently included features and add them to the tracks.

        Args:
            add_to_set (bool, optional): Whether to add the Features to the Tracks
            FeatureSet. Defaults to False. Should usually be set to True on the initial
            computation, but False on subsequent re-computations.
        """
        if add_to_set:
            self.add_features_to_set()
        # TODO: move this code to litt-utils
        if self.tracklet in self.features:
            self._assign_tracklet_ids()
        if self.lineage in self.features:
            self._assign_lineage_ids()

    def _assign_lineage_ids(self) -> None:
        lineage_id = 1

        for lineage in nx.weakly_connected_components(self.tracks.graph):
            nodes = list(lineage)
            lineage_ids = [
                lineage_id,
            ] * len(nodes)
            self.tracks._set_nodes_attr(nodes, self.lineage.key, lineage_ids)

            self.lineage_id_to_node[lineage_id] = nodes
            lineage_id += 1
        self.max_lineage_id = lineage_id - 1

    def _assign_tracklet_ids(self) -> None:
        """Add a tracklet_id attribute to a graph by removing division edges,
        assigning one id to each connected component. Also sets the max_tracklet_id and
        initializes a dictionary from tracklet_id to nodes
        """
        graph_copy = self.tracks.graph.copy()
        tracklet_id = 1

        parents = [node for node, degree in self.tracks.graph.out_degree() if degree >= 2]
        intertrack_edges = []

        # Remove all intertrack edges from a copy of the original graph
        for parent in parents:
            daughters = [child for p, child in self.tracks.graph.out_edges(parent)]
            for daughter in daughters:
                graph_copy.remove_edge(parent, daughter)
                intertrack_edges.append((parent, daughter))

        for tracklet in nx.weakly_connected_components(graph_copy):
            nodes = list(tracklet)
            tracklet_ids = [
                tracklet_id,
            ] * len(nodes)
            self.tracks._set_nodes_attr(nodes, self.tracklet.key, tracklet_ids)
            self.tracklet_id_to_node[tracklet_id] = list(tracklet)
            tracklet_id += 1
        self.max_tracklet_id = tracklet_id - 1
