from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import networkx as nx

from funtracks.data_model import SolutionTracks
from funtracks.features import Feature

from ._graph_annotator import GraphAnnotator

if TYPE_CHECKING:
    from collections.abc import Iterable


def TrackletID() -> Feature:
    """A feature representing tracklet ID for nodes.

    Returns:
        Feature: A feature dict representing tracklet ID
    """
    return {
        "feature_type": "node",
        "value_type": "int",
        "num_values": 1,
        "display_name": "Tracklet ID",
        "recompute": False,
        "required": True,
        "default_value": None,
    }


def LineageID() -> Feature:
    """A feature representing lineage ID for nodes.

    Returns:
        Feature: A feature dict representing lineage ID
    """
    return {
        "feature_type": "node",
        "value_type": "int",
        "num_values": 1,
        "display_name": "Lineage ID",
        "recompute": False,
        "required": True,
        "default_value": None,
    }


class TrackAnnotator(GraphAnnotator):
    """A graph annotator to compute tracklet and lineage IDs for SolutionTracks only.

    Currently, updating the tracklet and lineage IDs is left to Actions.

    Attributes:
        tracklet_id_to_nodes (dict[int, list[int]]): A mapping from tracklet ids to
            nodes in that tracklet
        lineage_id_to_nodes (dict[int, list[int]]): A mapping from lineage ids to nodes
            in that lineage
        max_tracklet_id (int): the maximum tracklet id used in the tracks
        max_lineage_id (int): the maximum lineage id used in the tracks

    Args:
        tracks (SolutionTracks): The tracks to be annotated. Must be a solution.
        tracklet_key (str | None, optional): A key that already holds the tracklet ids
            on the graph. If provided, must be there for every node and already hold
            valid tracklet ids. Defaults to None.
        lineage_key (str | None, optional): A key that already holds the lineage ids
            on the graph. If provided, must be there for every node and already hold
            valid lineage ids. Defaults to None.


    Raises:
        ValueError: if the provided Tracks are not SolutionTracks (not a binary lineage
            tree)
    """

    def __init__(
        self,
        tracks: SolutionTracks,
        tracklet_key: str | None = None,
        lineage_key: str | None = None,
    ):
        if not isinstance(tracks, SolutionTracks):
            raise ValueError("Currently the TrackAnnotator only works on SolutionTracks")

        self.tracklet_key = tracklet_key if tracklet_key is not None else "tracklet_id"
        self.lineage_key = lineage_key if lineage_key is not None else "lineage_id"

        feats = {
            self.tracklet_key: TrackletID(),
            self.lineage_key: LineageID(),
        }
        super().__init__(tracks, feats)

        self.tracklet_id_to_nodes: dict[int, list[int]] = {}
        self.lineage_id_to_nodes: dict[int, list[int]] = {}
        self.max_tracklet_id = 0
        self.max_lineage_id = 0

        if tracklet_key is not None and tracks.graph.number_of_nodes() > 0:
            max_id, id_to_nodes = self._get_max_id_and_map(self.tracklet_key)
            self.max_tracklet_id = max_id
            self.tracklet_id_to_nodes = id_to_nodes

        if lineage_key is not None and tracks.graph.number_of_nodes() > 0:
            max_id, id_to_nodes = self._get_max_id_and_map(self.lineage_key)
            self.max_lineage_id = max_id
            self.lineage_id_to_nodes = id_to_nodes

    def _get_max_id_and_map(self, key: str) -> tuple[int, dict[int, list[int]]]:
        """Get the maximum ID value and a mapping from ids to nodes with that id.

        Args:
            key (str): The key holding the id attribute. Can be for tracklets or lineages.

        Returns:
            tuple[int, dict[int, list[int]]]: The maximum id value, and a mapping from
                ids to a list of nodes with that id.
        """
        id_to_nodes = defaultdict(list)
        for node in self.tracks.nodes():
            _id: int = self.tracks.get_node_attr(node, key)
            id_to_nodes[_id].append(node)
        max_id = max(id_to_nodes.keys()) if len(id_to_nodes) > 0 else 0
        return max_id, dict(id_to_nodes)

    def compute(self) -> None:
        """Compute the currently included features and add them to the tracks."""
        # TODO: move this code to litt-utils
        if self.tracklet_key in self.features:
            self._assign_tracklet_ids()
        if self.lineage_key in self.features:
            self._assign_lineage_ids()

    def _assign_ids(
        self, components: Iterable[Iterable[int]], key: str
    ) -> tuple[int, dict[int, list[int]]]:
        """Assign a unique id to each component in the list and return the max used.

        IDs start at 1 and are sequential from there.

        Args:
            components (Iterable[Iterable[int]]): A list of sets of nodes to be labeled
                uniquely. Can be tracklets or lineages.
            key (str): the key to save the unique id at on the graph nodes

        Returns:
            tuple[int, dict[int, list[int]]]: the maximum id used, and a mapping from ids
                to lists of nodes with that id
        """
        id_to_node = {}
        _id = 1
        for component in components:
            nodes = list(component)
            ids = [
                _id,
            ] * len(nodes)
            self.tracks._set_nodes_attr(nodes, key, ids)

            id_to_node[_id] = nodes
            _id += 1
        return _id - 1, id_to_node

    def _assign_lineage_ids(self) -> None:
        """Add a lineage id attribute to each node of the solution tracks.

        Each connected component will get a unique id, and the relevant class
        attributes will be updated.
        """
        lineages = nx.weakly_connected_components(self.tracks.graph)
        max_id, ids_to_nodes = self._assign_ids(lineages, self.lineage_key)
        self.max_lineage_id = max_id
        self.lineage_id_to_nodes = ids_to_nodes

    def _assign_tracklet_ids(self) -> None:
        """Add a tracklet id attribute to each node of the solution tracks.

        After removing division edges, each connected component will get a unique ID,
        and the relevant class attributes will be updated.
        """
        graph_copy = self.tracks.graph.copy()
        parents = [node for node, degree in self.tracks.graph.out_degree() if degree >= 2]

        # Remove all intertrack edges from a copy of the original graph
        for parent in parents:
            daughters = self.tracks.successors(parent)
            for daughter in daughters:
                graph_copy.remove_edge(parent, daughter)

        tracklets = nx.weakly_connected_components(graph_copy)
        max_id, ids_to_nodes = self._assign_ids(tracklets, self.tracklet_key)
        self.max_tracklet_id = max_id
        self.tracklet_id_to_nodes = ids_to_nodes
