from __future__ import annotations

from typing import TYPE_CHECKING
from warnings import warn

from ..actions import (
    ActionGroup,
    AddEdges,
    DeleteEdges,
    DeleteNodes,
    TracksAction,
    UpdateNodeAttrs,
    UpdateNodeSegs,
    UpdateTrackID,
)
from ..actions.action_history import ActionHistory
from ..actions.solution_tracks import SolutionTracks
from ..actions.tracks import Attrs, Edge, Node, SegMask
from .graph_attributes import NodeAttr

if TYPE_CHECKING:
    from collections.abc import Iterable


class TracksController:
    """A set of high level functions to change the data model.
    All changes to the data should go through this API.
    """

    def __init__(self, tracks: SolutionTracks):
        self.tracks = tracks
        self.action_history = ActionHistory()
        self.node_id_counter = 1

    def add_nodes(
        self,
        attributes: Attrs,
        pixels: list[SegMask] | None = None,
    ) -> None:
        """Calls the _add_nodes function to add nodes. Calls the refresh signal when
        finished.

        Args:
            attributes (Attrs): dictionary containing at least time and position
                attributes
            pixels (list[SegMask] | None, optional): The pixels associated with each
                node, if a segmentation is present. Defaults to None.
        """
        result = self._add_nodes(attributes, pixels)
        if result is not None:
            action, nodes = result
            self.action_history.add_new_action(action)
            self.tracks.refresh.emit(nodes[0] if nodes else None)

    def delete_nodes(self, nodes: Iterable[Node]) -> None:
        """Calls the _delete_nodes function and then emits the refresh signal

        Args:
            nodes (Iterable[Node]): array of node_ids to be deleted
        """

        action = self._delete_nodes(nodes)
        self.action_history.add_new_action(action)
        self.tracks.refresh.emit()

    def _delete_nodes(
        self, nodes: Iterable[Node], pixels: Iterable[SegMask] | None = None
    ) -> TracksAction:
        """Delete the nodes provided by the array from the graph but maintain successor
        track_ids. Reconnect to the nearest predecessor and/or nearest successor
        on the same track, if any.

        Function logic:
        - delete all edges incident to the nodes
        - delete the nodes
        - add edges to preds and succs of nodes if they have the same track id
        - update track ids if we removed a division by deleting the dge

        Args:
            nodes (Iterable[Node]): array of node_ids to be deleted
            pixels (Iterable[SegMask] | None): pixels of the ndoes to be deleted, if
                known already. Will be computed if not provided.
        """
        actions: list[TracksAction] = []

        # find all the edges that should be deleted (no duplicates) and put them in a
        # single action. also keep track of which deletions removed a division, and save
        # the sibling nodes so we can update the track ids
        edges_to_delete = set()
        new_track_ids = []
        for node in nodes:
            for pred in self.tracks.graph.predecessors(node):
                edges_to_delete.add((pred, node))
                # determine if we need to relabel any tracks
                siblings = list(self.tracks.graph.successors(pred))
                if len(siblings) == 2:
                    # need to relabel the track id of the sibling to match the pred
                    # because you are implicitly deleting a division
                    siblings.remove(node)
                    sib = siblings[0]
                    # check if the sibling is also deleted, because then relabeling is
                    # not needed
                    if sib not in nodes:
                        new_track_id = self.tracks.get_track_id(pred)
                        new_track_ids.append((sib, new_track_id))
            for succ in self.tracks.graph.successors(node):
                edges_to_delete.add((node, succ))
        if len(edges_to_delete) > 0:
            actions.append(DeleteEdges(self.tracks, list(edges_to_delete)))

        if len(new_track_ids) > 0:
            for node, track_id in new_track_ids:
                actions.append(UpdateTrackID(self.tracks, node, track_id))

        track_ids = [self.tracks.get_track_id(node) for node in nodes]
        times = self.tracks.get_times(nodes)
        # remove nodes
        actions.append(DeleteNodes(self.tracks, nodes, pixels=pixels))

        # find all the skip edges to be made (no duplicates or intermediates to nodes
        # that are deleted) and put them in a single action
        skip_edges = set()
        for track_id, time in zip(track_ids, times, strict=False):
            pred, succ = self._get_pred_and_succ(track_id, time)
            if pred is not None and succ is not None:
                skip_edges.add((pred, succ))
        if len(skip_edges) > 0:
            actions.append(AddEdges(self.tracks, list(skip_edges)))

        return ActionGroup(self.tracks, actions=actions)

    def _update_node_segs(
        self,
        nodes: Iterable[Node],
        pixels: Iterable[SegMask],
        added=False,
    ) -> TracksAction:
        """Update the segmentation and segmentation-managed attributes for
        a set of nodes.

        Args:
            nodes (Iterable[Node]): The nodes to update
            pixels (list[SegMask]): The pixels for each node that were edited
            added (bool, optional): If the pixels were added to the nodes (True)
                or deleted (False). Defaults to False. Cannot mix adding and removing
                pixels in one call.

        Returns:
            TracksAction: _description_
        """
        return UpdateNodeSegs(self.tracks, nodes, pixels, added=added)

    def add_edges(self, edges: Iterable[Edge]) -> None:
        """Add edges to the graph. Also update the track ids and
        corresponding segmentations if applicable

        Args:
            edges (Iterable[Edge]): An iterable of edges, each with source and target
                node ids
        """
        make_valid_actions = []
        for edge in edges:
            is_valid, valid_action = self.is_valid(edge)
            if not is_valid:
                # warning was printed with details in is_valid call
                return
            if valid_action is not None:
                make_valid_actions.append(valid_action)
        main_action = self._add_edges(edges)
        action: TracksAction
        if len(make_valid_actions) > 0:
            make_valid_actions.append(main_action)
            action = ActionGroup(self.tracks, make_valid_actions)
        else:
            action = main_action
        self.action_history.add_new_action(action)
        self.tracks.refresh.emit()

    def update_node_attrs(self, nodes: Iterable[Node], attributes: Attrs):
        """Update the user provided node attributes (not the managed attributes).
        Also adds the action to the history and emits the refresh signal.

        Args:
            nodes (Iterable[Node]): The nodes to update the attributes for
            attributes (Attrs): A mapping from user-provided attributes to values for
                each node.
        """
        action = self._update_node_attrs(nodes, attributes)
        self.action_history.add_new_action(action)
        self.tracks.refresh.emit()

    def _update_node_attrs(
        self, nodes: Iterable[Node], attributes: Attrs
    ) -> TracksAction:
        """Update the user provided node attributes (not the managed attributes).

        Args:
            nodes (Iterable[Node]): The nodes to update the attributes for
            attributes (Attrs): A mapping from user-provided attributes to values for
                each node.

        Returns: A TracksAction object that performed the update
        """
        return UpdateNodeAttrs(self.tracks, nodes, attributes)

    def is_valid(self, edge: Edge) -> tuple[bool, TracksAction | None]:
        """Check if this edge is valid.
        Criteria:
        - not horizontal
        - not existing yet
        - no merges
        - no triple divisions
        - new edge should be the shortest possible connection between two nodes, given
            their track_ids (no skipping/bypassing any nodes of the same track_id).
            Check if there are any nodes of the same source or target track_id between
            source and target

        Args:
            edge (Edge): edge to be validated

        Returns:
            True if the edge is valid, false if invalid"""

        # make sure that the node2 is downstream of node1
        time1 = self.tracks.get_time(edge[0])
        time2 = self.tracks.get_time(edge[1])

        if time1 > time2:
            edge = (edge[1], edge[0])
            time1, time2 = time2, time1
        action = None
        # do all checks
        # reject if edge already exists
        if self.tracks.graph.has_edge(edge[0], edge[1]):
            warn("Edge is rejected because it exists already.", stacklevel=2)
            return False, action

        # reject if edge is horizontal
        elif self.tracks.get_time(edge[0]) == self.tracks.get_time(edge[1]):
            warn("Edge is rejected because it is horizontal.", stacklevel=2)
            return False, action

        # reject if target node already has an incoming edge
        elif self.tracks.graph.in_degree(edge[1]) > 0:
            warn(
                "Edge is rejected because merges are currently not allowed.", stacklevel=2
            )
            return False, action

        elif self.tracks.graph.out_degree(edge[0]) > 1:
            warn(
                "Edge is rejected because triple divisions are currently not allowed.",
                stacklevel=2,
            )
            return False, action

        elif time2 - time1 > 1:
            track_id2 = self.tracks.graph.nodes[edge[1]][NodeAttr.TRACK_ID.value]
            # check whether there are already any nodes with the same track id between
            # source and target (shortest path between equal track_ids rule)
            for t in range(time1 + 1, time2):
                nodes = [
                    n
                    for n, attr in self.tracks.graph.nodes(data=True)
                    if attr.get(self.tracks.time_attr) == t
                    and attr.get(NodeAttr.TRACK_ID.value) == track_id2
                ]
                if len(nodes) > 0:
                    warn("Please connect to the closest node", stacklevel=2)
                    return False, action

        # all checks passed!
        return True, action

    def delete_edges(self, edges: Iterable[Edge]):
        """Delete edges from the graph.

        Args:
            edges (Iterable[Edge]): The Nx2 array of edges to be deleted
        """

        for edge in edges:
            # First check if the to be deleted edges exist
            if not self.tracks.graph.has_edge(edge[0], edge[1]):
                warn("Cannot delete non-existing edge!", stacklevel=2)
                return
        action = self._delete_edges(edges)
        self.action_history.add_new_action(action)
        self.tracks.refresh.emit()

    def undo(self) -> bool:
        """Obtain the action to undo from the history, and invert.
        Returns:
            bool: True if the action was undone, False if there were no more actions
        """
        if self.action_history.undo():
            self.tracks.refresh.emit()
            return True
        else:
            return False

    def redo(self) -> bool:
        """Obtain the action to redo from the history
        Returns:
            bool: True if the action was re-done, False if there were no more actions
        """
        if self.action_history.redo():
            self.tracks.refresh.emit()
            return True
        else:
            return False

    def _get_new_node_ids(self, n: int) -> list[Node]:
        """Get a list of new node ids for creating new nodes.
        They will be unique from all existing nodes, but have no other guarantees.

        Args:
            n (int): The number of new node ids to return

        Returns:
            list[Node]: A list of new node ids.
        """
        ids = [self.node_id_counter + i for i in range(n)]
        self.node_id_counter += n
        for idx, _id in enumerate(ids):
            while self.tracks.graph.has_node(_id):
                _id = self.node_id_counter
                self.node_id_counter += 1
            ids[idx] = _id
        return ids
