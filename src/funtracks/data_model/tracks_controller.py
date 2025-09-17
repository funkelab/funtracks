from __future__ import annotations

import warnings
from typing import TYPE_CHECKING
from warnings import warn

from funtracks.exceptions import InvalidActionError

from ..actions import (
    ActionGroup,
    TracksAction,
    UpdateNodeAttrs,
)
from ..actions.action_history import ActionHistory
from ..user_actions import (
    UserAddEdge,
    UserAddNode,
    UserDeleteEdge,
    UserDeleteNode,
    UserUpdateSegmentation,
)
from .graph_attributes import NodeAttr
from .solution_tracks import SolutionTracks
from .tracks import Attrs, Edge, Node, SegMask

if TYPE_CHECKING:
    from collections.abc import Iterable


class TracksController:
    """A set of high level functions to change the data model.
    All changes to the data should go through this API.
    """

    def __init__(self, tracks: SolutionTracks):
        warnings.warn(
            "TracksController deprecated in favor of directly calling UserActions and"
            "will be removed in funtracks v2. You will need to keep the action history "
            "in your application and emit the tracks refresh.",
            DeprecationWarning,
            stacklevel=2,
        )
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

    def _add_nodes(
        self,
        attributes: Attrs,
        pixels: list[SegMask] | None = None,
    ) -> tuple[TracksAction, list[Node]] | None:
        """Add nodes to the graph. Includes all attributes and the segmentation.
        Will return the actions needed to add the nodes, and the node ids generated for
        the new nodes.
        If there is a segmentation, the attributes must include:
        - time
        - node_id
        - track_id
        If there is not a segmentation, the attributes must include:
        - time
        - pos
        - track_id

        Logic of the function:
        - remove edges (when we add a node in a track between two nodes
            connected by a skip edge)
        - add the nodes
        - add edges (to connect each node to its immediate
            predecessor and successor with the same track_id, if any)

        Args:
            attributes (Attrs): dictionary containing at least time and track id,
                and either node_id (if pixels are provided) or position (if not)
            pixels (list[SegMask] | None): A list of pixels associated with the node,
                or None if there is no segmentation. These pixels will be updated
                in the tracks.segmentation, set to the new node id
        """
        times = attributes[NodeAttr.TIME.value]
        nodes: list[Node]
        if pixels is not None:
            nodes = attributes["node_id"]
        else:
            nodes = self._get_new_node_ids(len(times))
        actions: list[TracksAction] = []
        nodes_added = []
        for i in range(len(nodes)):
            try:
                actions.append(
                    UserAddNode(
                        self.tracks,
                        node=nodes[i],
                        attributes={key: val[i] for key, val in attributes.items()},
                        pixels=pixels[i] if pixels is not None else None,
                    )
                )
                nodes_added.append(nodes[i])
            except InvalidActionError as e:
                warnings.warn(f"Failed to add node: {e}", stacklevel=2)

        return ActionGroup(self.tracks, actions), nodes_added

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
        pixels = list(pixels) if pixels is not None else None
        for i, node in enumerate(nodes):
            try:
                actions.append(
                    UserDeleteNode(
                        self.tracks,
                        node,
                        pixels=pixels[i] if pixels is not None else None,
                    )
                )
            except InvalidActionError as e:
                warnings.warn(f"Failed to delete node: {e}", stacklevel=2)
        return ActionGroup(self.tracks, actions)

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
        actions: list[TracksAction] = []
        for i, node in enumerate(nodes):
            actions.append(
                UpdateNodeAttrs(
                    self.tracks, node, {key: val[i] for key, val in attributes.items()}
                )
            )
        return ActionGroup(self.tracks, actions)

    def _add_edges(self, edges: Iterable[Edge]) -> TracksAction:
        """Add edges and attributes to the graph. Also update the track ids of the
        target node tracks and potentially sibling tracks.

        Args:
            edges (Iterable[edge]): An iterable of edges, each with source and target
                node ids

        Returns:
            A TracksAction containing all edits performed in this call
        """
        actions: list[TracksAction] = []
        for edge in edges:
            actions.append(UserAddEdge(self.tracks, edge))
        return ActionGroup(self.tracks, actions)

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

    def _delete_edges(self, edges: Iterable[Edge]) -> ActionGroup:
        actions: list[TracksAction] = []
        for edge in edges:
            actions.append(UserDeleteEdge(self.tracks, edge))
        return ActionGroup(self.tracks, actions)

    def update_segmentations(
        self,
        new_value: int,
        updated_pixels: list[tuple[SegMask, int]],
        current_timepoint: int,
        current_track_id: int,
    ):
        """Handle a change in the segmentation mask, checking for node addition,
        deletion, and attribute updates.

        NOTE: we have introduced a minor breaking change to this API that finn will need
        to adapt to - it used to parse the pixel change into different action lists,
        but that is now done in the UserUpdateSegmentation action

        Args:
            new_value (int)): the label that the user drew with
            updated_pixels (list[tuple[SegMask, int]]): a list of pixels changed
                and the value that was there before the user drew
            current_timepoint (int): the current time point in the viewer, used to set
                the selected node.
            current_track_id (int): the track_id to use when adding a new node, usually
                the currently selected track id in the viewer
        """

        action = UserUpdateSegmentation(
            self.tracks, new_value, updated_pixels, current_track_id
        )
        self.action_history.add_new_action(action)
        nodes_added = action.nodes_added
        times = self.tracks.get_times(nodes_added)
        if current_timepoint in times:
            node_to_select = nodes_added[times.index(current_timepoint)]
        else:
            node_to_select = None
        self.tracks.refresh.emit(node_to_select)

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
