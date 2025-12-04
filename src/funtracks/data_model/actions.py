"""This module contains all the low level actions used to control a Tracks object.
Low level actions should control these aspects of Tracks:
    - adding/removing nodes and edges to/from the segmentation and graph
    - Updating the segmentation and graph attributes that are controlled by the
      segmentation. Currently, position and area for nodes, and IOU for edges.
    - Keeping track of information needed to undo the given action. For removing a node,
    this means keeping track of the incident edges that were removed, along with their
    attributes.

The low level actions do not contain application logic, such as manipulating track ids,
or validation of "allowed" actions.
The actions should work on candidate graphs as well as solution graphs.
Action groups can be constructed to represent application-level actions constructed
from many low-level actions.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
import tracksdata as td
from typing_extensions import override

from .graph_attributes import NodeAttr
from .solution_tracks import SolutionTracks
from .tracks import Attrs, Edge, Node, SegMask, Tracks
from .tracksdata_utils import (
    compute_node_attrs_from_masks,
    compute_node_attrs_from_pixels,
    td_get_predecessors,
    td_get_successors,
    td_graph_edge_list,
)

if TYPE_CHECKING:
    from collections.abc import Iterable


class TracksAction:
    def __init__(self, tracks: Tracks):
        """An modular change that can be applied to the given Tracks. The tracks must
        be passed in at construction time so that metadata needed to invert the action
        can be extracted.
        The change should be applied in the init function.

        Args:
            tracks (Tracks): The tracks that this action will edit
        """
        self.tracks = tracks

    def inverse(self) -> TracksAction:
        """Get the inverse of this action. Calling this function does undo the action,
        since the change is applied in the action constructor.

        Raises:
            NotImplementedError: if the inverse is not implemented in the subclass

        Returns:
            TracksAction: An action that un-does this action, bringing the tracks
                back to the exact state it had before applying this action.
        """
        raise NotImplementedError("Inverse not implemented")


class ActionGroup(TracksAction):
    def __init__(
        self,
        tracks: Tracks,
        actions: list[TracksAction],
    ):
        """A group of actions that is also an action, used to modify the given tracks.
        This is useful for creating composite actions from the low-level actions.
        Composite actions can contain application logic and can be un-done as a group.

        Args:
            tracks (Tracks): The tracks that this action will edit
            actions (list[TracksAction]): A list of actions contained within the group,
                in the order in which they should be executed.
        """
        super().__init__(tracks)
        self.actions = actions

    @override
    def inverse(self) -> ActionGroup:
        actions = [action.inverse() for action in self.actions[::-1]]
        return ActionGroup(self.tracks, actions)


class AddNodes(TracksAction):
    """Action for adding new nodes. If a segmentation should also be added, the
    pixels for each node should be provided. The label to set the pixels will
    be taken from the node id. The existing pixel values are assumed to be
    zero - you must explicitly update any other segmentations that were overwritten
    using an UpdateNodes action if you want to be able to undo the action.
    """

    def __init__(
        self,
        tracks: Tracks,
        nodes: Iterable[Node],
        attributes: Attrs,
        pixels: Iterable[SegMask] | None = None,
    ):
        """Create an action to add new nodes, with optional segmentation

        Args:
            tracks (Tracks): The Tracks to add the nodes to
            nodes (Node): A list of node ids
            attributes (Attrs): Includes times and optionally positions
            pixels (list[SegMask] | None, optional): The segmentations associated with
                each node. Defaults to None.
        """
        super().__init__(tracks)
        self.nodes = nodes
        user_attrs = attributes.copy()
        self.times = attributes.pop(NodeAttr.TIME.value, None)
        self.positions = attributes.pop(NodeAttr.POS.value, None)
        self.pixels = pixels
        self.attributes = user_attrs
        self._apply()

    def inverse(self):
        """Invert the action to delete nodes instead"""
        return DeleteNodes(self.tracks, self.nodes)

    def _apply(self):
        """Apply the action, and set segmentation if provided in self.pixels"""

        attrs = self.attributes
        if attrs is None:
            attrs = {}

        final_pos: np.ndarray
        if self.tracks.segmentation is not None:
            if self.pixels is not None:
                computed_attrs = compute_node_attrs_from_pixels(
                    self.pixels, self.tracks.ndim, self.tracks.scale
                )  # if self.pixels is not None else computed_attrs
            elif "mask" in attrs:
                computed_attrs = compute_node_attrs_from_masks(
                    attrs["mask"], self.tracks.ndim, self.tracks.scale
                )
            if self.positions is None:
                final_pos = np.array(computed_attrs[NodeAttr.POS.value])
            else:
                final_pos = self.positions
            attrs[NodeAttr.AREA.value] = computed_attrs[NodeAttr.AREA.value]
        elif self.positions is None:
            raise ValueError("Must provide positions or segmentation and ids")
        else:
            final_pos = self.positions

        self.attributes[NodeAttr.POS.value] = final_pos

        # Add nodes to td graph
        required_attrs = self.tracks.graph.node_attr_keys.copy()
        if td.DEFAULT_ATTR_KEYS.NODE_ID in required_attrs:
            required_attrs.remove(td.DEFAULT_ATTR_KEYS.NODE_ID)
        if td.DEFAULT_ATTR_KEYS.SOLUTION not in attrs:
            attrs[td.DEFAULT_ATTR_KEYS.SOLUTION] = [1] * len(self.nodes)
        for attr in required_attrs:
            if attr not in attrs:
                attrs[attr] = [None] * len(self.nodes)

        node_dicts = []
        for i in range(len(self.nodes)):
            node_dict = {
                attr: np.array(values[i]) if attr == "pos" else values[i]
                for attr, values in attrs.items()
            }
            node_dicts.append(node_dict)

        for node_id, node_dict in zip(self.nodes, node_dicts, strict=True):
            # TODO: Teun: graph is now always a graphview, by definition!
            self.tracks.graph.add_node(attrs=node_dict, index=node_id)

        if self.pixels is not None:
            self.tracks.set_pixels(self.pixels, self.nodes)

        if isinstance(self.tracks, SolutionTracks):
            for node, track_id in zip(
                self.nodes, attrs[NodeAttr.TRACK_ID.value], strict=True
            ):
                if track_id not in self.tracks.track_id_to_node:
                    self.tracks.track_id_to_node[track_id] = []
                self.tracks.track_id_to_node[track_id].append(node)


class DeleteNodes(TracksAction):
    """Action of deleting existing nodes
    If the tracks contain a segmentation, this action also constructs a reversible
    operation for setting involved pixels to zero
    """

    def __init__(
        self,
        tracks: Tracks,
        nodes: Iterable[Node],
        pixels: Iterable[SegMask] | None = None,
    ):
        super().__init__(tracks)
        self.nodes = nodes
        self.attributes = {
            NodeAttr.TIME.value: self.tracks.get_times(nodes),
            self.tracks.pos_attr: self.tracks.get_positions(nodes),
            NodeAttr.TRACK_ID.value: self.tracks.get_nodes_attr(
                nodes, NodeAttr.TRACK_ID.value
            ),
            NodeAttr.AREA.value: self.tracks.get_nodes_attr(nodes, NodeAttr.AREA.value),
            td.DEFAULT_ATTR_KEYS.SOLUTION: self.tracks.get_nodes_attr(
                nodes, td.DEFAULT_ATTR_KEYS.SOLUTION
            ),
            td.DEFAULT_ATTR_KEYS.MASK: self.tracks.get_nodes_attr(
                nodes, td.DEFAULT_ATTR_KEYS.MASK
            ),
            td.DEFAULT_ATTR_KEYS.BBOX: self.tracks.get_nodes_attr(
                nodes, td.DEFAULT_ATTR_KEYS.BBOX
            ),
        }
        self.pixels = self.tracks.get_pixels(nodes) if pixels is None else pixels
        self._apply()

    def inverse(self):
        """Invert this action, and provide inverse segmentation operation if given"""

        return AddNodes(self.tracks, self.nodes, self.attributes, pixels=self.pixels)

    def _apply(self):
        """ASSUMES THERE ARE NO INCIDENT EDGES - raises valueerror if an edge will be
        removed by this operation
        Steps:
        - For each node
            set pixels to 0 if self.pixels is provided
        - Remove nodes from graph
        """

        if isinstance(self.tracks, SolutionTracks):
            for node in self.nodes:
                self.tracks.track_id_to_node[self.tracks.get_track_id(node)].remove(node)

        for node in self.nodes:
            self.tracks.graph.node_removed.emit_fast(node)
            self.tracks.graph.rx_graph.remove_node(
                self.tracks.graph._external_to_local[node]
            )
            self.tracks.graph._external_to_local.pop(node)

        self.tracks.graph._root.update_node_attrs(
            attrs={td.DEFAULT_ATTR_KEYS.SOLUTION: [0] * len(self.nodes)},
            node_ids=self.nodes,
        )


class UpdateNodeSegs(TracksAction):
    """Action for updating the segmentation associated with nodes. Cannot mix adding
    and removing pixels from segmentation: the added flag applies to all nodes"""

    def __init__(
        self,
        tracks: Tracks,
        nodes: Iterable[Node],
        pixels: Iterable[SegMask],
        added: bool = True,
    ):
        """
        Args:
            tracks (Tracks): The tracks to update the segmenatations for
            nodes (list[Node]): The nodes with updated segmenatations
            pixels (list[SegMask]): The pixels that were updated for each node
            added (bool, optional): If the provided pixels were added (True) or deleted
                (False) from all nodes. Defaults to True. Cannot mix adding and deleting
                pixels in one action.
        """
        super().__init__(tracks)
        self.nodes = nodes
        self.pixels = pixels
        self.added = added
        self._apply()

    def inverse(self):
        """Restore previous attributes"""
        return UpdateNodeSegs(
            self.tracks,
            self.nodes,
            pixels=self.pixels,
            added=not self.added,
        )

    def _apply(self):
        """Set new attributes"""
        values = self.nodes if self.added else [0 for _ in self.nodes]

        self.tracks.set_pixels(self.pixels, values, self.added, self.nodes)
        mask_list = [self.tracks.graph[n][td.DEFAULT_ATTR_KEYS.MASK] for n in self.nodes]
        computed_attrs = compute_node_attrs_from_masks(
            mask_list, self.tracks.ndim, self.tracks.scale
        )
        positions = np.array(computed_attrs[NodeAttr.POS.value])
        self.tracks.set_positions(self.nodes, positions)
        self.tracks._set_nodes_attr(
            self.nodes, NodeAttr.AREA.value, computed_attrs[NodeAttr.AREA.value]
        )

        # Get all incident edges using predecessors and successors
        incident_edges = []
        for node in self.nodes:
            # Add edges from predecessors
            for pred in td_get_predecessors(self.tracks.graph, node):
                incident_edges.append((pred, node))
            # Add edges from successors
            for succ in td_get_successors(self.tracks.graph, node):
                incident_edges.append((node, succ))
        for edge in incident_edges:
            new_edge_attrs = self.tracks._compute_edge_attrs([edge])
            self.tracks._set_edge_attributes([edge], new_edge_attrs)


class UpdateNodeAttrs(TracksAction):
    """Action for user updates to node attributes. Cannot update protected
    attributes (time, area, track id), as these are controlled by internal application
    logic."""

    def __init__(
        self,
        tracks: Tracks,
        nodes: Iterable[Node],
        attrs: Attrs,
    ):
        """
        Args:
            tracks (Tracks): The tracks to update the node attributes for
            nodes (Iterable[Node]): The nodes to update the attributes for
            attrs (Attrs): A mapping from attribute name to list of new attribute values
                for the given nodes.

        Raises:
            ValueError: If a protected attribute is in the given attribute mapping.
        """
        super().__init__(tracks)
        protected_attrs = [
            tracks.time_attr,
            NodeAttr.AREA.value,
            NodeAttr.TRACK_ID.value,
        ]
        for attr in attrs:
            if attr in protected_attrs:
                raise ValueError(f"Cannot update attribute {attr} manually")
        self.nodes = nodes
        self.prev_attrs = {
            attr: self.tracks.get_nodes_attr(nodes, attr) for attr in attrs
        }
        self.new_attrs = attrs
        self._apply()

    def inverse(self):
        """Restore previous attributes"""
        return UpdateNodeAttrs(
            self.tracks,
            self.nodes,
            self.prev_attrs,
        )

    def _apply(self):
        """Set new attributes"""
        for attr, values in self.new_attrs.items():
            self.tracks._set_nodes_attr(self.nodes, attr, values)


class AddEdges(TracksAction):
    """Action for adding new edges"""

    def __init__(self, tracks: Tracks, edges: Iterable[Edge]):
        super().__init__(tracks)
        self.edges = edges
        self._apply()

    def inverse(self):
        """Delete edges"""
        return DeleteEdges(self.tracks, self.edges)

    def _apply(self):
        """
        Steps:
        - add each edge to the graph. Assumes all edges are valid (they should be checked
        at this point already)
        """

        for edge in self.edges:
            if edge in td_graph_edge_list(self.tracks.graph):
                raise ValueError(f"Edge {edge} already exists in the graph")

        attrs: dict[str, Sequence[Any]] = {}
        attrs.update(self.tracks._compute_edge_attrs(self.edges))
        attrs[td.DEFAULT_ATTR_KEYS.SOLUTION] = [1] * len(self.edges)

        required_attrs = self.tracks.graph.edge_attr_keys
        for attr in required_attrs:
            if attr not in attrs:
                attrs[attr] = [None] * len(self.edges)

        for idx, edge in enumerate(self.edges):
            for node in edge:
                if node not in self.tracks.graph.node_ids():
                    raise KeyError(
                        f"Cannot add edge {edge}: endpoint {node} not in graph yet"
                    )

            edge = list(edge)

            edge_in_root = self.tracks.graph._root.has_edge(edge[0], edge[1])
            if edge_in_root:
                edge_id = self.tracks.graph._root.edge_id(edge[0], edge[1])

                # Check if edge is not in solution
                edge_in_solution = (
                    self.tracks.graph._root.edge_attrs()
                    .filter(pl.col(td.DEFAULT_ATTR_KEYS.EDGE_ID) == edge_id)[
                        td.DEFAULT_ATTR_KEYS.SOLUTION
                    ]
                    .item()
                )

                if not edge_in_solution:
                    # Reactivate edge in root for future usage
                    self.tracks.graph._root.update_edge_attrs(
                        edge_ids=[edge_id], attrs={td.DEFAULT_ATTR_KEYS.SOLUTION: [1]}
                    )
            else:
                edge_id = (
                    max(self.tracks.graph.edge_ids()) + 1
                    if len(self.tracks.graph.edge_ids()) > 0
                    else 0
                )

            edge_attrs = {key: vals[idx] for key, vals in attrs.items()}
            edge_attrs[td.DEFAULT_ATTR_KEYS.EDGE_ID] = edge_id

            # Create edge attributes for this specific edge
            self.tracks.graph.add_edge(
                source_id=edge[0],
                target_id=edge[1],
                attrs=edge_attrs,
            )


class DeleteEdges(TracksAction):
    """Action for deleting edges"""

    def __init__(self, tracks: Tracks, edges: Iterable[Edge]):
        super().__init__(tracks)
        self.edges = edges
        self._apply()

    def inverse(self):
        """Restore edges and their attributes"""
        return AddEdges(self.tracks, self.edges)

    def _apply(self):
        """Steps:
        - Remove the edges from the graph
        """
        for edge in self.edges:
            edge_id_to_remove = self.tracks.graph.edge_id(edge[0], edge[1])
            self.tracks.graph.rx_graph.remove_edge(
                self.tracks.graph._external_to_local[edge[0]],
                self.tracks.graph._external_to_local[edge[1]],
            )
            self.tracks.graph._root.update_edge_attrs(
                edge_ids=[edge_id_to_remove], attrs={td.DEFAULT_ATTR_KEYS.SOLUTION: [0]}
            )
            self.tracks.graph._edge_map_from_root.pop(edge_id_to_remove)


class UpdateTrackID(TracksAction):
    def __init__(self, tracks: SolutionTracks, start_node: Node, track_id: int):
        """
        Args:
            tracks (Tracks): The tracks to update
            start_node (Node): The node ID of the first node in the track. All successors
                with the same track id as this node will be updated.
            track_id (int): The new track id to assign.
        """
        super().__init__(tracks)
        self.tracks: SolutionTracks = tracks
        self.start_node = start_node
        self.old_track_id = self.tracks.get_track_id(start_node)
        self.new_track_id = track_id
        self._apply()

    def inverse(self) -> TracksAction:
        """Restore the previous track_id"""
        return UpdateTrackID(self.tracks, self.start_node, self.old_track_id)

    def _apply(self):
        """Assign a new track id to the track starting with start_node."""
        old_track_id = self.tracks.get_track_id(self.start_node)
        curr_node = self.start_node
        while self.tracks.get_track_id(curr_node) == old_track_id:
            # update the track id
            self.tracks.set_track_id(curr_node, self.new_track_id)
            # getting the next node (picks one if there are two)
            successors = td_get_successors(self.tracks.graph, curr_node)
            if len(successors) == 0:
                break
            curr_node = successors[0]
