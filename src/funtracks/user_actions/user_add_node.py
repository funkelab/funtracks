from __future__ import annotations

from typing import Any

import numpy as np

from funtracks.data_model.graph_attributes import NodeAttr
from funtracks.data_model.solution_tracks import SolutionTracks
from funtracks.exceptions import InvalidActionError

from ..actions._base import ActionGroup
from ..actions.add_delete_edge import AddEdge, DeleteEdge
from ..actions.add_delete_node import AddNode


class UserAddNode(ActionGroup):
    """Determines which basic actions to call when a user adds a node

    - Get the track id
    - Check if the track has divided earlier in time -> raise InvalidActionException
    - Check if there is an earlier and/or later node in the track
        - If there is earlier and later node, remove the edge between them
        - Add edges between the earlier/later nodes and the new node
    """

    def __init__(
        self,
        tracks: SolutionTracks,
        node: int,
        attributes: dict[str, Any],
        pixels: tuple[np.ndarray, ...] | None = None,
    ):
        """
        Args:
            tracks (SolutionTracks): the tracks to add the node to
            node (int): The node id of the new node to add
            attributes (dict[str, Any]): A dictionary from attribute strings to values.
                Must contain "time" and "track_id".
            pixels (tuple[np.ndarray, ...] | None, optional): The pixels of the associated
                segmentation to add to the tracks. Defaults to None.

        Raises:
            ValueError: If the attributes dictionary does not contain either `time` or
                `track_id`.
            ValueError: If a node with the given ID already exists in the tracks.
            InvalidActionError: If the node is trying to be added to a track that
                divided in a previous time point.
        """
        super().__init__(tracks, actions=[])
        if NodeAttr.TIME.value not in attributes:
            raise ValueError(
                f"Cannot add node without time. Please add "
                f"{NodeAttr.TIME.value} attribute"
            )
        if NodeAttr.TRACK_ID.value not in attributes:
            raise ValueError(
                "Cannot add node without track id. Please add "
                f"{NodeAttr.TRACK_ID.value} attribute"
            )
        if self.tracks.graph.has_node(node):
            raise ValueError(f"Node {node} already exists in the tracks, cannot add.")

        track_id = attributes[NodeAttr.TRACK_ID.value]
        time = attributes[NodeAttr.TIME.value]
        pred, succ = self.tracks.get_track_neighbors(track_id, time)
        # check if you are adding a node to a track that divided previously
        if pred is not None and self.tracks.graph.out_degree(pred) == 2:
            raise InvalidActionError(
                "Cannot add node here - upstream division event detected."
            )
        # remove skip edge that will be replaced by new edges after adding nodes
        if pred is not None and succ is not None:
            self.actions.append(DeleteEdge(tracks, (pred, succ)))
        # add predecessor and successor edges
        self.actions.append(AddNode(tracks, node, attributes, pixels))
        if pred is not None:
            self.actions.append(AddEdge(tracks, (pred, node)))
        if succ is not None:
            self.actions.append(AddEdge(tracks, (node, succ)))
