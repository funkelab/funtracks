from __future__ import annotations

from typing import TYPE_CHECKING

from funtracks.exceptions import InvalidActionError

from ..actions._base import ActionGroup
from ..actions.update_segmentation import UpdateNodeSeg
from .user_delete_node import UserDeleteNode

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from funtracks.data_model import Tracks
    from funtracks.data_model.tracks import Node


def _validate(tracks: Tracks, nodes: Iterable[Node]) -> list[Node]:
    """De-duplicate the nodes and check that they all exist in the solution."""

    nodes = list(dict.fromkeys(nodes))  # de-duplicate, keep order
    if len(nodes) < 2:
        raise InvalidActionError("Select at least two nodes to merge.")
    for node in nodes:
        if not tracks.graph_solution.has_node(node):
            raise InvalidActionError(f"Node {node} not in solution, can't merge")
    return nodes


def get_merge_groups(tracks: Tracks, nodes: Iterable[Node]) -> dict[int, list[Node]]:
    """Group the selected nodes into the horizontal sets that will be merged.

    Nodes that are alone in their time point cannot be merged with anything and
    are left out of the result.

    Args:
        tracks (Tracks): The tracks the nodes belong to.
        nodes (Iterable[Node]): The selected nodes.

    Returns:
        dict[int, list[Node]]: A mapping from time point to the nodes selected in
            that time point, only for time points holding more than one node, and
            sorted by time.

    Raises:
        InvalidActionError: If fewer than two distinct nodes are given, if a node is
            not in the solution graph, or if no two selected nodes share a time point.
    """

    nodes = _validate(tracks, nodes)
    times = tracks.get_times(nodes)
    by_time: dict[int, list[Node]] = {}
    for node, time in zip(nodes, times, strict=True):
        by_time.setdefault(int(time), []).append(node)

    groups = {time: group for time, group in sorted(by_time.items()) if len(group) > 1}
    if not groups:
        raise InvalidActionError(
            "Cannot merge the selected nodes: no two of them share a time point. "
            "Please select at least two nodes in the same time point."
        )
    return groups


def get_track_id_options(tracks: Tracks, nodes: Iterable[Node]) -> dict[int, list[int]]:
    """Get the tracklet ids that can be picked for each horizontal set.

    Args:
        tracks (Tracks): The tracks the nodes belong to.
        nodes (Iterable[Node]): The selected nodes.

    Returns:
        dict[int, list[int]]: A mapping from time point to the sorted tracklet ids of
            the nodes selected in that time point.

    Raises:
        InvalidActionError: See :func:`get_merge_groups`.
    """

    return {
        time: sorted(tracks.get_track_id(node) for node in group)
        for time, group in get_merge_groups(tracks, nodes).items()
    }


class UserMergeNodes(ActionGroup):
    """Merge each horizontal set of selected nodes into a single node.

    The selection is grouped by time point, and every group holding more than one node
    is merged. Selected nodes that are alone in their time point are ignored. Merging
    works like flood filling one label with another: the masks of the other nodes in the
    group are added to the node that carries the requested tracklet id, and those other
    nodes are deleted. The surviving node therefore keeps its own id and edges.

    The requested tracklet id picks the node to merge into; it is not forced onto the
    result afterwards. Deleting the other nodes can still re-assign tracklet ids through
    the usual rules, for example when the merged nodes were the two children of a
    division and that division collapses into a single track.

    Args:
        tracks (Tracks): The tracks to merge the nodes in.
        nodes (Iterable[Node]): The nodes to merge. At least two of them must share a
            time point.
        track_ids (int | Mapping[int, int] | None, optional): The tracklet id to keep.
            Either a single tracklet id, which must be present in every merged group,
            or a mapping from time point to the tracklet id to keep in that group. If
            None, the lowest tracklet id of each group is kept. Defaults to None.
        _top_level (bool): If True, add this action to the history and emit refresh.
            Set to False when used as a sub-action inside a compound action.
            Defaults to True.

    Raises:
        InvalidActionError: If fewer than two distinct nodes are given, if a node is
            not in the solution graph, if no two selected nodes share a time point, if
            the tracks have no segmentation, or if a requested tracklet id does not
            name exactly one node of its group.
    """

    def __init__(
        self,
        tracks: Tracks,
        nodes: Iterable[Node],
        track_ids: int | Mapping[int, int] | None = None,
        _top_level: bool = True,
    ):
        super().__init__(tracks, actions=[])
        self.tracks: Tracks  # Narrow type from base class

        if tracks.segmentation is None:
            raise InvalidActionError("Cannot merge nodes without a segmentation.")

        # Resolve every group to the node it merges into before applying anything:
        # merging one group can re-assign tracklet ids elsewhere in the selection (a
        # collapsing division, for instance), and the requested ids name the nodes as
        # they are now, not as they will be halfway through the merge.
        plan = [
            (group, self._get_node_to_keep(group, time, track_ids))
            for time, group in get_merge_groups(tracks, nodes).items()
        ]
        self.kept_nodes = [node for _, node in plan]
        for group, node_to_keep in plan:
            self._merge_group(group, node_to_keep)

        if _top_level:
            self.tracks.action_history.add_new_action(self)
            self.tracks.refresh.emit(self.kept_nodes[-1])

    def _get_node_to_keep(
        self,
        group: list[Node],
        time: int,
        track_ids: int | Mapping[int, int] | None,
    ) -> Node:
        """Get the node of the group that carries the requested tracklet id."""

        if track_ids is None:
            track_id = min(self.tracks.get_track_id(node) for node in group)
        elif isinstance(track_ids, int):
            track_id = track_ids
        elif time in track_ids:
            track_id = track_ids[time]
        else:
            raise InvalidActionError(
                f"No tracklet id given for the nodes {group} in time point {time}."
            )

        keep = [node for node in group if self.tracks.get_track_id(node) == track_id]
        if len(keep) != 1:
            raise InvalidActionError(
                f"Tracklet id {track_id} names {len(keep)} of the nodes {group}, "
                "expected exactly one node to merge into."
            )
        return keep[0]

    def _merge_group(self, group: list[Node], node_to_keep: Node) -> None:
        """Merge one horizontal set into the given node."""

        for node in group:
            if node == node_to_keep:
                continue
            # Grow the kept node by the other node's mask, then remove that node, the
            # same two steps a flood fill takes when it paints over a label.
            self.actions.append(
                UpdateNodeSeg(
                    self.tracks, node_to_keep, self.tracks.get_mask(node), added=True
                )
            )
            self.actions.append(UserDeleteNode(self.tracks, node, _top_level=False))
