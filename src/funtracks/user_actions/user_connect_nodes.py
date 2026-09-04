from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from funtracks.exceptions import InvalidActionError

from ..actions._base import ActionGroup
from .user_add_edge import UserAddEdge
from .user_delete_edge import UserDeleteEdge

if TYPE_CHECKING:
    from collections.abc import Iterable

    from funtracks.data_model import Tracks
    from funtracks.data_model.tracks import Node


def _validate(tracks: Tracks, nodes: Iterable[Node]) -> list[Node]:
    """De-duplicate the nodes and check that they all exist in the solution."""

    nodes = list(dict.fromkeys(nodes))  # de-duplicate, keep order
    if len(nodes) < 2:
        raise InvalidActionError("Select at least two nodes to connect.")
    for node in nodes:
        if not tracks.graph_solution.has_node(node):
            raise InvalidActionError(f"Node {node} not in solution, can't connect")
    return nodes


def _sort_by_time(tracks: Tracks, nodes: list[Node]) -> list[Node]:
    """Sort the nodes by time, raising if two of them share a time point."""

    times = dict(zip(nodes, tracks.get_times(nodes), strict=True))
    by_time: dict[int, list[Node]] = {}
    for node, time in times.items():
        by_time.setdefault(time, []).append(node)
    horizontal = {time: group for time, group in by_time.items() if len(group) > 1}
    if horizontal:
        details = "; ".join(
            f"nodes {group} at time {time}" for time, group in horizontal.items()
        )
        raise InvalidActionError(
            "Cannot connect nodes that are in the same time point: "
            f"{details}. Please select at most one node per time point.",
            forceable=False,
        )
    return sorted(nodes, key=times.__getitem__)


def _get_pairs(tracks: Tracks, nodes: Iterable[Node]) -> list[tuple[Node, Node]]:
    """Get the consecutive (source, target) pairs of the time sorted selection."""

    sorted_nodes = _sort_by_time(tracks, _validate(tracks, nodes))
    return list(zip(sorted_nodes[:-1], sorted_nodes[1:], strict=True))


def _find_conflicts(
    tracks: Tracks, pairs: list[tuple[Node, Node]], linear: bool
) -> list[tuple[Node, Node]]:
    """Collect the existing edges that prevent the chain from being created.

    An edge conflicts if it is an incoming edge of one of the targets (a node can only
    have one parent), or if it is an outgoing edge of one of the sources that would
    push that source beyond a two-way division. In linear mode, every outgoing edge of
    a source that leaves the chain conflicts, so that no divisions are left behind.
    """

    conflicts: list[tuple[Node, Node]] = []

    for source, target in pairs:
        conflicts.extend(
            (pred, target) for pred in tracks.predecessors(target) if pred != source
        )

    for source, target in pairs:
        kept = [
            succ
            for succ in tracks.successors(source)
            if succ != target and (source, succ) not in conflicts
        ]
        # in linear mode no sibling may survive, otherwise only a three-way division
        # is a problem
        if kept and (linear or len(kept) + 1 > 2):
            conflicts.extend((source, succ) for succ in kept)

    return list(dict.fromkeys(conflicts))


class UserConnectNodes(ActionGroup):
    """Connect a set of selected nodes into a single chain, or disconnect them again.

    The selected nodes are sorted by time and connected pairwise, so that they form
    one chain. Gaps in time are allowed: the nodes do not need to be in consecutive
    time points. If the source of a new edge has no successor yet, the tracklet id of
    the first (earliest) node propagates along the whole chain.

    A source that already has exactly one successor outside of the selection is
    handled according to ``linear``. By default a division is created there, and the
    regular division track id rules apply (see :class:`UserAddEdge`). With
    ``linear=True`` that existing edge is treated as a conflict instead, so that the
    selection ends up as one linear track without divisions. Outgoing edges of the
    last node of the chain are never touched, since no new edge is added there.

    This action is its own inverse in the following sense: if all consecutive pairs
    of the selection are already connected, the edges between them are removed
    instead, splitting the selection into separate tracklets that each get their own
    tracklet id. If even one pair is not connected yet, the action connects the
    missing pairs and never removes any of the existing chain edges.

    Args:
        tracks (Tracks): The tracks to connect the nodes in.
        nodes (Iterable[Node]): The nodes to connect (or disconnect). At least two
            nodes are required, and no two nodes may be in the same time point.
        linear (bool, optional): Whether to connect the nodes into a linear track,
            treating existing outgoing edges of the sources as conflicts instead of
            creating divisions. Defaults to False.
        force (bool, optional): Whether to force the action by removing conflicting
            edges. Defaults to False.
        _top_level (bool): If True, add this action to the history and emit refresh.
            Set to False when used as a sub-action inside a compound action.
            Defaults to True.

    Raises:
        InvalidActionError: If fewer than two distinct nodes are given, if a node is
            not in the solution graph, or if two nodes share a time point (never
            forceable). If connecting would conflict with existing edges, a forceable
            InvalidActionError is raised unless ``force`` is True.
    """

    def __init__(
        self,
        tracks: Tracks,
        nodes: Iterable[Node],
        linear: bool = False,
        force: bool = False,
        _top_level: bool = True,
    ):
        super().__init__(tracks, actions=[])
        self.tracks: Tracks  # Narrow type from base class

        pairs = _get_pairs(tracks, nodes)
        missing = [edge for edge in pairs if not tracks.graph_solution.has_edge(*edge)]
        if not missing:
            self._disconnect(pairs)
        else:
            self._connect(pairs, missing, linear=linear, force=force)

        if _top_level:
            self.tracks.action_history.add_new_action(self)
            self.tracks.refresh.emit()

    @staticmethod
    def has_division_choice(tracks: Tracks, nodes: Iterable[Node]) -> bool:
        """Whether connecting these nodes linearly differs from connecting them with
        divisions, i.e. whether it is worth asking the user which one they want.

        Returns False for a selection that cannot be connected at all, or that would
        be disconnected instead.
        """

        try:
            pairs = _get_pairs(tracks, nodes)
        except InvalidActionError:
            return False
        if all(tracks.graph_solution.has_edge(*edge) for edge in pairs):
            return False  # this selection would be disconnected instead
        with_divisions = _find_conflicts(tracks, pairs, linear=False)
        linear = _find_conflicts(tracks, pairs, linear=True)
        return set(with_divisions) != set(linear)

    def _disconnect(self, pairs: list[tuple[Node, Node]]) -> None:
        """Remove the edges between all consecutive selected nodes."""

        for edge in pairs:
            self.actions.append(UserDeleteEdge(self.tracks, edge, _top_level=False))

    def _connect(
        self,
        pairs: list[tuple[Node, Node]],
        missing: list[tuple[Node, Node]],
        linear: bool,
        force: bool,
    ) -> None:
        """Add the missing edges, first removing conflicting edges if forced."""

        conflicts = _find_conflicts(self.tracks, pairs, linear=linear)
        if conflicts:
            if not force:
                listed = ", ".join(str(edge) for edge in conflicts)
                raise InvalidActionError(
                    f"Cannot connect the selected nodes: edge(s) {listed} conflict "
                    "with the requested connection.",
                    forceable=True,
                )
            warnings.warn(
                f"Removing edge(s) {conflicts} to connect the selected nodes.",
                stacklevel=3,
            )
            for edge in conflicts:
                self.actions.append(UserDeleteEdge(self.tracks, edge, _top_level=False))

        for edge in missing:
            self.actions.append(UserAddEdge(self.tracks, edge, _top_level=False))
