"""Shared logic for connecting and disconnecting a selection of nodes.

Both :class:`UserConnectNodes` and :class:`UserDisconnectNodes` treat a selection as
one chain: the nodes are sorted by time and paired up consecutively, and the action
operates on the edges between those pairs. Gaps in time are allowed, so the nodes do
not need to be in consecutive time points, but two nodes in the same time point cannot
be chained and are rejected.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from funtracks.exceptions import InvalidActionError

if TYPE_CHECKING:
    from collections.abc import Iterable

    from funtracks.data_model import Tracks
    from funtracks.data_model.tracks import Node


def validate_nodes(tracks: Tracks, nodes: Iterable[Node]) -> list[Node]:
    """De-duplicate the nodes and check that they all exist in the solution."""

    nodes = list(dict.fromkeys(nodes))  # de-duplicate, keep order
    if len(nodes) < 2:
        raise InvalidActionError("Select at least two nodes.")
    for node in nodes:
        if not tracks.graph_solution.has_node(node):
            raise InvalidActionError(f"Node {node} not in solution")
    return nodes


def sort_by_time(tracks: Tracks, nodes: list[Node]) -> list[Node]:
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
            "Cannot chain nodes that are in the same time point: "
            f"{details}. Please select at most one node per time point.",
            forceable=False,
        )
    return sorted(nodes, key=times.__getitem__)


def connection_pairs(tracks: Tracks, nodes: Iterable[Node]) -> list[tuple[Node, Node]]:
    """Get the consecutive (source, target) pairs of the time sorted selection.

    Raises:
        InvalidActionError: If fewer than two distinct nodes are given, if a node is
            not in the solution graph, or if two nodes share a time point.
    """

    sorted_nodes = sort_by_time(tracks, validate_nodes(tracks, nodes))
    return list(zip(sorted_nodes[:-1], sorted_nodes[1:], strict=True))


def find_conflicts(
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


def is_connected_chain(tracks: Tracks, nodes: Iterable[Node]) -> bool:
    """Whether the selection already forms one connected chain in time order, so that
    there is nothing left to connect and it can only be disconnected.

    Returns False for a selection that cannot be chained at all (fewer than two nodes,
    a node that is not in the solution, or two nodes in the same time point), so that
    the caller can go ahead and let :class:`UserConnectNodes` raise the real error.
    """

    try:
        pairs = connection_pairs(tracks, nodes)
    except InvalidActionError:
        return False
    return all(tracks.graph_solution.has_edge(*edge) for edge in pairs)
