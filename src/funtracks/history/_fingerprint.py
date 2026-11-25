"""Identify the graph state a history log starts from."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from funtracks.data_model import Tracks

# The fingerprint fields that replay checks; the others are information only
CHECKED_FIELDS = ("n_nodes", "n_edges", "max_node_id", "digest")


def fingerprint(tracks: Tracks, digest: bool = True) -> dict[str, Any]:
    """Fingerprint of the solution's nodes and edges.

    Track ids are deliberately left out: they are recomputed from the edges and can
    differ between loads of the same graph.

    Args:
        tracks: The tracks to fingerprint.
        digest: If True, include a SHA-256 over the sorted node ids and edges. Pass
            False to skip it on huge graphs; it is then recorded as None.

    Returns:
        dict[str, Any]: ``n_nodes``, ``n_edges``, ``max_node_id`` and ``digest`` of
            the solution (checked on replay), plus ``n_nodes_full`` and
            ``n_edges_full`` of ``graph_full`` (information only).
    """
    solution = tracks.graph_solution
    node_ids = np.sort(np.asarray(solution.node_ids(), dtype=np.int64))
    edges = np.asarray(solution.edge_list(), dtype=np.int64).reshape(-1, 2)
    result: dict[str, Any] = {
        "n_nodes": int(node_ids.size),
        "n_edges": int(edges.shape[0]),
        "max_node_id": int(node_ids[-1]) if node_ids.size else None,
        "digest": None,
        "n_nodes_full": int(tracks.graph_full.num_nodes()),
        "n_edges_full": int(tracks.graph_full.num_edges()),
    }
    if digest:
        # Sort edges by source, then target
        edges = edges[np.lexsort((edges[:, 1], edges[:, 0]))]
        sha = hashlib.sha256()
        sha.update(node_ids.tobytes())
        sha.update(np.ascontiguousarray(edges).tobytes())
        result["digest"] = "sha256:" + sha.hexdigest()
    return result


def baseline_mismatches(expected: dict[str, Any], actual: dict[str, Any]) -> list[str]:
    """The checked fingerprint fields that differ, as readable messages.

    ``digest`` is only compared when both sides have one.

    Args:
        expected: The fingerprint recorded in the log.
        actual: The fingerprint of the tracks being replayed onto.

    Returns:
        list[str]: One message per differing field; empty if they match.
    """
    mismatches = []
    for field in CHECKED_FIELDS:
        a, b = expected.get(field), actual.get(field)
        if field == "digest" and (a is None or b is None):
            continue
        if a != b:
            mismatches.append(f"{field}: log has {a!r}, tracks have {b!r}")
    return mismatches
