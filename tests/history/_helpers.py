"""Shared helpers for the history log tests."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from funtracks.cli import compare_tracks
from funtracks.data_model import Tracks
from funtracks.exceptions import InvalidActionError
from funtracks.features import Feature
from funtracks.history import recompute_track_ids
from funtracks.user_actions import (
    UserAddEdge,
    UserAddFeature,
    UserAddNode,
    UserConnectNodes,
    UserDeleteEdge,
    UserDeleteFeature,
    UserDeleteNode,
    UserDisconnectNodes,
    UserMergeNodes,
    UserSetDivision,
    UserSwapPredecessors,
    UserUpdateNodesAttrs,
    UserUpdateSegmentation,
)
from funtracks.utils.tracksdata_utils import td_mask_to_pixels

SOURCE = {"kind": "none", "path": None}


def group_feature(name: str = "group") -> Feature:
    return Feature(
        feature_type="node",
        value_type="bool",
        num_values=1,
        display_name=name,
        default_value=False,
    )


def partition(tracks: Tracks, key: str) -> set[frozenset[int]]:
    groups: dict = defaultdict(set)
    for node in tracks.graph_solution.node_ids():
        groups[tracks.get_node_attr(node, key)].add(int(node))
    return {frozenset(g) for g in groups.values()}


def track_partitions(tracks: Tracks) -> tuple[set, set]:
    return (
        partition(tracks, tracks.features.tracklet_key),
        partition(tracks, tracks.features.lineage_key),
    )


def full_graph_state(tracks: Tracks) -> tuple[dict, dict]:
    """Every node and edge in graph_full with its solution flag."""
    nodes = {
        int(n): bool(tracks.get_node_attr(n, "solution"))
        for n in tracks.graph_full.node_ids()
    }
    edges = {
        (row["source_id"], row["target_id"]): bool(row["solution"])
        for row in tracks.graph_full.edge_attrs(attr_keys=["solution"]).to_dicts()
    }
    return nodes, edges


def assert_live_track_ids_match_topology(tracks: Tracks) -> None:
    """The incrementally kept track ids group nodes like a fresh recompute does."""
    live = track_partitions(tracks)
    recompute_track_ids(tracks)
    assert track_partitions(tracks) == live


def assert_replay_matches(replayed: Tracks, session: Tracks) -> None:
    """The replayed tracks equal the session's, track id numbers aside."""
    recompute_track_ids(session)
    assert compare_tracks(replayed, session) == []
    assert full_graph_state(replayed) == full_graph_state(session)


def pixels(tracks: Tracks, node: int) -> tuple[np.ndarray, ...]:
    return td_mask_to_pixels(
        tracks.get_mask(node), tracks.get_time(node), ndim=tracks.ndim
    )


# Errors a user action raises for an edit that is not allowed
EXPECTED_ERRORS = (InvalidActionError, ValueError)


def random_session(tracks: Tracks, seed: int, n_steps: int) -> int:
    """Apply random user actions, undos and redos. Returns how many succeeded."""
    rng = np.random.default_rng(seed)
    with_seg = tracks.segmentation is not None
    succeeded = 0
    for _ in range(n_steps):
        nodes = sorted(tracks.graph_solution.node_ids())
        edges = [tuple(e) for e in tracks.graph_solution.edge_list()]
        choice = rng.integers(0, 14)
        try:
            if choice == 0:
                tracks.undo()
            elif choice == 1:
                tracks.redo()
            elif choice == 2 and nodes:
                UserDeleteNode(tracks, int(rng.choice(nodes)))
            elif choice == 3 and len(nodes) >= 2:
                a, b = (int(n) for n in rng.choice(nodes, 2, replace=False))
                if tracks.get_time(a) > tracks.get_time(b):
                    a, b = b, a
                if tracks.get_time(a) < tracks.get_time(b):
                    UserAddEdge(tracks, (a, b), force=bool(rng.integers(0, 2)))
            elif choice == 4 and edges:
                UserDeleteEdge(tracks, edges[rng.integers(0, len(edges))])
            elif choice == 5 and len(nodes) >= 2:
                selection = [int(n) for n in rng.choice(nodes, 2, replace=False)]
                UserConnectNodes(tracks, selection)
            elif choice == 6 and len(nodes) >= 2:
                selection = [int(n) for n in rng.choice(nodes, 2, replace=False)]
                UserDisconnectNodes(tracks, selection)
            elif choice == 7 and len(nodes) >= 3:
                trio = tuple(int(n) for n in rng.choice(nodes, 3, replace=False))
                UserSetDivision(tracks, trio)
            elif choice == 8 and len(nodes) >= 2:
                pair = tuple(int(n) for n in rng.choice(nodes, 2, replace=False))
                UserSwapPredecessors(tracks, pair)
            elif choice == 9:
                if "group" in tracks.features:
                    UserDeleteFeature(tracks, "group")
                else:
                    UserAddFeature(tracks, "group", group_feature())
            elif choice == 10 and "group" in tracks.features and nodes:
                members = [int(n) for n in rng.choice(nodes, min(2, len(nodes)))]
                values = [bool(v) for v in rng.integers(0, 2, len(members))]
                UserUpdateNodesAttrs(tracks, members, {"group": values})
            elif choice == 11 and not with_seg:
                node = tracks.get_next_node_id()
                attrs = {
                    tracks.features.time_key: int(rng.integers(0, 5)),
                    tracks.features.tracklet_key: int(rng.integers(1, 6)),
                    "pos": [float(v) for v in rng.uniform(0, 100, 2)],
                }
                UserAddNode(tracks, node, attrs, force=bool(rng.integers(0, 2)))
            elif choice == 11 and with_seg and len(nodes) >= 2:
                same_time = _pair_in_same_time(tracks, nodes, rng)
                if same_time is not None:
                    UserMergeNodes(tracks, list(same_time))
            elif choice == 12 and with_seg and nodes:
                # Erase part of a node's mask
                node = int(rng.choice(nodes))
                coords = pixels(tracks, node)
                keep = rng.random(len(coords[0])) < 0.5
                erased = tuple(c[keep] for c in coords)
                if len(erased[0]):
                    UserUpdateSegmentation(tracks, 0, [(erased, node)], 1)
            elif choice == 13 and with_seg and len(nodes) >= 2:
                # Paint part of one node with the label of another in the same frame
                same_time = _pair_in_same_time(tracks, nodes, rng)
                if same_time is not None:
                    target, label = same_time
                    coords = pixels(tracks, target)
                    keep = rng.random(len(coords[0])) < 0.5
                    painted = tuple(c[keep] for c in coords)
                    if len(painted[0]):
                        UserUpdateSegmentation(tracks, label, [(painted, target)], 1)
            else:
                continue
            succeeded += 1
        except EXPECTED_ERRORS:
            pass
    return succeeded


def _pair_in_same_time(tracks, nodes, rng) -> tuple[int, int] | None:
    by_time: dict = defaultdict(list)
    for node in nodes:
        by_time[tracks.get_time(node)].append(int(node))
    candidates = [group for group in by_time.values() if len(group) >= 2]
    if not candidates:
        return None
    group = candidates[rng.integers(0, len(candidates))]
    a, b = rng.choice(group, 2, replace=False)
    return int(a), int(b)
