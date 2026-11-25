"""Replaying a recorded session onto its baseline reproduces the session's result."""

import numpy as np
import pytest

from funtracks.history import HistoryRecorder, read_log, replay
from funtracks.user_actions import (
    UserAddEdge,
    UserAddFeature,
    UserConnectNodes,
    UserDeleteEdge,
    UserDeleteFeature,
    UserDeleteNode,
    UserDisconnectNodes,
    UserMergeNodes,
    UserSetDivision,
    UserSwapPredecessors,
    UserUpdateNodeAttrs,
    UserUpdateNodesAttrs,
    UserUpdateSegmentation,
)

from ._helpers import (
    SOURCE,
    assert_live_track_ids_match_topology,
    assert_replay_matches,
    group_feature,
    pixels,
    random_session,
)

# Fixture graph: nodes 1 (t=0), 2 (t=1), 3 (t=1), 4 (t=2), 5 (t=4), 6 (t=4)
# Edges: 1 -> 2, 1 -> 3, 3 -> 4, 4 -> 5


def record(tracks, path, session):
    recorder = HistoryRecorder(tracks, path, annotator="tester", source=SOURCE)
    session(tracks)
    recorder.close()
    assert not recorder.failed


def scripted_session(tracks):
    with_seg = tracks.segmentation is not None
    # Groups: create, add and remove members
    UserAddFeature(tracks, "group", group_feature())
    UserUpdateNodesAttrs(tracks, [1, 2, 4], {"group": [True, True, True]})
    UserUpdateNodesAttrs(tracks, [2], {"group": [False]})

    # Topology edits
    UserDeleteEdge(tracks, (3, 4))
    UserAddEdge(tracks, (3, 6))
    UserDisconnectNodes(tracks, [4, 5])
    UserConnectNodes(tracks, [4, 5])
    UserSwapPredecessors(tracks, (5, 6))
    UserSetDivision(tracks, (4, 5, 6))
    UserDeleteNode(tracks, 2)

    # GURQ: undo, undo, redo, then a new action
    tracks.undo()
    tracks.undo()
    tracks.redo()
    UserUpdateNodeAttrs(tracks, 3, {"group": True})

    if with_seg:
        # Erase part of node 4, then paint part of 6 with 5's label
        coords = pixels(tracks, 4)
        UserUpdateSegmentation(tracks, 0, [(tuple(c[::2] for c in coords), 4)], 1)
        coords = pixels(tracks, 6)
        UserUpdateSegmentation(tracks, 5, [(tuple(c[::3] for c in coords), 6)], 1)
        UserMergeNodes(tracks, [5, 6])
        tracks.undo()

    # Delete the group, then undo that and the entry before it
    UserDeleteFeature(tracks, "group")
    tracks.undo()
    tracks.undo()


@pytest.mark.parametrize("ndim", [3, 4])
@pytest.mark.parametrize("with_seg", [True, False])
def test_scripted_session_replays(get_tracks, tmp_path, ndim, with_seg):
    session = get_tracks(ndim=ndim, with_seg=with_seg, prefill_track_ids=True)
    log = tmp_path / "log.jsonl"
    record(session, log, scripted_session)
    assert_live_track_ids_match_topology(session)

    replayed = get_tracks(ndim=ndim, with_seg=with_seg, prefill_track_ids=True)
    replay(replayed, log)
    assert_replay_matches(replayed, session)

    types = [line["type"] for line in read_log(log)]
    assert types[0] == "session_start"
    assert types[-1] == "session_end"
    assert {"action", "undo", "redo"} <= set(types)
    assert "partial" not in types


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("with_seg", [True, False])
def test_random_session_replays(get_tracks, tmp_path, seed, with_seg):
    session = get_tracks(ndim=3, with_seg=with_seg, prefill_track_ids=True)
    log = tmp_path / "log.jsonl"
    succeeded = []
    record(session, log, lambda t: succeeded.append(random_session(t, seed, 200)))
    assert succeeded[0] > 20
    assert_live_track_ids_match_topology(session)

    replayed = get_tracks(ndim=3, with_seg=with_seg, prefill_track_ids=True)
    replay(replayed, log)
    assert_replay_matches(replayed, session)


def test_replay_onto_renumbered_baseline(get_tracks, tmp_path):
    session = get_tracks(ndim=3, with_seg=True, prefill_track_ids=True)
    log = tmp_path / "log.jsonl"
    record(session, log, scripted_session)

    replayed = get_tracks(ndim=3, with_seg=True, prefill_track_ids=True)
    # Renumber the baseline's track ids: a permutation, as after a reload
    key = replayed.features.tracklet_key
    nodes = list(replayed.graph_solution.node_ids())
    old_ids = [replayed.get_node_attr(n, key) for n in nodes]
    mapping = {old: 100 + i for i, old in enumerate(sorted(set(old_ids)))}
    replayed._set_nodes_attr(nodes, key, [mapping[i] for i in old_ids])
    replayed.track_annotator.tracklet_id_to_nodes = {}
    for node, old in zip(nodes, old_ids, strict=True):
        replayed.track_annotator.tracklet_id_to_nodes.setdefault(mapping[old], [])
        replayed.track_annotator.tracklet_id_to_nodes[mapping[old]].append(node)
    replayed.track_annotator.max_tracklet_id = max(mapping.values())

    replay(replayed, log)
    assert_replay_matches(replayed, session)


def test_until_seq_stops_early(get_tracks, tmp_path):
    session = get_tracks(ndim=3, with_seg=False, prefill_track_ids=True)
    log = tmp_path / "log.jsonl"

    def two_edits(tracks):
        UserDeleteEdge(tracks, (3, 4))
        UserDeleteEdge(tracks, (4, 5))

    record(session, log, two_edits)
    replayed = get_tracks(ndim=3, with_seg=False, prefill_track_ids=True)
    replay(replayed, log, until_seq=1)
    assert not replayed.graph_solution.has_edge(3, 4)
    assert replayed.graph_solution.has_edge(4, 5)
    assert np.unique([replayed.get_track_id(n) for n in (4, 5)]).size == 1
