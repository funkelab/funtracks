import logging

import pytest

from funtracks.actions import DeleteEdge
from funtracks.exceptions import InvalidActionError
from funtracks.history import HistoryRecorder, read_log, replay
from funtracks.user_actions import UserAddEdge, UserDeleteEdge, UserDeleteNode

from ._helpers import SOURCE, assert_replay_matches


def lines_of(path, *types):
    return [line for line in read_log(path) if not types or line["type"] in types]


def test_session_lines(get_tracks, tmp_path):
    tracks = get_tracks(ndim=3, with_seg=False, prefill_track_ids=True)
    log = tmp_path / "sub" / "log.jsonl"
    recorder = HistoryRecorder(
        tracks, log, annotator="ann", source=SOURCE, extra={"note": 1}
    )
    UserDeleteEdge(tracks, (3, 4))
    tracks.undo()
    tracks.redo()
    recorder.set_annotator("other")
    recorder.mark_saved(tmp_path / "out.geff", tmp_path / "next.jsonl")
    recorder.close()
    recorder.close()  # closing twice is fine

    lines = lines_of(log)
    assert [line["type"] for line in lines] == [
        "session_start",
        "action",
        "undo",
        "redo",
        "annotator",
        "save",
        "session_end",
    ]
    start, action, undo, redo = lines[:4]
    assert start["annotator"] == "ann"
    assert start["extra"] == {"note": 1}
    assert set(start["versions"]) >= {"funtracks", "tracksdata", "python"}
    assert action["name"] == "UserDeleteEdge"
    assert action["params"] == {"edge": [3, 4]}
    assert action["basic"] == [{"a": "DeleteEdge", "edge": [3, 4]}]
    assert (undo["of"], redo["of"]) == (action["seq"], action["seq"])
    assert all(line["v"] == 1 for line in lines)


def test_gurq_of(get_tracks, tmp_path):
    """After undo, undo, redo and a new action, undo walks back through the inverses."""
    tracks = get_tracks(ndim=3, with_seg=False, prefill_track_ids=True)
    log = tmp_path / "log.jsonl"
    recorder = HistoryRecorder(tracks, log, annotator="ann", source=SOURCE)
    UserDeleteEdge(tracks, (3, 4))  # seq 1
    UserDeleteEdge(tracks, (4, 5))  # seq 2
    tracks.undo()  # seq 3, of 2
    tracks.undo()  # seq 4, of 1
    tracks.redo()  # seq 5, of 1
    UserDeleteEdge(tracks, (1, 2))  # seq 6; the undo of seq 2 moves onto the stack
    tracks.undo()  # undoes seq 6
    tracks.undo()  # undoes the inverse written as seq 3
    recorder.close()
    ofs = [line["of"] for line in lines_of(log, "undo", "redo")]
    assert ofs == [2, 1, 1, 6, 3]


def test_failed_action_is_logged_as_partial(get_tracks, tmp_path, monkeypatch, caplog):
    tracks = get_tracks(ndim=3, with_seg=False, prefill_track_ids=True)
    baseline = get_tracks(ndim=3, with_seg=False, prefill_track_ids=True)
    log = tmp_path / "log.jsonl"
    recorder = HistoryRecorder(tracks, log, annotator="ann", source=SOURCE)

    original = DeleteEdge.__init__

    def fail_on_second_edge(self, tracks, edge):
        if edge == (4, 5):
            raise RuntimeError("injected")
        original(self, tracks, edge)

    monkeypatch.setattr(DeleteEdge, "__init__", fail_on_second_edge)
    with caplog.at_level(logging.WARNING), pytest.raises(RuntimeError):
        UserDeleteNode(tracks, 4)  # deletes (3, 4), then fails on (4, 5)
    monkeypatch.setattr(DeleteEdge, "__init__", original)
    recorder.close()

    (partial,) = lines_of(log, "partial")
    assert partial["name"] == "UserDeleteNode"
    assert partial["rolled_back"] is True
    assert "injected" in partial["error"]
    assert [b["a"] for b in partial["basic"]] == ["DeleteEdge", "AddEdge"]
    assert "partial" in caplog.text
    replay(baseline, log)
    assert_replay_matches(baseline, tracks)


def test_invalid_edit_writes_nothing(get_tracks, tmp_path):
    tracks = get_tracks(ndim=3, with_seg=False, prefill_track_ids=True)
    log = tmp_path / "log.jsonl"
    recorder = HistoryRecorder(tracks, log, annotator="ann", source=SOURCE)
    with pytest.raises(InvalidActionError):
        UserAddEdge(tracks, (2, 5))  # 5 already has a parent
    recorder.close()
    assert not log.exists()


def test_file_is_created_at_first_edit(get_tracks, tmp_path):
    tracks = get_tracks(ndim=3, with_seg=False, prefill_track_ids=True)
    log = tmp_path / "log.jsonl"
    recorder = HistoryRecorder(tracks, log, annotator="ann", source=SOURCE)
    recorder.set_annotator("other")  # before any edit: goes into the header
    recorder.mark_saved(tmp_path / "out.geff", tmp_path / "next.jsonl")  # ignored
    assert not log.exists()
    UserDeleteEdge(tracks, (3, 4))
    recorder.close()
    lines = lines_of(log)
    assert [line["type"] for line in lines] == ["session_start", "action", "session_end"]
    assert lines[0]["annotator"] == "other"
    assert lines[0]["mono"] == 0.0


def test_error_after_commit_keeps_action_line(get_tracks, tmp_path):
    tracks = get_tracks(ndim=3, with_seg=False, prefill_track_ids=True)
    log = tmp_path / "log.jsonl"
    recorder = HistoryRecorder(tracks, log, annotator="ann", source=SOURCE)

    def broken_listener(*args):
        raise RuntimeError("listener broke")

    tracks.refresh.connect(broken_listener)
    with pytest.raises(Exception, match="listener broke"):
        UserDeleteEdge(tracks, (3, 4))
    tracks.refresh.disconnect(broken_listener)
    recorder.close()
    assert [line["type"] for line in lines_of(log)] == [
        "session_start",
        "action",
        "session_end",
    ]


def test_unclaimed_steps_are_flushed_before_next_action(get_tracks, tmp_path):
    tracks = get_tracks(ndim=3, with_seg=False, prefill_track_ids=True)
    log = tmp_path / "log.jsonl"
    recorder = HistoryRecorder(tracks, log, annotator="ann", source=SOURCE)
    DeleteEdge(tracks, (1, 2))  # a basic action outside any user action
    UserDeleteEdge(tracks, (3, 4))
    recorder.close()
    partial, action = lines_of(log, "partial", "action")
    assert partial["basic"] == [{"a": "DeleteEdge", "edge": [1, 2]}]
    assert action["basic"] == [{"a": "DeleteEdge", "edge": [3, 4]}]


def test_feature_toggles_are_not_logged(get_tracks, tmp_path):
    tracks = get_tracks(ndim=3, with_seg=True, prefill_track_ids=True)
    log = tmp_path / "log.jsonl"
    recorder = HistoryRecorder(tracks, log, annotator="ann", source=SOURCE)
    tracks.disable_features(["area"])
    tracks.enable_features(["area"])
    recorder.close()
    assert not log.exists()


def test_recorder_failure_does_not_break_editing(get_tracks, tmp_path, caplog):
    tracks = get_tracks(ndim=3, with_seg=False, prefill_track_ids=True)
    # A file where the log's directory should be: creating the log fails
    (tmp_path / "not_a_dir").write_text("")
    recorder = HistoryRecorder(
        tracks, tmp_path / "not_a_dir" / "log.jsonl", annotator="ann", source=SOURCE
    )
    with caplog.at_level(logging.WARNING):
        UserDeleteEdge(tracks, (3, 4))
        UserDeleteEdge(tracks, (4, 5))
    assert not tracks.graph_solution.has_edge(4, 5)
    assert recorder.failed
    assert caplog.text.count("stopped recording") == 1
    recorder.close()
