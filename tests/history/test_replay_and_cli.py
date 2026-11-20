import json

import pytest

from funtracks.cli import compare_tracks, main
from funtracks.history import (
    BaselineMismatchError,
    HistoryRecorder,
    read_log,
    recompute_track_ids,
    replay,
)
from funtracks.import_export import import_from_geff, write_to_geff
from funtracks.user_actions import UserAddEdge, UserDeleteEdge, UserDeleteNode

from ._helpers import SOURCE


def record_edits(tracks, path, previous_log=None):
    recorder = HistoryRecorder(
        tracks,
        path,
        annotator="ann",
        source={"kind": "geff", "path": None},
        previous_log=previous_log,
    )
    UserDeleteEdge(tracks, (3, 4))
    UserAddEdge(tracks, (3, 6))
    recorder.close()


def test_reader_skips_truncated_last_line(get_tracks, tmp_path):
    log = tmp_path / "log.jsonl"
    record_edits(get_tracks(ndim=3, with_seg=False, prefill_track_ids=True), log)
    text = log.read_text()
    log.write_text(text + '{"v": 1, "type": "act')
    assert len(list(read_log(log))) == len(text.splitlines())

    # A malformed line elsewhere is an error
    log.write_text('{"v": 1, "type": "act\n' + text)
    with pytest.raises(ValueError, match="line 1"):
        list(read_log(log))


def test_reader_refuses_unknown_version(tmp_path):
    log = tmp_path / "log.jsonl"
    log.write_text(json.dumps({"v": 99, "type": "session_start"}) + "\n")
    with pytest.raises(ValueError, match="version 99"):
        list(read_log(log))


def test_baseline_mismatch(get_tracks, tmp_path):
    log = tmp_path / "log.jsonl"
    record_edits(get_tracks(ndim=3, with_seg=False, prefill_track_ids=True), log)
    other = get_tracks(ndim=3, with_seg=False, prefill_track_ids=True)
    UserDeleteEdge(other, (1, 2))
    with pytest.raises(BaselineMismatchError, match="n_edges"):
        replay(other, log)
    replay(other, log, check_baseline=False)  # --force


def test_next_session_replays_onto_reloaded_save(get_tracks, tmp_path):
    """A save drops soft-deleted candidates, so the next session's undo of a delete
    revives a node in memory but adds it anew on the reloaded geff."""
    tracks = get_tracks(ndim=3, with_seg=True, prefill_track_ids=True)
    UserDeleteNode(tracks, 4)
    saved = tmp_path / "saved.geff"
    write_to_geff(tracks, saved)

    log = tmp_path / "next.jsonl"
    recorder = HistoryRecorder(tracks, log, annotator="ann", source=SOURCE)
    UserDeleteNode(tracks, 2)
    tracks.undo()  # revives 2: it is still in graph_full
    recorder.close()

    reloaded = import_from_geff(saved)
    replay(reloaded, log)
    recompute_track_ids(tracks)
    # Solutions only: the reloaded geff has no soft-deleted candidates
    assert compare_tracks(reloaded, tracks) == []


def test_cli_replay_verify_summary(get_tracks, tmp_path, capsys):
    base = tmp_path / "base.geff"
    write_to_geff(get_tracks(ndim=3, with_seg=False, prefill_track_ids=True), base)

    # Two chained sessions, as after a save
    session = import_from_geff(base)
    first, second = tmp_path / "first.jsonl", tmp_path / "second.jsonl"
    record_edits(session, first)
    recorder = HistoryRecorder(
        session, second, annotator="ann", source=SOURCE, previous_log=first
    )
    UserDeleteEdge(session, (1, 2))
    recorder.close()

    out = tmp_path / "out.geff"
    assert main(["history", "replay", str(second), "--base", str(base), "--out",
                 str(out), "--follow"]) == 0  # fmt: skip
    # A saved geff has no soft-deleted candidates, so compare the solutions only
    recompute_track_ids(session)
    assert compare_tracks(import_from_geff(out), session) == []

    expected = tmp_path / "expected.geff"
    write_to_geff(session, expected)
    assert main(["history", "verify", str(first), "--base", str(base), "--against",
                 str(expected)]) == 1  # fmt: skip
    assert "differs" in capsys.readouterr().out

    csv_path = tmp_path / "summary.csv"
    assert main(["history", "summary", str(first), str(second), "--csv",
                 str(csv_path)]) == 0  # fmt: skip
    summary = capsys.readouterr().out
    assert "UserDeleteEdge" in summary and "UserAddEdge" in summary
    assert csv_path.read_text().startswith("annotator,session,action")


def test_cli_refuses_sql_logs(get_tracks, tmp_path, capsys):
    base = tmp_path / "base.geff"
    tracks = get_tracks(ndim=3, with_seg=False, prefill_track_ids=True)
    write_to_geff(tracks, base)
    log = tmp_path / "log.jsonl"
    recorder = HistoryRecorder(
        tracks, log, annotator="ann", source={"kind": "sql", "path": "x.db"}
    )
    UserDeleteEdge(tracks, (3, 4))
    recorder.close()
    args = ["history", "replay", str(log), "--base", str(base), "--out", "o.geff"]
    assert main(args) == 1
    assert "auditing only" in capsys.readouterr().err
