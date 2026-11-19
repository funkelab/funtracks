"""Read history logs and apply them to tracks again."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from funtracks.actions.serialization import action_from_dict

from ._fingerprint import baseline_mismatches, fingerprint
from .recorder import LOG_VERSION

if TYPE_CHECKING:
    from funtracks.data_model import Tracks

# Lines whose basic actions are applied on replay
_REPLAYED_TYPES = ("action", "partial", "undo", "redo")


class BaselineMismatchError(Exception):
    """The tracks are not the graph state the history log starts from."""


def read_log(path: Path | str) -> Iterator[dict[str, Any]]:
    """Read the lines of a history log.

    A malformed *last* line is skipped: it is what a crash in the middle of writing
    leaves behind. A malformed line anywhere else raises.

    Args:
        path: The log file.

    Yields:
        dict[str, Any]: One parsed line at a time.

    Raises:
        ValueError: If a line other than the last one is not valid JSON, or a line
            has an unsupported schema version.
    """
    with Path(path).open(encoding="utf-8") as file:
        lines = file.read().splitlines()
    for number, text in enumerate(lines, start=1):
        if not text.strip():
            continue
        try:
            line = json.loads(text)
        except json.JSONDecodeError as error:
            if number == len(lines):
                return
            raise ValueError(f"{path}, line {number}: not valid JSON") from error
        if line.get("v") != LOG_VERSION:
            raise ValueError(
                f"{path}, line {number}: unsupported history log version "
                f"{line.get('v')!r}; this funtracks reads version {LOG_VERSION}"
            )
        yield line


def recompute_track_ids(tracks: Tracks) -> None:
    """Recompute all tracklet and lineage ids from the solution's edges."""
    tracks.track_annotator.compute()


def replay(
    tracks: Tracks,
    log: Path | str | Iterable[dict[str, Any]],
    *,
    until_seq: int | None = None,
    check_baseline: bool = True,
) -> None:
    """Apply the edits of a history log to the tracks.

    The tracks should be the log's baseline: the graph state when the log started.
    The basic actions of every ``action``, ``partial``, ``undo`` and ``redo`` line are
    applied in order; other lines are skipped. Track ids are not logged, so they are
    recomputed from the edges at the end: the result has the same tracks as the
    session, possibly with different id numbers.

    The replayed edits are not added to the undo history. Don't replay onto tracks
    that have a ``HistoryRecorder`` attached: it would log the replayed steps as
    ``partial`` entries.

    Args:
        tracks: The tracks to apply the edits to.
        log: A log file, or its lines (e.g. several logs chained together).
        until_seq: If given, stop after the line with this ``seq``.
        check_baseline: If True, check that the tracks match the baseline of every
            ``session_start`` line before applying what follows it.

    Raises:
        BaselineMismatchError: If ``check_baseline`` and the tracks do not match.
        ValueError: If the log is malformed or has an unknown action.
    """
    lines = read_log(log) if isinstance(log, Path | str) else log
    for line in lines:
        line_type = line.get("type")
        if line_type == "session_start" and check_baseline:
            mismatches = baseline_mismatches(line["baseline"], fingerprint(tracks))
            if mismatches:
                raise BaselineMismatchError(
                    "The tracks are not the baseline of this history log: "
                    + "; ".join(mismatches)
                )
        elif line_type in _REPLAYED_TYPES:
            for basic in line["basic"]:
                action_from_dict(tracks, basic)
            if until_seq is not None and line["seq"] >= until_seq:
                break
    recompute_track_ids(tracks)
    tracks.refresh.emit()
