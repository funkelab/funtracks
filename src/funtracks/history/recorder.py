"""Write every edit of a Tracks to a JSON Lines history log, as it happens."""

from __future__ import annotations

import functools
import json
import logging
import os
import platform
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any

from funtracks.actions import ActionGroup, BasicAction, UpdateTrackIDs
from funtracks.actions.serialization import encode_value

from ._fingerprint import fingerprint
from ._provenance import package_provenance

if TYPE_CHECKING:
    from funtracks.actions import Action
    from funtracks.data_model import Tracks

logger = logging.getLogger(__name__)

LOG_VERSION = 1
_DEFAULT_PACKAGES = ("funtracks", "tracksdata")


def _guarded(method: Callable[..., None]) -> Callable[..., None]:
    """Never let a recorder error escape into the edit that triggered it.

    psygnal re-raises an exception from a listener in the code that emitted the
    signal, which would abort the user's edit. Instead, warn once, mark the recorder
    as failed, and stop recording.
    """

    @functools.wraps(method)
    def wrapper(self: HistoryRecorder, *args: Any) -> None:
        if self.failed or self._closed:
            return
        try:
            method(self, *args)
        except Exception:  # noqa: BLE001 -- recording must never break an edit
            self._fail()

    return wrapper


class HistoryRecorder:
    """Record the edits of one Tracks to a history log file.

    One file holds one session on one baseline: the graph state when the recorder
    was created, identified by a fingerprint. Replaying the file onto that baseline
    gives the state at the end of the session. The file is only created at the first
    edit, so a session without edits leaves no file. See the history log docs for the
    format.

    Each user action is written as one ``action`` line with the basic actions it
    applied, undo and redo as ``undo`` / ``redo`` lines with the basic actions that
    ran. Steps that no user action, undo or redo claimed (e.g. after an error) are
    written as a ``partial`` line, so that the log holds every change.

    Recording never breaks editing: if anything in the recorder fails, it warns once,
    sets ``failed`` and stops recording.

    Args:
        tracks: The tracks to record.
        path: The log file. Created (with its parent directories) at the first edit.
        annotator: Who is editing, written with every session.
        source: Where the tracks came from, as ``{"kind": "geff" | "sql" | "none",
            "path": str | None}``.
        provenance_packages: Extra distributions whose version and commit to record,
            on top of funtracks and tracksdata (e.g. ``["napari-track-edit"]``).
        previous_log: The log of the session this one continues from, if any.
        extra: Free-form information for the session header.
        digest: Whether to include the SHA-256 digest in the baseline fingerprint.
        fsync: If True, fsync after every line, not just flush.

    Failing to create or write the file never raises: it sets ``failed``.
    """

    def __init__(
        self,
        tracks: Tracks,
        path: Path | str,
        *,
        annotator: str,
        source: dict[str, Any],
        provenance_packages: list[str] | None = None,
        previous_log: Path | str | None = None,
        extra: dict[str, Any] | None = None,
        digest: bool = True,
        fsync: bool = False,
    ):
        self.tracks = tracks
        self.path = Path(path)
        self.failed = False
        self._closed = False
        self._fsync = fsync
        self._start = time.monotonic()
        self._pending: list[dict[str, Any]] = []
        self._seq = 0
        # seq of the line that created each entry of the undo stack, and of the undo
        # line behind each entry of the redo stack, mirroring ActionHistory (GURQ).
        # Entries from before this log started (e.g. after a save, which starts a
        # new log but keeps the undo history) have no seq here: None.
        history = tracks.action_history
        self._seq_by_index: list[int | None] = [None] * len(history.undo_stack)
        self._redo_seqs: list[int | None] = [None] * len(history.redo_stack)

        packages = [*_DEFAULT_PACKAGES, *(provenance_packages or [])]
        versions: dict[str, Any] = {name: package_provenance(name) for name in packages}
        versions["python"] = platform.python_version()

        # Taken now, written with the first edit
        self._file: IO[str] | None = None
        self._session_start = {
            "type": "session_start",
            "annotator": annotator,
            "versions": versions,
            "source": source,
            "baseline": fingerprint(tracks, digest=digest),
            "previous_log": str(previous_log) if previous_log is not None else None,
            "extra": extra or {},
        }
        self._session_start_time = datetime.now().astimezone()
        tracks.action_applied.connect(self._on_action_applied)
        tracks.action_history.history_event.connect(self._on_history_event)
        tracks.user_action_started.connect(self._on_user_action_started)
        tracks.user_action_failed.connect(self._on_user_action_failed)

    # ---- public API ----

    def set_annotator(self, name: str) -> None:
        """Record that a different person continues editing from here on."""
        if self._file is None:
            # No edits yet: the session simply starts with this annotator
            self._session_start["annotator"] = name
        else:
            self._write_guarded({"type": "annotator", "annotator": name})

    def mark_saved(self, path: Path | str, next_log: Path | str) -> None:
        """Record that the tracks were saved, and which log continues the session.

        Nothing is written if there were no edits.
        """
        if self._file is None:
            return
        self._write_guarded(
            {"type": "save", "path": str(path), "next_log": str(next_log)}
        )

    def close(self) -> None:
        """Flush unclaimed steps, write ``session_end`` and stop recording.

        Safe to call more than once.
        """
        if self._closed:
            return
        self._flush_orphans()
        if self._file is not None:
            self._write_guarded({"type": "session_end"})
        self._closed = True
        for signal, slot in (
            (self.tracks.action_applied, self._on_action_applied),
            (self.tracks.action_history.history_event, self._on_history_event),
            (self.tracks.user_action_started, self._on_user_action_started),
            (self.tracks.user_action_failed, self._on_user_action_failed),
        ):
            signal.disconnect(slot, missing_ok=True)
        if self._file is not None:
            try:
                self._file.close()
            except OSError:
                self._fail()

    # ---- signal handlers ----

    @_guarded
    def _on_action_applied(self, action: BasicAction) -> None:
        # Track ids are recomputed from the edges on replay
        if isinstance(action, UpdateTrackIDs):
            return
        # Encode right away: a mask may be changed in place later on
        self._pending.append(action.to_dict())

    @_guarded
    def _on_user_action_started(self, group: ActionGroup) -> None:
        self._flush_orphans()

    @_guarded
    def _on_user_action_failed(
        self, group: ActionGroup, error: BaseException, rolled_back: bool
    ) -> None:
        if not rolled_back:
            # Already committed: its action line was written, nothing to undo
            logger.warning(
                "%s raised after it was added to the undo history: %r",
                type(group).__name__,
                error,
            )
            return
        if not self._pending:
            # Failed before changing anything, e.g. an invalid edit: nothing to log
            return
        self._write_partial(
            name=type(group).__name__,
            params=group.to_summary(),
            error=repr(error),
            rolled_back=True,
        )

    @_guarded
    def _on_history_event(self, kind: str, action: Action, index: int) -> None:
        if kind == "commit":
            seq = self._write_steps(
                "action",
                name=type(action).__name__,
                params=action.to_summary() if isinstance(action, ActionGroup) else {},
            )
            # add_new_action moves the redo stack onto the undo stack first
            self._seq_by_index.extend(self._redo_seqs)
            self._redo_seqs = []
            self._seq_by_index.append(seq)
        elif kind == "undo":
            seq = self._write_steps("undo", of=self._seq_by_index[index])
            self._redo_seqs.append(seq)
        elif kind == "redo":
            self._write_steps("redo", of=self._seq_by_index[index])
            self._redo_seqs.pop()

    # ---- writing ----

    def _write_steps(self, line_type: str, **fields: Any) -> int:
        """Write a line holding the pending steps, and return its seq."""
        if self._file is None:
            self._create_file()
        self._seq += 1
        self._write(
            {"type": line_type, "seq": self._seq, **fields, "basic": self._pending}
        )
        self._pending = []
        return self._seq

    def _write_partial(
        self, name: str | None, params: dict, error: str | None, rolled_back: bool
    ) -> None:
        self._write_steps(
            "partial", name=name, params=params, error=error, rolled_back=rolled_back
        )
        logger.warning(
            "History log %s: %s changed the tracks without completing (error: %s). "
            "This points at a bug; the changes are logged as a partial entry.",
            self.path,
            name or "an edit",
            error,
        )

    @_guarded
    def _flush_orphans(self) -> None:
        """Write steps that no user action, undo or redo claimed as a partial line."""
        if self._pending:
            self._write_partial(name=None, params={}, error=None, rolled_back=False)

    def _write_guarded(self, line: dict[str, Any]) -> None:
        if self.failed or self._closed:
            return
        try:
            self._write(line)
        except Exception:  # noqa: BLE001 -- recording must never break an edit
            self._fail()

    def _create_file(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8")
        self._write(self._session_start, now=self._session_start_time, mono=0.0)

    def _write(
        self, line: dict[str, Any], now: datetime | None = None, mono: float | None = None
    ) -> None:
        assert self._file is not None
        now = now if now is not None else datetime.now().astimezone()
        mono = mono if mono is not None else time.monotonic() - self._start
        header = {
            "v": LOG_VERSION,
            "t": now.isoformat(timespec="milliseconds"),
            "mono": round(mono, 3),
        }
        text = json.dumps(
            {**header, **encode_value(line)}, allow_nan=True, separators=(",", ":")
        )
        self._file.write(text + "\n")
        self._file.flush()
        if self._fsync:
            os.fsync(self._file.fileno())

    def _fail(self) -> None:
        if not self.failed:
            logger.warning(
                "History log %s stopped recording after an error; edits from here on "
                "are not logged.",
                self.path,
                exc_info=True,
            )
        self.failed = True
