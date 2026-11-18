"""The ``funtracks`` command line tool.

Subcommands:

- ``funtracks history replay``: apply history logs to a geff, e.g. to recover edits
  after a crash.
- ``funtracks history summary``: who did what, and how long they spent on it.
- ``funtracks history verify``: check that a log reproduces a saved result.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ``funtracks`` command line tool.

    Args:
        argv: The arguments, without the program name. Defaults to ``sys.argv``.

    Returns:
        int: The exit code.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 2
    return int(args.func(args))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="funtracks")
    commands = parser.add_subparsers(title="commands")
    history = commands.add_parser("history", help="Work with edit history logs.")
    history_commands = history.add_subparsers(title="history commands")

    replay = history_commands.add_parser(
        "replay", help="Apply history logs to a geff and save the result."
    )
    replay.add_argument("logs", nargs="+", type=Path, help="Log files, in order.")
    replay.add_argument("--base", required=True, type=Path, help="The baseline geff.")
    replay.add_argument("--out", required=True, type=Path, help="Where to save.")
    replay.add_argument("--until", type=int, help="Stop after the line with this seq.")
    replay.add_argument(
        "--force", action="store_true", help="Don't check the baseline fingerprint."
    )
    replay.add_argument(
        "--follow",
        action="store_true",
        help="Also replay the earlier logs, following previous_log from the last log.",
    )
    replay.set_defaults(func=_replay_command)

    summary = history_commands.add_parser(
        "summary", help="Summarize edits per annotator, session and action."
    )
    summary.add_argument("logs", nargs="+", type=Path, help="Log files.")
    summary.add_argument(
        "--idle-gap",
        type=float,
        default=120.0,
        help="Gaps between edits longer than this many seconds count as idle "
        "(default: 120).",
    )
    summary.add_argument("--csv", type=Path, help="Also write the summary as CSV.")
    summary.set_defaults(func=_summary_command)

    verify = history_commands.add_parser(
        "verify", help="Check that a log reproduces a saved geff."
    )
    verify.add_argument("log", type=Path, help="The log file.")
    verify.add_argument("--base", required=True, type=Path, help="The baseline geff.")
    verify.add_argument(
        "--against", required=True, type=Path, help="The geff the log should give."
    )
    verify.set_defaults(func=_verify_command)
    return parser


# ---- replay ----


def _replay_command(args: argparse.Namespace) -> int:
    from funtracks.history import BaselineMismatchError
    from funtracks.import_export import write_to_geff

    logs = _follow_chain(args.logs[-1]) if args.follow else list(args.logs)
    try:
        tracks = _replay_logs(logs, args.base, until=args.until, force=args.force)
    except (BaselineMismatchError, _SqlLogError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    write_to_geff(tracks, args.out)
    print(f"Replayed {len(logs)} log(s) onto {args.base}; saved to {args.out}")
    return 0


class _SqlLogError(Exception):
    """A log recorded on a database-backed graph cannot be replayed onto a geff."""


def _replay_logs(
    logs: list[Path], base: Path, until: int | None = None, force: bool = False
) -> Any:
    from funtracks.history import read_log, replay
    from funtracks.import_export import import_from_geff

    lines = []
    for log in logs:
        log_lines = list(read_log(log))
        for line in log_lines:
            if line.get("type") == "session_start" and line["source"]["kind"] == "sql":
                raise _SqlLogError(
                    f"{log} was recorded on a database (.db) that is edited in place, "
                    "so it is for auditing only: replaying it onto the same database "
                    "would apply the edits twice. Replaying database logs is not "
                    "supported yet."
                )
        lines.extend(log_lines)
    tracks = import_from_geff(base)
    replay(tracks, lines, until_seq=until, check_baseline=not force)
    return tracks


def _follow_chain(last_log: Path) -> list[Path]:
    """The logs from the first of the chain up to ``last_log``, via previous_log."""
    from funtracks.history import read_log

    chain = [last_log]
    seen = {last_log.resolve()}
    while True:
        start = next(
            (line for line in read_log(chain[0]) if line["type"] == "session_start"),
            None,
        )
        previous = start.get("previous_log") if start else None
        if previous is None:
            return chain
        path = Path(previous)
        if path.resolve() in seen or not path.exists():
            return chain
        seen.add(path.resolve())
        chain.insert(0, path)


# ---- summary ----

_SUMMARY_COLUMNS = [
    "annotator",
    "session",
    "action",
    "count",
    "undo",
    "redo",
    "partial",
    "first",
    "last",
    "active_seconds",
]


def _summary_command(args: argparse.Namespace) -> int:
    rows = summarize_logs(args.logs, idle_gap=args.idle_gap)
    _print_table(rows)
    if args.csv is not None:
        with args.csv.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=_SUMMARY_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
    return 0


def summarize_logs(logs: Sequence[Path], idle_gap: float = 120.0) -> list[dict]:
    """Summarize history logs per annotator, session and user action.

    Each session gets one row per action name (user-action class), plus a row with
    action ``"*"`` for the whole session: its counts, undo/redo/partial counts, first
    and last timestamps, and active time. Active time is the sum of the gaps between
    consecutive events (by their ``mono`` time) that are shorter than ``idle_gap``.
    Per-action rows attribute each gap to the action that ends it.

    Args:
        logs: The log files. Each is one session.
        idle_gap: Gaps longer than this many seconds count as idle.

    Returns:
        list[dict]: One dict per row, with the keys of the CSV columns.
    """
    from funtracks.history import read_log

    rows = []
    for log in logs:
        annotator = None
        previous_mono: float | None = None
        per_action: dict[str, dict[str, Any]] = defaultdict(_empty_row)
        total = _empty_row()
        for line in read_log(log):
            line_type = line["type"]
            if line_type == "session_start":
                annotator = line.get("annotator")
                previous_mono = line["mono"]
                continue
            if line_type == "annotator":
                annotator = line["annotator"]
                continue
            if line_type not in ("action", "partial", "undo", "redo"):
                continue
            name = line.get("name") or line_type
            gap = line["mono"] - previous_mono if previous_mono is not None else 0.0
            active = gap if gap < idle_gap else 0.0
            previous_mono = line["mono"]
            for row in (per_action[name], total):
                row["annotator"] = annotator
                if line_type in ("undo", "redo", "partial"):
                    row[line_type] += 1
                else:
                    row["count"] += 1
                row["first"] = row["first"] or line["t"]
                row["last"] = line["t"]
                row["active_seconds"] += active
        for name, row in sorted(per_action.items()):
            rows.append({**row, "session": log.name, "action": name})
        rows.append({**total, "session": log.name, "action": "*"})
    for row in rows:
        row["active_seconds"] = round(row["active_seconds"], 1)
    return rows


def _empty_row() -> dict[str, Any]:
    return {
        "annotator": None,
        "count": 0,
        "undo": 0,
        "redo": 0,
        "partial": 0,
        "first": None,
        "last": None,
        "active_seconds": 0.0,
    }


def _print_table(rows: list[dict]) -> None:
    widths = {
        col: max(len(col), *(len(str(row[col])) for row in rows)) if rows else len(col)
        for col in _SUMMARY_COLUMNS
    }
    print("  ".join(col.ljust(widths[col]) for col in _SUMMARY_COLUMNS))
    for row in rows:
        print("  ".join(str(row[col]).ljust(widths[col]) for col in _SUMMARY_COLUMNS))


# ---- verify ----


def _verify_command(args: argparse.Namespace) -> int:
    from funtracks.history import BaselineMismatchError
    from funtracks.import_export import import_from_geff

    try:
        replayed = _replay_logs([args.log], args.base)
    except (BaselineMismatchError, _SqlLogError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    expected = import_from_geff(args.against)
    differences = compare_tracks(replayed, expected)
    if differences:
        print("The replayed log differs from the expected geff:")
        for difference in differences[:20]:
            print(f"  {difference}")
        return 1
    print("The replayed log matches the expected geff.")
    return 0


def compare_tracks(a: Any, b: Any) -> list[str]:
    """Differences between the solutions of two Tracks, as readable messages.

    Compares node and edge sets, every node and edge attribute except track ids,
    and the partition of nodes into tracklets and into lineages (track id numbers
    may differ, the grouping may not).

    Args:
        a: The first Tracks.
        b: The second Tracks.

    Returns:
        list[str]: The differences; empty if the solutions match.
    """
    differences = []
    nodes_a = set(a.graph_solution.node_ids())
    nodes_b = set(b.graph_solution.node_ids())
    if nodes_a != nodes_b:
        differences.append(
            f"nodes only in replay: {sorted(nodes_a - nodes_b)[:10]}, "
            f"only in expected: {sorted(nodes_b - nodes_a)[:10]}"
        )
    edges_a = {tuple(e) for e in a.graph_solution.edge_list()}
    edges_b = {tuple(e) for e in b.graph_solution.edge_list()}
    if edges_a != edges_b:
        differences.append(
            f"edges only in replay: {sorted(edges_a - edges_b)[:10]}, "
            f"only in expected: {sorted(edges_b - edges_a)[:10]}"
        )

    if set(a.features) != set(b.features):
        differences.append(
            f"features only in replay: {sorted(set(a.features) - set(b.features))}, "
            f"only in expected: {sorted(set(b.features) - set(a.features))}"
        )
    track_keys = {a.features.tracklet_key, a.features.lineage_key}
    for key in sorted(set(a.features.node_features) - track_keys):
        if key not in b.features:
            continue
        for node in sorted(nodes_a & nodes_b):
            if not _values_equal(a.get_node_attr(node, key), b.get_node_attr(node, key)):
                differences.append(f"node {node}: {key!r} differs")
    for key in sorted(a.features.edge_features):
        if key not in b.features:
            continue
        for edge in sorted(edges_a & edges_b):
            if not _values_equal(a.get_edge_attr(edge, key), b.get_edge_attr(edge, key)):
                differences.append(f"edge {edge}: {key!r} differs")

    for key in track_keys - {None}:
        if _partition(a, key) != _partition(b, key):
            differences.append(f"the grouping of nodes by {key!r} differs")
    return differences


def _partition(tracks: Any, key: str) -> set[frozenset[int]]:
    groups: dict[Any, set[int]] = defaultdict(set)
    for node in tracks.graph_solution.node_ids():
        groups[tracks.get_node_attr(node, key)].add(int(node))
    return {frozenset(group) for group in groups.values()}


def _values_equal(a: Any, b: Any) -> bool:
    from tracksdata.nodes import Mask

    if isinstance(a, Mask) or isinstance(b, Mask):
        if not (isinstance(a, Mask) and isinstance(b, Mask)):
            return False
        return bool(
            np.array_equal(np.asarray(a.bbox), np.asarray(b.bbox))
            and np.array_equal(a.mask, b.mask)
        )
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (np.isnan(a) and np.isnan(b))
    try:
        return bool(np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True))
    except TypeError:
        return bool(np.array_equal(np.asarray(a), np.asarray(b)))


if __name__ == "__main__":
    sys.exit(main())
