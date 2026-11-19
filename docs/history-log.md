# History log

A `HistoryRecorder` writes every edit of a `Tracks` to a [JSON Lines](https://jsonlines.org)
file as it happens. The log can be replayed onto the graph it started from, to recover
edits after a crash or to reproduce a result. It also records who edited and when.

```python
from funtracks.history import HistoryRecorder, replay

recorder = HistoryRecorder(
    tracks,
    "embryo.geff.history/session.jsonl",
    annotator="alice",
    source={"kind": "geff", "path": "embryo.geff"},
)
...  # edit through user actions, undo, redo
recorder.close()

replay(fresh_tracks, "embryo.geff.history/session.jsonl")
```

## What is recorded

- Every user action, with the basic actions it applied (feature create/delete included).
- Undo and redo, with the basic actions that ran.
- Annotator changes and save markers (written by the application).

Not recorded: track ids, turning computed features on or off, scale, and UI state.

## File semantics

- One file is one session on one **baseline**: the graph state when the recorder was
  created. Replaying the file onto its baseline gives the state at the end of the session.
- The file is only created at the first edit: a session without edits leaves no file.
- After a save, the application starts a new file whose baseline is the saved state; its
  `previous_log` points to the old file.
- A file without `session_end` means the session crashed. A truncated last line (a crash
  mid-write) is ignored when reading.

## Lines

Every line has `v` (format version, `1`), `t` (wall-clock time, ISO 8601), `mono`
(seconds since the session started, unaffected by clock changes) and `type`:

| `type` | Replayed | Fields |
|---|---|---|
| `session_start` | baseline check | `annotator`, `versions`, `source`, `baseline`, `previous_log`, `extra` |
| `action` | yes | `seq`, `name` (user action), `params` (summary for humans), `basic` |
| `partial` | yes | as `action`, plus `error` and `rolled_back`; written when changes were made outside a completed action, which points at a bug |
| `undo` / `redo` | yes | `seq`, `of` (the `seq` undone or redone, `null` if from before this log), `basic` |
| `annotator` | no | `annotator` |
| `save` | no | `path`, `next_log` |
| `session_end` | no | — |

Replay applies the `basic` lists in order and ignores `params`.

### Basic actions

Each entry of `basic` is `{"a": "<class name>", ...}` with the action's arguments:
`AddNode` (`node`, `attributes`), `DeleteNode` (`node`), `AddEdge` (`edge`,
`attributes`), `DeleteEdge` (`edge`), `UpdateNodeAttrs` (`node`, `attrs`),
`UpdateNodeSeg` (`node`, `mask`, `added`, `mask_key`), `AddFeature` (`key`, `feature`,
`values`) and `DeleteFeature` (`key`).

Features that annotators compute (area, iou, ...) are left out of `attributes`: replay
recomputes them. Values of a custom annotator's features are therefore only reproduced
if the same annotator is registered when replaying.

### Track ids

Track ids are never logged. Replay recomputes them from the edges at the end, so the
result has the same tracks as the session, possibly with different id numbers.

### Values

numpy arrays are tagged so they decode with their dtype: `{"__a__": [...], "dtype": ...}`
for short 1-D arrays, `{"__and__": {...}}` (compressed bytes) otherwise. Masks are written
as `{"__mask__": {"bbox", "shape", "bits"}}`.

### Baseline and versions

`baseline` holds `n_nodes`, `n_edges`, `max_node_id` and a SHA-256 `digest` of the
solution's node ids and edges, which replay checks, plus `n_nodes_full` / `n_edges_full`
for information. `versions` records each package's version and, for editable or git
installs, the exact commit and whether the checkout had uncommitted changes.

## Command line

```bash
# Recover edits: replay logs onto the baseline geff and save the result
funtracks history replay LOG [LOG ...] --base BASE.geff --out OUT.geff [--follow] [--until SEQ] [--force]

# Who did what, and how long they were active
funtracks history summary LOG [LOG ...] [--idle-gap 120] [--csv OUT.csv]

# Check that a log reproduces a saved result
funtracks history verify LOG --base BASE.geff --against EXPECTED.geff
```

`--follow` also replays the earlier logs of the chain. Logs recorded on a database (`.db`)
cannot be replayed yet: the database is edited in place, so they are for auditing only.
