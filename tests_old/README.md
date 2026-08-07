# `tests_old/` — frozen old-API backward-compatibility suite

This is a **verbatim copy of the test suite as it existed on `main` before the
`persistent-graph` rework** (the old `SolutionTracks` / `tracks.graph` /
`is_solution` / `create_empty_graphview_graph` API). Its job is to run that **old API
against the new code**, proving the backward-compatibility shims still work
(`SolutionTracks`, the deprecated `Tracks.graph` property, the `GraphView`-accepting
`Tracks.__init__`, `create_empty_graphview_graph`, `UpdateTrackID`, and the geff
module-location re-exports).

## Rules

- **Frozen. Do not add or update tests here.** All new and changed tests go in
  `tests/`, which is the single source of truth. This folder only ever shrinks
  (skips) or gets deleted.
- Tests that exercised behavior **intentionally removed** in `persistent-graph` are
  marked `@pytest.mark.skip` with a reason (e.g. `in_degree`/`out_degree`,
  `load_v1_tracks(solution=)`, the `SolutionTracks`-only `TrackAnnotator`,
  `from_tracks` graph-identity). A handful of tests had a one-line adaptation where
  only an error-message string or a fixture idiom changed (see `git diff` vs `main`).
- The suite emits many `DeprecationWarning`s by design — that is the proof the old
  paths run. They are silenced *within this folder only* by a
  `pytest_collection_modifyitems` hook in `tests_old/conftest.py`.

## Infra hooks wiring this folder in (revert these when deleting)

- `.github/workflows/test.yml`: pytest runs `... tests tests_old`
- `justfile` `test` target: `... tests/ tests_old/`
- `pyproject.toml` `[tool.ruff] extend-exclude`: `tests_old` (not linted/formatted)
- `.pre-commit-config.yaml` top-level `exclude: ^tests_old/` (no hook touches it)

## Deleting this suite (one commit, when the deprecation layer is removed)

1. `rm -rf tests_old/`
2. Revert the four infra hooks listed above.
3. Remove the `src` deprecation shims: `src/funtracks/data_model/solution_tracks.py`
   (+ its `__init__` export), the `Tracks.graph` property, the `GraphView`
   `Tracks.__init__` path, `create_empty_graphview_graph`, `UpdateTrackID`, and the
   `import_export/{export,import}_from_geff.py` re-export modules.
