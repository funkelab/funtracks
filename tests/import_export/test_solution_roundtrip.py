"""Regression tests for the solution-flag corruption found via motile_tracker.

Covers:
- AddEdge marks new edges solution=True even when the schema default is wrong.
- geff round-trip preserves the solution schema as Boolean with default True.
"""

import numpy as np
import polars as pl

from funtracks.actions.add_delete_edge import AddEdge
from funtracks.actions.add_delete_node import AddNode
from funtracks.import_export import export_to_geff, import_from_geff


def _roundtrip(tracks, tmp_path, name="rt.geff"):
    out = tmp_path / name
    export_to_geff(tracks, out, overwrite=True)
    return import_from_geff(out / "tracks.geff")


def test_geff_roundtrip_preserves_solution_schema(get_tracks, tmp_path):
    tracks = get_tracks(ndim=3, with_seg=True, is_solution=True)
    loaded = _roundtrip(tracks, tmp_path)

    edge_schema = loaded.graph_solution._edge_attr_schemas()["solution"]
    node_schema = loaded.graph_solution._node_attr_schemas()["solution"]

    assert edge_schema.dtype == pl.Boolean
    assert edge_schema.default_value is True
    assert node_schema.dtype == pl.Boolean
    assert node_schema.default_value is True


def test_add_edge_is_solution_true_after_geff_roundtrip(get_tracks, tmp_path):
    tracks = get_tracks(ndim=3, with_seg=True, is_solution=True)
    loaded = _roundtrip(tracks, tmp_path)
    g = loaded.graph_solution

    # find any source at frame t and target at t+1 with no edge between them
    rows = list(g.node_attrs(attr_keys=["node_id", "t"]).sort("t").iter_rows(named=True))
    by_t: dict[int, list[int]] = {}
    for r in rows:
        by_t.setdefault(r["t"], []).append(r["node_id"])
    src = tgt = None
    for t in sorted(by_t):
        if t + 1 in by_t:
            for s in by_t[t]:
                for d in by_t[t + 1]:
                    if not g.has_edge(s, d):
                        src, tgt = s, d
                        break
                if src is not None:
                    break
        if src is not None:
            break
    assert src is not None, "fixture has no addable edge"

    AddEdge(loaded, (src, tgt))
    assert loaded.get_edge_attr((src, tgt), "solution") is True


def test_add_node_is_solution_true_after_geff_roundtrip(get_tracks, tmp_path):
    tracks = get_tracks(ndim=3, with_seg=False, is_solution=True)
    loaded = _roundtrip(tracks, tmp_path)

    new_id = max(loaded.graph_solution.node_ids()) + 1
    AddNode(
        loaded,
        new_id,
        {
            "t": 0,
            "track_id": 999,
            "lineage_id": 999,
            "pos": np.array([1.0, 2.0]),
        },
    )
    assert loaded.get_node_attr(new_id, "solution") is True
