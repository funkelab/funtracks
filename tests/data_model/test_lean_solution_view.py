"""The lean ``graph_solution``: built without the blob columns nothing reads from it
(currently ``mask``), which are read through to ``graph_full`` on demand
(``root_fallback=True``, royerlab/tracksdata#340)."""

import numpy as np
import tracksdata as td

from funtracks.data_model import Tracks
from funtracks.utils.tracksdata_utils import (
    _LEAN_VIEW_EXCLUDED,
    all_node_attr_keys,
    create_empty_graph,
)

track_attrs = {"time_attr": "t", "tracklet_attr": "track_id"}


def test_lean_by_backend(graph_2d_with_segmentation):
    """Auto mode is lean only where it saves something: a rustworkx-backed view shares
    the root's attribute dicts, so leaving a column out of it frees nothing."""
    graph = graph_2d_with_segmentation
    tracks = Tracks(graph, ndim=3, **track_attrs)

    rx_rooted = isinstance(graph, td.graph.RustWorkXGraph)
    assert ("mask" in tracks.graph_solution.node_attr_keys()) is rx_rooted
    # graph_full always holds it, whichever view was built.
    assert "mask" in tracks.graph_full.node_attr_keys()


def test_membership_tests_need_all_node_attr_keys(graph_2d_with_segmentation):
    """The trap a lean view sets: `"mask" in graph_solution.node_attr_keys()` is False
    while reading mask works, so a membership test takes a "no masks" branch silently.

    all_node_attr_keys is the answer to "what can I ask this graph for", and feature
    detection uses it - otherwise a feature stored in an excluded column looks missing
    and gets recomputed.
    """
    tracks = Tracks(
        graph_2d_with_segmentation, ndim=3, **track_attrs, lean_solution_view=True
    )

    assert "mask" not in tracks.graph_solution.node_attr_keys()
    assert "mask" in all_node_attr_keys(tracks.graph_solution)
    assert tracks._check_existing_feature("mask")
    # A key on neither the view nor its root is still absent, and still raises on read.
    assert "nope" not in all_node_attr_keys(tracks.graph_solution)
    assert not tracks._check_existing_feature("nope")


def test_lean_view_reads_masks_from_graph_full(graph_2d_with_segmentation):
    """The claim: the view does not hold the column, but the masks still read."""
    tracks = Tracks(
        graph_2d_with_segmentation, ndim=3, **track_attrs, lean_solution_view=True
    )
    assert "mask" not in tracks.graph_solution.node_attr_keys()

    for node in tracks.graph_solution.node_ids():
        through_view = (
            tracks.graph_solution.filter(node_ids=[node])
            .node_attrs(attr_keys=["mask"])["mask"]
            .item()
        )
        on_root = tracks.graph_full.nodes[node]["mask"]
        assert np.array_equal(through_view.mask, on_root.mask)
        assert np.array_equal(through_view.bbox, on_root.bbox)
        assert np.array_equal(tracks.get_mask(node).mask, on_root.mask)


def test_lean_view_reads_only_the_views_nodes(graph_2d_with_segmentation):
    """A fallback read must not leak the soft-deleted nodes graph_full still has."""
    tracks = Tracks(
        graph_2d_with_segmentation, ndim=3, **track_attrs, lean_solution_view=True
    )
    deleted = int(tracks.graph_solution.node_ids()[0])
    tracks.graph_full.update_node_attrs(attrs={"solution": [False]}, node_ids=[deleted])
    tracks.graph_solution.remove_node_from_view(deleted)

    df = tracks.graph_solution.node_attrs(attr_keys=["node_id", "mask"])
    assert set(df["node_id"]) == set(tracks.graph_solution.node_ids())
    assert deleted not in set(df["node_id"])


def test_lean_view_renders_the_same_segmentation(graph_2d_with_segmentation):
    """The end-to-end assertion: identical arrays with and without a lean view."""
    graph = graph_2d_with_segmentation
    lean = Tracks(graph, ndim=3, **track_attrs, lean_solution_view=True)
    full = Tracks(graph, ndim=3, **track_attrs, lean_solution_view=False)

    assert lean.segmentation is not None and full.segmentation is not None
    for t in range(lean.segmentation.shape[0]):
        np.testing.assert_array_equal(
            np.asarray(lean.segmentation[t]), np.asarray(full.segmentation[t])
        )


def test_lean_view_mask_edit_reads_back(graph_2d_with_segmentation):
    """A mask written through the lean view lands on graph_full, reads back through the
    view, and shows up in the rendered array. There is no local copy to go stale."""
    tracks = Tracks(
        graph_2d_with_segmentation, ndim=3, **track_attrs, lean_solution_view=True
    )
    node = int(tracks.graph_solution.node_ids()[0])
    time = tracks.get_time(node)

    old_mask = tracks.get_mask(node)
    new_array = np.zeros_like(old_mask.mask)
    new_array[0, 0] = True
    tracks.update_mask(node, td.nodes.Mask(new_array, bbox=old_mask.bbox))

    read_back = (
        tracks.graph_solution.filter(node_ids=[node])
        .node_attrs(attr_keys=["mask"])["mask"]
        .item()
    )
    assert np.array_equal(read_back.mask, new_array)
    assert np.array_equal(tracks.graph_full.nodes[node]["mask"].mask, new_array)
    assert (np.asarray(tracks.segmentation[time]) == node).sum() == 1


def test_no_mask_column_is_never_lean():
    """Points-only tracks have nothing to exclude, so the flag stays False even on a
    backend where a lean view would otherwise pay off."""
    graph = create_empty_graph(
        node_attributes=["pos"], node_default_values=[0.0], backend="sql"
    )
    graph.bulk_add_nodes([{"t": 0, "pos": [1.0, 2.0]}, {"t": 1, "pos": [3.0, 4.0]}])
    tracks = Tracks(graph, ndim=3, time_attr="t")

    assert set(tracks.graph_solution.node_attr_keys()) == set(
        tracks.graph_full.node_attr_keys()
    )


def test_features_added_later_join_the_lean_view(graph_2d_with_segmentation):
    """Only the excluded columns stay out. A column added after construction is
    materialised in the live view like any other - they are small scalars."""
    tracks = Tracks(
        graph_2d_with_segmentation, ndim=3, **track_attrs, lean_solution_view=True
    )
    tracks.enable_features(["circularity"])

    assert "circularity" in tracks.graph_solution.node_attr_keys()
    assert _LEAN_VIEW_EXCLUDED.isdisjoint(tracks.graph_solution.node_attr_keys())


def test_nested_subgraph_of_a_lean_view_serves_masks(graph_2d_with_segmentation):
    """A subgraph of a lean view inherits its key list and its fallback, so it neither
    claims the mask column nor serves schema defaults for it.

    Guards the tracksdata side of the pin: before that fix a nested view advertised the
    root's columns while its payloads came from the lean parent, so masks read back as
    null - silently, which is worse than the KeyError the fallback replaced.
    """
    tracks = Tracks(
        graph_2d_with_segmentation, ndim=3, **track_attrs, lean_solution_view=True
    )
    subset = [int(node) for node in tracks.graph_solution.node_ids()[:2]]

    nested = tracks.graph_solution.filter(node_ids=subset).subgraph()

    assert "mask" not in nested.node_attr_keys()
    assert set(nested.node_ids()) == set(subset)
    masks = nested.node_attrs(attr_keys=["node_id", "mask"])
    for node, mask in zip(masks["node_id"], masks["mask"], strict=True):
        assert mask is not None
        assert np.array_equal(mask.mask, tracks.get_mask(node).mask)


def test_private_solution_view_all_attrs(graph_2d_with_segmentation):
    """The export escape hatch: a view that holds every column, for code that copies
    the view's own storage (detach, nested subgraph) rather than querying it."""
    graph = graph_2d_with_segmentation
    tracks = Tracks(graph, ndim=3, **track_attrs, lean_solution_view=True)

    full_view = tracks._solution_view_all_attrs()
    assert "mask" in full_view.node_attr_keys()
    assert set(full_view.node_ids()) == set(tracks.graph_solution.node_ids())
    detached = full_view.detach()
    for node in detached.node_ids():
        assert np.array_equal(
            detached.nodes[node]["mask"].mask, tracks.get_mask(node).mask
        )

    not_lean = Tracks(graph, ndim=3, **track_attrs, lean_solution_view=False)
    assert not_lean._solution_view_all_attrs() is not_lean.graph_solution
