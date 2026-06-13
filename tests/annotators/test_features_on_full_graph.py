"""Step 5 regression: detection features (regionprops, iou) live on graph_full.

These features are intrinsic to a detection/link and must stay computed for *all*
nodes/edges — including soft-deleted (solution=False) candidates — so the full and
solution graphs never drift and candidates are ready for re-solving. Track-id features
remain solution-only and are covered elsewhere.
"""

import pytest

from funtracks.actions import DeleteEdge, DeleteNode
from funtracks.annotators import EdgeAnnotator, RegionpropsAnnotator


def _annotator(tracks, cls):
    return next(ann for ann in tracks.annotators if isinstance(ann, cls))


@pytest.mark.parametrize("ndim", [3, 4])
def test_regionprops_persist_and_recompute_on_soft_deleted_node(get_tracks, ndim):
    tracks = get_tracks(ndim=ndim, with_seg=True, is_solution=True)
    rp_ann = _annotator(tracks, RegionpropsAnnotator)
    tracks.enable_features(["pos", "area"])
    rp_ann.compute(["area"])

    node = 5  # leaf node of the fixture graph
    area_before = tracks.get_node_attr(node, "area")
    assert area_before is not None

    # Soft-delete: leaves the solution view, stays in graph_full as solution=False.
    DeleteNode(tracks, node)
    assert node not in tracks.graph_solution.node_ids()
    assert node in tracks.graph_full.node_ids()
    assert tracks.graph_full.nodes[node]["solution"] is False

    # NEW: the detection feature is still readable (helpers read graph_full, so this no
    # longer KeyErrors on an out-of-solution node) and its value is preserved.
    assert tracks.get_node_attr(node, "area") == area_before

    # NEW: a bulk recompute covers the soft-deleted candidate (iterates graph_full).
    tracks._set_node_attr(node, "area", None)  # wipe to prove recompute reaches it
    rp_ann.compute(["area"])
    assert tracks.get_node_attr(node, "area") == area_before


@pytest.mark.parametrize("ndim", [3, 4])
def test_iou_computed_on_soft_deleted_candidate_edge(get_tracks, ndim):
    tracks = get_tracks(ndim=ndim, with_seg=True, is_solution=True)
    edge_ann = _annotator(tracks, EdgeAnnotator)
    tracks.enable_features(["iou"])
    edge_ann.compute(["iou"])

    edge = (4, 5)
    iou_before = tracks.get_edge_attr(edge, "iou")
    assert iou_before is not None

    # Soft-delete the edge: gone from the solution, kept in graph_full as a candidate.
    DeleteEdge(tracks, edge)
    assert not tracks.graph_solution.has_edge(*edge)
    assert tracks.graph_full.has_edge(*edge)

    # NEW: iou is still readable on the candidate edge (get_edge_attr reads graph_full).
    assert tracks.get_edge_attr(edge, "iou") == iou_before

    # NEW: a bulk recompute reaches the solution=False edge (compute iterates graph_full
    # successors, which include candidate edges).
    tracks._set_edge_attr(edge, "iou", 0.0)  # wipe to prove recompute reaches it
    edge_ann.compute(["iou"])
    assert tracks.get_edge_attr(edge, "iou") == iou_before
