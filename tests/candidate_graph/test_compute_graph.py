import numpy as np
import pytest

from funtracks.candidate_graph import (
    compute_graph_from_points_list,
    compute_graph_from_seg,
)


def test_graph_from_segmentation_2d(get_tracks):
    tracks = get_tracks(ndim=3, with_seg=True)
    segmentation_2d = np.asarray(tracks.segmentation)

    cand_graph = compute_graph_from_seg(
        segmentation=segmentation_2d,
        max_edge_distance=100,
        iou=True,
    )

    # Same node IDs as the segmentation labels
    assert set(cand_graph.node_ids()) == set(tracks.graph.node_ids())

    # t, pos, area must match the source graph for every node
    for node in cand_graph.node_ids():
        for key in ["t", "pos", "area"]:
            assert np.array(cand_graph.nodes[node][key]) == pytest.approx(
                np.array(tracks.graph.nodes[node][key]), abs=0.01
            )

    # Only adjacent frames are connected; nodes 5,6 at t=4 are isolated
    # because t=3 has no nodes (add_cand_edges only links frame → frame+1)
    assert sorted(cand_graph.edge_list()) == [[1, 2], [1, 3], [2, 4], [3, 4]]

    # For edges shared with tracks.graph, iou must agree
    cand_edges = {tuple(e) for e in cand_graph.edge_list()}
    ref_edges = {tuple(e) for e in tracks.graph.edge_list()}
    for src, tgt in cand_edges & ref_edges:
        cand_iou = cand_graph.edges[cand_graph.edge_id(src, tgt)]["iou"]
        ref_iou = tracks.graph.edges[tracks.graph.edge_id(src, tgt)]["iou"]
        assert cand_iou == pytest.approx(ref_iou, abs=0.01)

    # lower edge distance: only (1, 3) is within 15 pixels (~11.2), (1, 2) is ~42 away
    cand_graph = compute_graph_from_seg(
        segmentation=segmentation_2d,
        max_edge_distance=15,
    )
    assert set(cand_graph.node_ids()) == set(tracks.graph.node_ids())
    assert sorted(cand_graph.edge_list()) == [[1, 3]]


def test_graph_from_segmentation_3d(get_tracks):
    tracks = get_tracks(ndim=4, with_seg=True)
    segmentation_3d = np.asarray(tracks.segmentation)

    cand_graph = compute_graph_from_seg(
        segmentation=segmentation_3d,
        max_edge_distance=100,
    )

    assert set(cand_graph.node_ids()) == set(tracks.graph.node_ids())

    for node in cand_graph.node_ids():
        for key in ["t", "pos", "area"]:
            assert np.array(cand_graph.nodes[node][key]) == pytest.approx(
                np.array(tracks.graph.nodes[node][key]), abs=0.01
            )

    # Only adjacent frames connected; nodes 5,6 at t=4 isolated (gap at t=3)
    assert sorted(cand_graph.edge_list()) == [[1, 2], [1, 3], [2, 4], [3, 4]]


def test_graph_from_segmentation_with_duplicate_nodes(get_tracks):
    tracks = get_tracks(ndim=3, with_seg=True)
    segmentation_2d = np.asarray(tracks.segmentation).copy()
    segmentation_2d[1][20:25, 20:25] = 1  # duplicate label 1 in frame 1
    with pytest.raises(ValueError, match="Duplicate values found among nodes"):
        compute_graph_from_seg(
            segmentation=segmentation_2d,
            max_edge_distance=100,
        )


def test_graph_from_points_list():
    points_list = np.array(
        [
            # t, z, y, x
            [0, 1, 1, 1],
            [2, 3, 3, 3],
            [1, 2, 2, 2],
            [2, 6, 6, 6],
            [2, 1, 1, 1],
        ]
    )
    cand_graph = compute_graph_from_points_list(points_list, max_edge_distance=3)
    assert cand_graph.num_edges() == 3
    assert len(list(cand_graph.predecessors(3))) == 0

    # test scale
    cand_graph = compute_graph_from_points_list(
        points_list, max_edge_distance=3, scale=[1, 1, 1, 5]
    )
    assert cand_graph.num_edges() == 0
    assert len(list(cand_graph.predecessors(3))) == 0
    assert np.array(cand_graph.nodes[0]["pos"]) == pytest.approx([1, 1, 5])
    assert cand_graph.nodes[0]["t"] == 0
