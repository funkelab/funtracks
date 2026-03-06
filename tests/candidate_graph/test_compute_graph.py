from collections import Counter

import numpy as np
import pytest

from funtracks.candidate_graph import (
    compute_graph_from_points_list,
    compute_graph_from_seg,
)
from funtracks.data_model.graph_attributes import EdgeAttr, NodeAttr


def test_graph_from_segmentation_2d(segmentation_2d, graph_2d):
    # test with 2D segmentation
    cand_graph = compute_graph_from_seg(
        segmentation=segmentation_2d,
        max_edge_distance=100,
        iou=True,
    )
    assert Counter(list(cand_graph.nodes)) == Counter(list(graph_2d.nodes))
    assert Counter(list(cand_graph.edges)) == Counter(list(graph_2d.edges))
    for node in cand_graph.nodes:
        assert Counter(cand_graph.nodes[node]) == Counter(graph_2d.nodes[node])
    for edge in cand_graph.edges:
        assert (
            pytest.approx(cand_graph.edges[edge][EdgeAttr.IOU.value], abs=0.01)
            == graph_2d.edges[edge][EdgeAttr.IOU.value]
        )

    # lower edge distance
    cand_graph = compute_graph_from_seg(
        segmentation=segmentation_2d,
        max_edge_distance=15,
    )
    assert Counter(list(cand_graph.nodes)) == Counter([1, 2, 3])
    assert Counter(list(cand_graph.edges)) == Counter([(1, 3)])


def test_graph_from_segmentation_3d(segmentation_3d, graph_3d):
    # test with 3D segmentation
    cand_graph = compute_graph_from_seg(
        segmentation=segmentation_3d,
        max_edge_distance=100,
    )
    assert Counter(list(cand_graph.nodes)) == Counter(list(graph_3d.nodes))
    assert Counter(list(cand_graph.edges)) == Counter(list(graph_3d.edges))
    for node in cand_graph.nodes:
        assert Counter(cand_graph.nodes[node]) == Counter(graph_3d.nodes[node])
    for edge in cand_graph.edges:
        assert pytest.approx(cand_graph.edges[edge], abs=0.01) == graph_3d.edges[edge]


def test_graph_from_segmentation_with_duplicate_nodes(segmentation_2d):
    segmentation_2d[1][20:25, 20:25] = 1  # add a duplicate label to another time point
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
    assert cand_graph.number_of_edges() == 3
    assert len(cand_graph.in_edges(3)) == 0

    # test scale
    cand_graph = compute_graph_from_points_list(
        points_list, max_edge_distance=3, scale=[1, 1, 1, 5]
    )
    assert cand_graph.number_of_edges() == 0
    assert len(cand_graph.in_edges(3)) == 0
    assert cand_graph.nodes[0][NodeAttr.POS.value] == [1, 1, 5]
    assert cand_graph.nodes[0][NodeAttr.TIME.value] == 0
