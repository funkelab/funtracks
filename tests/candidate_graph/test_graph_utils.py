from collections import Counter

import numpy as np

from funtracks.candidate_graph import add_cand_edges, nodes_from_segmentation
from funtracks.candidate_graph.utils import (
    _compute_node_frame_dict,
    nodes_from_points_list,
)


# nodes_from_segmentation
def test_nodes_from_segmentation_empty():
    # test with empty segmentation
    empty_graph, node_frame_dict = nodes_from_segmentation(
        np.zeros((3, 1, 10, 10), dtype="int32")
    )
    assert len(empty_graph.node_ids()) == 0
    assert node_frame_dict == {}


def test_nodes_from_segmentation_2d(get_tracks):
    tracks = get_tracks(ndim=3, with_seg=True)
    segmentation_2d = np.asarray(tracks.segmentation)

    # test with 2D segmentation
    node_graph, node_frame_dict = nodes_from_segmentation(
        segmentation=segmentation_2d,
    )
    assert sorted(node_graph.node_ids()) == [1, 2, 3, 4, 5, 6]
    assert node_graph.nodes[2]["t"] == 1
    assert np.array_equal(node_graph.nodes[2]["pos"], np.array([20, 80]))
    # measurements are not computed here; they are regionprops features on Tracks
    assert "area" not in node_graph.node_attr_keys()

    assert node_frame_dict[0] == [1]
    assert Counter(node_frame_dict[1]) == Counter([2, 3])

    # a scale does not move the positions: they stay in pixel coordinates
    node_graph, node_frame_dict = nodes_from_segmentation(
        segmentation=segmentation_2d, scale=[1, 1, 2]
    )
    assert sorted(node_graph.node_ids()) == [1, 2, 3, 4, 5, 6]
    assert node_graph.nodes[2]["t"] == 1
    assert np.array_equal(node_graph.nodes[2]["pos"], np.array([20, 80]))

    assert node_frame_dict[0] == [1]
    assert Counter(node_frame_dict[1]) == Counter([2, 3])


def test_nodes_from_segmentation_3d(get_tracks):
    tracks = get_tracks(ndim=4, with_seg=True)
    segmentation_3d = np.asarray(tracks.segmentation)

    # test with 3D segmentation
    node_graph, node_frame_dict = nodes_from_segmentation(
        segmentation=segmentation_3d,
    )
    assert sorted(node_graph.node_ids()) == [1, 2, 3, 4, 5, 6]
    assert node_graph.nodes[2]["t"] == 1
    assert np.array_equal(node_graph.nodes[2]["pos"], np.array([20, 50, 80]))
    # measurements are not computed here; they are regionprops features on Tracks
    assert "area" not in node_graph.node_attr_keys()

    assert node_frame_dict[0] == [1]
    assert Counter(node_frame_dict[1]) == Counter([2, 3])

    # a scale does not move the positions: they stay in pixel coordinates
    node_graph, node_frame_dict = nodes_from_segmentation(
        segmentation=segmentation_3d, scale=[1, 1, 4.5, 1]
    )
    assert sorted(node_graph.node_ids()) == [1, 2, 3, 4, 5, 6]
    assert node_graph.nodes[2]["t"] == 1
    assert np.array_equal(node_graph.nodes[2]["pos"], np.array([20.0, 50.0, 80.0]))

    assert node_frame_dict[0] == [1]
    assert Counter(node_frame_dict[1]) == Counter([2, 3])


# add_cand_edges
def test_add_cand_edges_2d(get_tracks):
    tracks = get_tracks(ndim=3, with_seg=True)
    segmentation_2d = np.asarray(tracks.segmentation)
    node_graph, node_frame_dict = nodes_from_segmentation(segmentation_2d)
    add_cand_edges(node_graph, max_edge_distance=50, node_frame_dict=node_frame_dict)
    # dist(1→2) ≈ 42.4, dist(1→3) ≈ 11.2 — both within 50; nodes 4,5,6 too far
    assert sorted(node_graph.edge_list()) == [[1, 2], [1, 3]]


def test_add_cand_edges_3d(get_tracks):
    tracks = get_tracks(ndim=4, with_seg=True)
    segmentation_3d = np.asarray(tracks.segmentation)
    node_graph, node_frame_dict = nodes_from_segmentation(segmentation_3d)
    add_cand_edges(node_graph, max_edge_distance=15, node_frame_dict=node_frame_dict)
    # dist(1→3) ≈ 11.2, dist(1→2) ≈ 42.4 — only (1, 3) within 15
    assert sorted(node_graph.edge_list()) == [[1, 3]]


def test_add_cand_edges_with_scale(get_tracks):
    """max_edge_distance is in world units, so anisotropy changes which edges fit.

    Positions are in pixels: dist(1→3) ≈ 11.2 pixels, made up of a 10 pixel step
    along y and a 5 pixel step along x. Stretching x by 4 makes that ≈ 22.4, which
    no longer fits in a 15 unit budget.
    """
    tracks = get_tracks(ndim=3, with_seg=True)
    segmentation_2d = np.asarray(tracks.segmentation)
    node_graph, node_frame_dict = nodes_from_segmentation(segmentation_2d)

    add_cand_edges(node_graph, max_edge_distance=15, node_frame_dict=node_frame_dict)
    assert sorted(node_graph.edge_list()) == [[1, 3]]

    node_graph, node_frame_dict = nodes_from_segmentation(segmentation_2d)
    add_cand_edges(
        node_graph,
        max_edge_distance=15,
        node_frame_dict=node_frame_dict,
        scale=[1, 1, 4],
    )
    assert sorted(node_graph.edge_list()) == []


def test_compute_node_frame_dict(get_tracks):
    tracks = get_tracks(ndim=3, with_seg=True)
    segmentation_2d = np.asarray(tracks.segmentation)
    node_graph, _ = nodes_from_segmentation(segmentation_2d)
    node_frame_dict = _compute_node_frame_dict(node_graph)
    assert node_frame_dict[0] == [1]
    assert Counter(node_frame_dict[1]) == Counter([2, 3])


def test_nodes_from_points_list_2d():
    points_list = np.array(
        [
            [0, 1, 2, 3],
            [2, 3, 4, 5],
            [1, 2, 3, 4],
        ]
    )
    cand_graph, node_frame_dict = nodes_from_points_list(points_list)
    assert sorted(cand_graph.node_ids()) == [0, 1, 2]
    assert cand_graph.nodes[0]["t"] == 0
    assert np.array_equal(cand_graph.nodes[0]["pos"], np.array([1, 2, 3]))
    assert cand_graph.nodes[1]["t"] == 2
    assert np.array_equal(cand_graph.nodes[1]["pos"], np.array([3, 4, 5]))
