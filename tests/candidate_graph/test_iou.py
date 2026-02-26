import numpy as np
import pytest

from funtracks.candidate_graph import add_cand_edges, nodes_from_segmentation
from funtracks.candidate_graph.iou import _compute_ious, add_iou


def test_compute_ious_2d(get_tracks):
    tracks = get_tracks(ndim=3, with_seg=True)
    segmentation_2d = np.asarray(tracks.segmentation)

    ious = _compute_ious(segmentation_2d[0], segmentation_2d[1])
    expected = [(1, 3, 555.46 / 1408.0)]
    for iou, expected_iou in zip(ious, expected, strict=False):
        assert iou == pytest.approx(expected_iou, abs=0.01)

    ious = _compute_ious(segmentation_2d[1], segmentation_2d[1])
    expected = [(2, 2, 1.0), (3, 3, 1.0)]
    for iou, expected_iou in zip(ious, expected, strict=False):
        assert iou == pytest.approx(expected_iou, abs=0.01)


def test_compute_ious_3d(get_tracks):
    tracks = get_tracks(ndim=4, with_seg=True)
    segmentation_3d = np.asarray(tracks.segmentation)

    ious = _compute_ious(segmentation_3d[0], segmentation_3d[1])
    expected = [(1, 3, 0.30)]
    for iou, expected_iou in zip(ious, expected, strict=False):
        assert iou == pytest.approx(expected_iou, abs=0.01)

    ious = _compute_ious(segmentation_3d[1], segmentation_3d[1])
    expected = [(2, 2, 1.0), (3, 3, 1.0)]
    for iou, expected_iou in zip(ious, expected, strict=False):
        assert iou == pytest.approx(expected_iou, abs=0.01)


def test_add_iou_2d(get_tracks):
    tracks = get_tracks(ndim=3, with_seg=True)
    segmentation_2d = np.asarray(tracks.segmentation)

    cand_graph, node_frame_dict = nodes_from_segmentation(segmentation_2d)
    add_cand_edges(cand_graph, max_edge_distance=100, node_frame_dict=node_frame_dict)
    add_iou(cand_graph, segmentation_2d, node_frame_dict=node_frame_dict)

    # For edges shared with tracks.graph, iou must agree
    cand_edges = {tuple(e) for e in cand_graph.edge_list()}
    ref_edges = {tuple(e) for e in tracks.graph.edge_list()}
    for src, tgt in cand_edges & ref_edges:
        cand_iou = cand_graph.edges[cand_graph.edge_id(src, tgt)]["iou"]
        ref_iou = tracks.graph.edges[tracks.graph.edge_id(src, tgt)]["iou"]
        assert cand_iou == pytest.approx(ref_iou, abs=0.01)
