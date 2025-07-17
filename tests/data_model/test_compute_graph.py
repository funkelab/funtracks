from collections import Counter

import numpy as np
import pytest
from funtracks.data_model.graph_utils import (
    compute_graph_from_seg,
    compute_graph_from_points_list
)

from funtracks.features.regionprops_annotator import (
    Area,
)

from funtracks.features.node_features import Position, Time
from funtracks.features.feature_set import FeatureSet
from funtracks.data_model.graph_attributes import EdgeAttr, NodeAttr

def test_graph_from_segmentation_2d(segmentation_2d, graph_2d):

    features = FeatureSet(
        time_feature=Time(),
        pos_feature=Position(axes=["t", "y", "x"]),
        extra_features=[Area(3)],
    )
    # test with 2D segmentation
    cand_graph = compute_graph_from_seg(
        segmentation=segmentation_2d,
        features=features,
        max_edge_distance=100,
        iou=True,
    )
    assert Counter(list(cand_graph.nodes)) == Counter(list(graph_2d.nodes))
    assert Counter(list(cand_graph.edges)) == Counter(list(graph_2d.edges))
    for node in cand_graph.nodes:
        assert Counter(cand_graph.nodes[node]) == Counter(graph_2d.nodes[node])
    for edge in cand_graph.edges:
        print(cand_graph.edges[edge])
        assert (
            pytest.approx(cand_graph.edges[edge][EdgeAttr.IOU.value], abs=0.01)
            == graph_2d.edges[edge][EdgeAttr.IOU.value]
        )

    # lower edge distance
    cand_graph = compute_graph_from_seg(
        segmentation=segmentation_2d,
        features=features,
        max_edge_distance=15,
    )
    assert Counter(list(cand_graph.nodes)) == Counter([1, 2, 3])
    assert Counter(list(cand_graph.edges)) == Counter([(1, 3)])


def test_graph_from_segmentation_3d(segmentation_3d, graph_3d):
    # test with 3D segmentation

    features = FeatureSet(
        time_feature=Time(),
        pos_feature=Position(axes=["t", "z", "y", "x"]),
        extra_features=[],
    )
        
    cand_graph = compute_graph_from_seg(
        segmentation=segmentation_3d,
        features=features,
        max_edge_distance=100,
    )

    print('graph nodes', cand_graph.nodes)
    print('gt nodes', graph_3d.nodes)

    assert Counter(list(cand_graph.nodes)) == Counter(list(graph_3d.nodes))
    assert Counter(list(cand_graph.edges)) == Counter(list(graph_3d.edges))
    for node in cand_graph.nodes:
        print(cand_graph.nodes[node], graph_3d.nodes[node])
        assert Counter(cand_graph.nodes[node]) == Counter(graph_3d.nodes[node])
    for edge in cand_graph.edges:
        assert pytest.approx(cand_graph.edges[edge], abs=0.01) == graph_3d.edges[edge]

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
    features = FeatureSet(
        time_feature=Time(),
        pos_feature=Position(axes=["t", "z", "y", "x"]),
        extra_features=[Area(4)],
    )
    cand_graph = compute_graph_from_points_list(points_list, features=features, max_edge_distance=3)
    assert cand_graph.number_of_edges() == 3
    assert len(cand_graph.in_edges(3)) == 0

    # test scale
    cand_graph = compute_graph_from_points_list(
        points_list, features=features, max_edge_distance=3, scale=[1, 1, 1, 5]
    )
    assert cand_graph.number_of_edges() == 0
    assert len(cand_graph.in_edges(3)) == 0
    assert cand_graph.nodes[0][features.position.key] == [1, 1, 5]
    assert cand_graph.nodes[0][features.time.key] == 0
