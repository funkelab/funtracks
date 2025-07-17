from collections import Counter

import networkx as nx
import numpy as np
from funtracks.data_model.graph_attributes import NodeAttr

from funtracks.features.regionprops_annotator import (
    Area,
    Circularity,
    EllipsoidAxes,
    Intensity,
    Perimeter,
    RPFeature,
)

from funtracks.features.node_features import Position, Time

from funtracks.data_model.graph_utils import (
    add_cand_edges,
    nodes_from_segmentation,
    _compute_node_frame_dict,
    nodes_from_points_list,
)

from funtracks.features.feature_set import FeatureSet

# nodes_from_segmentation
def test_nodes_from_segmentation_empty():
    # test with empty segmentation

    features = FeatureSet(
        time_feature=Time(),
        pos_feature=Position(axes=["t", "y", "x"]),
        extra_features=[Area(3)],
    )
    empty_graph, node_frame_dict = nodes_from_segmentation(
        np.zeros((3, 1, 10, 10), dtype="int32"),
        features=features, 
    )
    assert Counter(empty_graph.nodes) == Counter([])
    assert node_frame_dict == {}


def test_nodes_from_segmentation_2d(segmentation_2d):

    features = FeatureSet(
            time_feature=Time(),
            pos_feature=Position(axes=["t", "y", "x"]),
            extra_features=[Area(3)],
        )
    # test with 2D segmentation
    node_graph, node_frame_dict = nodes_from_segmentation(
        segmentation=segmentation_2d,
        features=features
    )
    assert Counter(list(node_graph.nodes)) == Counter([1, 2, 3, 4, 5, 6])
    assert node_graph.nodes[2][features.time.key] == 1
    assert node_graph.nodes[2][Area(3).key] == 305 # TODO can we add attr names to featureset?
    assert node_graph.nodes[2][features.position.key] == (20, 80)

    assert node_frame_dict[0] == [1]
    assert Counter(node_frame_dict[1]) == Counter([2, 3])

    # test with scaling
    node_graph, node_frame_dict = nodes_from_segmentation(
        segmentation=segmentation_2d, features=features, scale=[1, 1, 2]
    )
    assert Counter(list(node_graph.nodes)) == Counter([1, 2, 3, 4, 5, 6])
    assert node_graph.nodes[2][features.time.key] == 1
    assert node_graph.nodes[2][Area(3).key] == 610
    assert node_graph.nodes[2][features.position.key] == (20, 160)

    assert node_frame_dict[0] == [1]
    assert Counter(node_frame_dict[1]) == Counter([2, 3])


def test_nodes_from_segmentation_3d(segmentation_3d):
    # test with 3D segmentation
    features = FeatureSet(
        time_feature=Time(),
        pos_feature=Position(axes=["t", "z", "y", "x"]),
        extra_features=[Area(4)],
    )
    node_graph, node_frame_dict = nodes_from_segmentation(
        segmentation=segmentation_3d,
        features=features
    )
    assert Counter(list(node_graph.nodes)) == Counter([1, 2, 3])
    assert node_graph.nodes[2][features.time.key] == 1
    assert node_graph.nodes[2][Area(4).key] == 4169
    assert node_graph.nodes[2][features.position.key] == (20, 50, 80)

    assert node_frame_dict[0] == [1]
    assert Counter(node_frame_dict[1]) == Counter([2, 3])

    # test with scaling
    node_graph, node_frame_dict = nodes_from_segmentation(
        segmentation=segmentation_3d, features=features, scale=[1, 1, 4.5, 1]
    )
    assert Counter(list(node_graph.nodes)) == Counter([1, 2, 3])
    assert node_graph.nodes[2][features.time.key] == 1
    assert node_graph.nodes[2][Area(4).key] == 4169 * 4.5
    assert node_graph.nodes[2][features.position.key] == (20, 50*4.5, 80)

    assert node_frame_dict[0] == [1]
    assert Counter(node_frame_dict[1]) == Counter([2, 3])


# add_cand_edges
def test_add_cand_edges_2d(graph_2d):
    cand_graph = nx.create_empty_copy(graph_2d)
    add_cand_edges(cand_graph, time_key='time', pos_key="pos", max_edge_distance=50)
    assert Counter(list(cand_graph.edges)) == Counter(list(graph_2d.edges))


def test_add_cand_edges_3d(graph_3d):
    cand_graph = nx.create_empty_copy(graph_3d)
    add_cand_edges(cand_graph, time_key='time', pos_key="pos", max_edge_distance=15)
    graph_3d.remove_edge(1, 2)
    assert Counter(list(cand_graph.edges)) == Counter(list(graph_3d.edges))


def test_compute_node_frame_dict(graph_2d):
    node_frame_dict = _compute_node_frame_dict(graph_2d)
    expected = {
        0: [
            1,
        ],
        1: [2, 3],
    }
    assert node_frame_dict == expected


def test_nodes_from_points_list_2d():
    points_list = np.array(
        [
            [0, 1, 2, 3],
            [2, 3, 4, 5],
            [1, 2, 3, 4],
        ]
    )
    cand_graph, node_frame_dict = nodes_from_points_list(points_list, time_key="time", pos_key="pos")
    assert Counter(list(cand_graph.nodes)) == Counter([0, 1, 2])
    assert cand_graph.nodes[0][NodeAttr.TIME.value] == 0
    assert (cand_graph.nodes[0][NodeAttr.POS.value] == np.array([1, 2, 3])).all()
    assert cand_graph.nodes[1][NodeAttr.TIME.value] == 2
    assert (cand_graph.nodes[1][NodeAttr.POS.value] == np.array([3, 4, 5])).all()
