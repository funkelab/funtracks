import networkx as nx
import numpy as np
import pytest
import tracksdata as td
from skimage.draw import disk

from funtracks.data_model import EdgeAttr, NodeAttr
from funtracks.data_model.utils import convert_nx_to_td_indexedrxgraph


@pytest.fixture
def segmentation_2d():
    frame_shape = (100, 100)
    total_shape = (5, *frame_shape)
    segmentation = np.zeros(total_shape, dtype="int32")
    # make frame with one cell in center with label 1
    rr, cc = disk(center=(50, 50), radius=20, shape=(100, 100))
    segmentation[0][rr, cc] = 1

    # make frame with two cells
    # first cell centered at (20, 80) with label 2
    # second cell centered at (60, 45) with label 3
    rr, cc = disk(center=(20, 80), radius=10, shape=frame_shape)
    segmentation[1][rr, cc] = 2
    rr, cc = disk(center=(60, 45), radius=15, shape=frame_shape)
    segmentation[1][rr, cc] = 3

    # continue track 3 with squares from 0 to 4 in x and y with label 3
    segmentation[2, 0:4, 0:4] = 4
    segmentation[4, 0:4, 0:4] = 5

    # unconnected node
    segmentation[4, 96:100, 96:100] = 6

    return segmentation


@pytest.fixture
def graph_2d():
    graph_nx = nx.DiGraph()
    nodes = [
        (
            1,
            {
                NodeAttr.POS.value: [50, 50],
                NodeAttr.TIME.value: 0,
                NodeAttr.AREA.value: 1245,
                NodeAttr.TRACK_ID.value: 1,
                td.DEFAULT_ATTR_KEYS.SOLUTION: 1,
            },
        ),
        (
            2,
            {
                NodeAttr.POS.value: [20, 80],
                NodeAttr.TIME.value: 1,
                NodeAttr.TRACK_ID.value: 2,
                NodeAttr.AREA.value: 305,
                td.DEFAULT_ATTR_KEYS.SOLUTION: 1,
            },
        ),
        (
            3,
            {
                NodeAttr.POS.value: [60, 45],
                NodeAttr.TIME.value: 1,
                NodeAttr.AREA.value: 697,
                NodeAttr.TRACK_ID.value: 3,
                td.DEFAULT_ATTR_KEYS.SOLUTION: 1,
            },
        ),
        (
            4,
            {
                NodeAttr.POS.value: [1.5, 1.5],
                NodeAttr.TIME.value: 2,
                NodeAttr.AREA.value: 16,
                NodeAttr.TRACK_ID.value: 3,
                td.DEFAULT_ATTR_KEYS.SOLUTION: 1,
            },
        ),
        (
            5,
            {
                NodeAttr.POS.value: [1.5, 1.5],
                NodeAttr.TIME.value: 4,
                NodeAttr.AREA.value: 16,
                NodeAttr.TRACK_ID.value: 3,
                td.DEFAULT_ATTR_KEYS.SOLUTION: 1,
            },
        ),
        # unconnected node
        (
            6,
            {
                NodeAttr.POS.value: [97.5, 97.5],
                NodeAttr.TIME.value: 4,
                NodeAttr.AREA.value: 16,
                NodeAttr.TRACK_ID.value: 5,
                td.DEFAULT_ATTR_KEYS.SOLUTION: 1,
            },
        ),
    ]
    edges = [
        (1, 2, {EdgeAttr.IOU.value: 0.0, td.DEFAULT_ATTR_KEYS.SOLUTION: 1}),
        (1, 3, {EdgeAttr.IOU.value: 0.39311, td.DEFAULT_ATTR_KEYS.SOLUTION: 1}),
        (3, 4, {EdgeAttr.IOU.value: 0.0, td.DEFAULT_ATTR_KEYS.SOLUTION: 1}),
        (4, 5, {EdgeAttr.IOU.value: 1.0, td.DEFAULT_ATTR_KEYS.SOLUTION: 1}),
    ]
    graph_nx.add_nodes_from(nodes)
    graph_nx.add_edges_from(edges)

    graph_td = convert_nx_to_td_indexedrxgraph(graph_nx)

    return graph_td


@pytest.fixture
def graph_2d_list():
    graph_nx = nx.DiGraph()
    nodes = [
        (
            1,
            {
                "y": 100,
                "x": 50,
                NodeAttr.TIME.value: 0,
                NodeAttr.AREA.value: 1245,
                NodeAttr.TRACK_ID.value: 1,
            },
        ),
        (
            2,
            {
                "y": 20,
                "x": 100,
                NodeAttr.TIME.value: 1,
                NodeAttr.AREA.value: 500,
                NodeAttr.TRACK_ID.value: 2,
            },
        ),
    ]
    graph_nx.add_nodes_from(nodes)
    graph_td = convert_nx_to_td_indexedrxgraph(graph_nx)
    return graph_td


def sphere(center, radius, shape):
    assert len(center) == len(shape)
    indices = np.moveaxis(np.indices(shape), 0, -1)  # last dim is the index
    distance = np.linalg.norm(np.subtract(indices, np.asarray(center)), axis=-1)
    mask = distance <= radius
    return mask


@pytest.fixture
def segmentation_3d():
    frame_shape = (100, 100, 100)
    total_shape = (2, *frame_shape)
    segmentation = np.zeros(total_shape, dtype="int32")
    # make frame with one cell in center with label 1
    mask = sphere(center=(50, 50, 50), radius=20, shape=frame_shape)
    segmentation[0][mask] = 1

    # make frame with two cells
    # first cell centered at (20, 50, 80) with label 2
    # second cell centered at (60, 50, 45) with label 3
    mask = sphere(center=(20, 50, 80), radius=10, shape=frame_shape)
    segmentation[1][mask] = 2
    mask = sphere(center=(60, 50, 45), radius=15, shape=frame_shape)
    segmentation[1][mask] = 3

    return segmentation


@pytest.fixture
def graph_3d():
    graph_nx = nx.DiGraph()
    nodes = [
        (
            1,
            {
                NodeAttr.POS.value: [50, 50, 50],
                NodeAttr.TIME.value: 0,
                NodeAttr.TRACK_ID.value: 1,
                td.DEFAULT_ATTR_KEYS.SOLUTION: 1,
            },
        ),
        (
            2,
            {
                NodeAttr.POS.value: [20, 50, 80],
                NodeAttr.TIME.value: 1,
                NodeAttr.TRACK_ID.value: 1,
                td.DEFAULT_ATTR_KEYS.SOLUTION: 1,
            },
        ),
        (
            3,
            {
                NodeAttr.POS.value: [60, 50, 45],
                NodeAttr.TIME.value: 1,
                NodeAttr.TRACK_ID.value: 1,
                td.DEFAULT_ATTR_KEYS.SOLUTION: 1,
            },
        ),
    ]
    edges = [
        (1, 2),
        (1, 3),
    ]
    graph_nx.add_nodes_from(nodes)
    graph_nx.add_edges_from(edges)

    graph_td = convert_nx_to_td_indexedrxgraph(graph_nx)

    return graph_td
