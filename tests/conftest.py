import networkx as nx
import numpy as np
import pytest
from skimage.draw import disk

from funtracks.data_model import EdgeAttr, NodeAttr


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
    graph = nx.DiGraph()
    nodes = [
        (
            1,
            {
                NodeAttr.POS.value: [50, 50],
                NodeAttr.TIME.value: 0,
                NodeAttr.AREA.value: 1245,
                NodeAttr.TRACK_ID.value: 1,
            },
        ),
        (
            2,
            {
                NodeAttr.POS.value: [20, 80],
                NodeAttr.TIME.value: 1,
                NodeAttr.TRACK_ID.value: 2,
                NodeAttr.AREA.value: 305,
            },
        ),
        (
            3,
            {
                NodeAttr.POS.value: [60, 45],
                NodeAttr.TIME.value: 1,
                NodeAttr.AREA.value: 697,
                NodeAttr.TRACK_ID.value: 3,
            },
        ),
        (
            4,
            {
                NodeAttr.POS.value: [1.5, 1.5],
                NodeAttr.TIME.value: 2,
                NodeAttr.AREA.value: 16,
                NodeAttr.TRACK_ID.value: 3,
            },
        ),
        (
            5,
            {
                NodeAttr.POS.value: [1.5, 1.5],
                NodeAttr.TIME.value: 4,
                NodeAttr.AREA.value: 16,
                NodeAttr.TRACK_ID.value: 3,
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
            },
        ),
    ]
    edges = [
        (1, 2, {EdgeAttr.IOU.value: 0.0}),
        (1, 3, {EdgeAttr.IOU.value: 0.395}),
        (
            3,
            4,
            {EdgeAttr.IOU.value: 0.0},
        ),
        (
            4,
            5,
            {EdgeAttr.IOU.value: 1.0},
        ),
    ]
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    return graph


@pytest.fixture
def graph_2d_list():
    graph = nx.DiGraph()
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
    graph.add_nodes_from(nodes)
    return graph


def sphere(center, radius, shape):
    assert len(center) == len(shape)
    indices = np.moveaxis(np.indices(shape), 0, -1)  # last dim is the index
    distance = np.linalg.norm(np.subtract(indices, np.asarray(center)), axis=-1)
    mask = distance <= radius
    return mask


@pytest.fixture
def segmentation_3d():
    frame_shape = (100, 100, 100)
    total_shape = (5, *frame_shape)
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

    # continue track 3 with squares from 0 to 4 in x and y with label 3
    segmentation[2, 0:4, 0:4, 0:4] = 4
    segmentation[4, 0:4, 0:4, 0:4] = 5

    # unconnected node
    segmentation[4, 96:100, 96:100, 96:100] = 6
    return segmentation


@pytest.fixture
def graph_3d():
    graph = nx.DiGraph()
    nodes = [
        (
            1,
            {
                "pos": [50, 50, 50],
                "time": 0,
                "track_id": 1,
                "selected": True,
            },
        ),
        (
            2,
            {
                "pos": [20, 50, 80],
                "time": 1,
                "track_id": 2,
                "selected": True,
            },
        ),
        (
            3,
            {
                "pos": [60, 50, 45],
                "time": 1,
                "track_id": 3,
                "selected": True,
            },
        ),
        (
            4,
            {
                "pos": [1.5, 1.5, 1.5],
                "time": 2,
                "track_id": 3,
                "selected": True,
            },
        ),
        (
            5,
            {
                "pos": [1.5, 1.5, 1.5],
                "time": 4,
                "track_id": 3,
                "selected": True,
            },
        ),
        # unconnected node
        (
            6,
            {
                "pos": [97.5, 97.5, 97.5],
                "time": 4,
                "track_id": 5,
                "selected": True,
            },
        ),
    ]
    edges = [
        (1, 2, {"distance": 42.426, "iou": 0.0, "selected": True, "span": 1}),
        (1, 3, {"distance": 11.18, "iou": 0.302, "selected": True, "span": 1}),
        (3, 4, {"distance": 87.56, "iou": 0.0, "selected": True, "span": 1}),
        (4, 5, {"distance": 0.0, "iou": 1.0, "selected": True, "span": 2}),
    ]
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    return graph
