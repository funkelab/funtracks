import copy

import networkx as nx
import numpy as np
import pytest
from skimage.draw import disk

from funtracks.data_model import NodeAttr

# Feature list constants for consistent test usage
FEATURES_WITH_SEG = ["pos", "area", "iou"]
FEATURES_NO_SEG = ["pos"]
SOLUTION_FEATURES_WITH_SEG = ["pos", "area", "iou", "tracklet_id"]
SOLUTION_FEATURES_NO_SEG = ["pos", "tracklet_id"]


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
def graph_clean():
    """Base graph with only time and track_id - no positions or computed features."""
    graph = nx.DiGraph()
    nodes = [
        (1, {"time": 0, "track_id": 1}),
        (2, {"time": 1, "track_id": 2}),
        (3, {"time": 1, "track_id": 3}),
        (4, {"time": 2, "track_id": 3}),
        (5, {"time": 4, "track_id": 3}),
        (6, {"time": 4, "track_id": 5}),
    ]
    edges = [(1, 2), (1, 3), (3, 4), (4, 5)]
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    return graph


@pytest.fixture
def graph_2d_with_position(graph_clean):
    """Graph with 2D positions - for tests without segmentation."""
    graph = copy.deepcopy(graph_clean)
    positions = {
        1: [50, 50],
        2: [20, 80],
        3: [60, 45],
        4: [1.5, 1.5],
        5: [1.5, 1.5],
        6: [97.5, 97.5],
    }
    for node_id, pos in positions.items():
        graph.nodes[node_id]["pos"] = pos
    return graph


@pytest.fixture
def graph_2d_with_computed_features(graph_2d_with_position):
    """Graph with 2D positions and computed features - for loaded data tests."""
    graph = copy.deepcopy(graph_2d_with_position)
    areas = {1: 1245, 2: 305, 3: 697, 4: 16, 5: 16, 6: 16}
    for node_id, area in areas.items():
        graph.nodes[node_id]["area"] = area

    ious = {(1, 2): 0.0, (1, 3): 0.395, (3, 4): 0.0, (4, 5): 1.0}
    for edge, iou in ious.items():
        graph.edges[edge]["iou"] = iou
    return graph


@pytest.fixture
def graph_3d_with_position(graph_clean):
    """Graph with 3D positions - for tests without segmentation."""
    graph = copy.deepcopy(graph_clean)
    positions = {
        1: [50, 50, 50],
        2: [20, 50, 80],
        3: [60, 50, 45],
        4: [1.5, 1.5, 1.5],
        5: [1.5, 1.5, 1.5],
        6: [97.5, 97.5, 97.5],
    }
    for node_id, pos in positions.items():
        graph.nodes[node_id]["pos"] = pos
    return graph


@pytest.fixture
def graph_3d_with_computed_features(graph_3d_with_position):
    """Graph with 3D positions and computed features - for loaded data tests."""
    graph = copy.deepcopy(graph_3d_with_position)
    # Add area attributes for 3D segmentation (computed from sphere volumes and cubes)
    areas = {1: 33401, 2: 4169, 3: 14147, 4: 64, 5: 64, 6: 64}
    for node_id, area in areas.items():
        graph.nodes[node_id]["area"] = area

    ious = {(1, 2): 0.0, (1, 3): 0.302, (3, 4): 0.0, (4, 5): 1.0}
    for edge, iou in ious.items():
        graph.edges[edge]["iou"] = iou
    return graph


@pytest.fixture
def get_tracks(request):
    """Factory fixture to create Tracks or SolutionTracks instances.

    Returns a factory function that can be called with:
        ndim: 3 for 2D spatial + time, 4 for 3D spatial + time
        with_seg: Whether to include segmentation
        is_solution: Whether to return SolutionTracks instead of Tracks

    Example:
        tracks = get_tracks(ndim=3, with_seg=True, is_solution=True)

    Note:
        Automatically uses feature constants (FEATURES_WITH_SEG, etc.) to specify
        existing_features based on with_seg and is_solution parameters.
    """
    from funtracks.data_model import SolutionTracks, Tracks

    def _make_tracks(
        ndim: int,
        with_seg: bool = True,
        is_solution: bool = False,
    ):
        # Get the appropriate graph and segmentation fixtures
        if with_seg:
            if ndim == 3:
                graph = request.getfixturevalue("graph_2d_with_computed_features")
                seg = request.getfixturevalue("segmentation_2d")
            else:  # ndim == 4
                graph = request.getfixturevalue("graph_3d_with_computed_features")
                seg = request.getfixturevalue("segmentation_3d")
        else:
            if ndim == 3:
                graph = request.getfixturevalue("graph_2d_with_position")
            else:  # ndim == 4
                graph = request.getfixturevalue("graph_3d_with_position")
            seg = None

        # Use feature constants to determine existing_features
        if is_solution:
            existing_features = (
                SOLUTION_FEATURES_WITH_SEG if with_seg else SOLUTION_FEATURES_NO_SEG
            )
        else:
            existing_features = FEATURES_WITH_SEG if with_seg else FEATURES_NO_SEG

        # Make a deep copy to avoid fixture pollution across tests
        graph = copy.deepcopy(graph)

        # Create the appropriate Tracks type
        if is_solution:
            return SolutionTracks(
                graph, segmentation=seg, ndim=ndim, existing_features=existing_features
            )
        else:
            return Tracks(
                graph, segmentation=seg, ndim=ndim, existing_features=existing_features
            )

    return _make_tracks


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
def get_graph(request):
    """Factory fixture to get graph by ndim and feature level.

    Args:
        ndim: 3 for 2D spatial + time, 4 for 3D spatial + time
        with_features: "clean" (no features), "position" (pos only),
                      "computed" (pos + area + iou)

    Returns:
        A deep copy of the requested graph

    Example:
        graph = get_graph(ndim=3, with_features="clean")
    """

    def _get_graph(ndim: int, with_features: str = "clean"):
        if with_features == "clean":
            graph = request.getfixturevalue("graph_clean")
        elif with_features == "position":
            if ndim == 3:
                graph = request.getfixturevalue("graph_2d_with_position")
            else:  # ndim == 4
                graph = request.getfixturevalue("graph_3d_with_position")
        elif with_features == "computed":
            if ndim == 3:
                graph = request.getfixturevalue("graph_2d_with_computed_features")
            else:  # ndim == 4
                graph = request.getfixturevalue("graph_3d_with_computed_features")
        else:
            raise ValueError(
                f"with_features must be 'clean', 'position', or 'computed', "
                f"got {with_features}"
            )

        # Return a deep copy to avoid fixture pollution
        return copy.deepcopy(graph)

    return _get_graph


@pytest.fixture
def get_segmentation(request):
    """Factory fixture to get segmentation by ndim.

    Args:
        ndim: 3 for 2D spatial + time, 4 for 3D spatial + time

    Returns:
        The segmentation array (not copied since it's not typically modified)

    Example:
        seg = get_segmentation(ndim=3)
    """

    def _get_segmentation(ndim: int):
        if ndim == 3:
            return request.getfixturevalue("segmentation_2d")
        else:  # ndim == 4
            return request.getfixturevalue("segmentation_3d")

    return _get_segmentation
