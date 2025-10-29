import copy
from collections.abc import Callable
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
import pytest
from skimage.draw import disk

if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import NDArray

    from funtracks.data_model import SolutionTracks, Tracks

# Feature list constants for consistent test usage
FEATURES_WITH_SEG = ["pos", "area", "iou"]
FEATURES_NO_SEG = ["pos"]
SOLUTION_FEATURES_WITH_SEG = ["pos", "area", "iou", "track_id"]
SOLUTION_FEATURES_NO_SEG = ["pos", "track_id"]


@pytest.fixture
def segmentation_2d() -> "NDArray[np.int32]":
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


def _make_graph(
    *,
    ndim: int = 3,
    with_pos: bool = False,
    with_track_id: bool = False,
    with_area: bool = False,
    with_iou: bool = False,
) -> nx.DiGraph:
    """Generate a test graph with configurable features.

    Args:
        ndim: 3 for 2D spatial + time, 4 for 3D spatial + time
        with_pos: Include position attribute
        with_track_id: Include track_id attribute
        with_area: Include area attribute (requires with_pos=True)
        with_iou: Include iou edge attribute (requires with_area=True)

    Returns:
        A graph with the requested features
    """
    graph = nx.DiGraph()

    # Base node data (always has time)
    base_nodes = [
        (1, {"t": 0}),
        (2, {"t": 1}),
        (3, {"t": 1}),
        (4, {"t": 2}),
        (5, {"t": 4}),
        (6, {"t": 4}),
    ]

    # Position data
    if ndim == 3:  # 2D spatial
        positions = {
            1: [50, 50],
            2: [20, 80],
            3: [60, 45],
            4: [1.5, 1.5],
            5: [1.5, 1.5],
            6: [97.5, 97.5],
        }
        areas = {1: 1245, 2: 305, 3: 697, 4: 16, 5: 16, 6: 16}
        ious = {(1, 2): 0.0, (1, 3): 0.395, (3, 4): 0.0, (4, 5): 1.0}
    else:  # 3D spatial
        positions = {
            1: [50, 50, 50],
            2: [20, 50, 80],
            3: [60, 50, 45],
            4: [1.5, 1.5, 1.5],
            5: [1.5, 1.5, 1.5],
            6: [97.5, 97.5, 97.5],
        }
        areas = {1: 33401, 2: 4169, 3: 14147, 4: 64, 5: 64, 6: 64}
        ious = {(1, 2): 0.0, (1, 3): 0.302, (3, 4): 0.0, (4, 5): 1.0}

    # Track IDs
    track_ids = {1: 1, 2: 2, 3: 3, 4: 3, 5: 3, 6: 5}

    # Build nodes with requested features
    nodes = []
    for node_id, attrs in base_nodes:
        node_attrs: dict[str, Any] = dict(attrs)  # Start with time
        if with_pos:
            node_attrs["pos"] = positions[node_id]
        if with_track_id:
            node_attrs["track_id"] = track_ids[node_id]
        if with_area:
            node_attrs["area"] = areas[node_id]
        nodes.append((node_id, node_attrs))

    edges = [(1, 2), (1, 3), (3, 4), (4, 5)]

    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)

    # Add IOUs to edges if requested
    if with_iou:
        for edge, iou in ious.items():
            if edge in graph.edges:
                graph.edges[edge]["iou"] = iou

    return graph


@pytest.fixture
def graph_clean() -> nx.DiGraph:
    """Base graph with only time - no positions or computed features."""
    return _make_graph(ndim=3)


@pytest.fixture
def graph_2d_with_position() -> nx.DiGraph:
    """Graph with 2D positions - for Tracks without segmentation."""
    return _make_graph(ndim=3, with_pos=True)


@pytest.fixture
def graph_2d_with_track_id() -> nx.DiGraph:
    """Graph with 2D positions and track_id - for SolutionTracks without segmentation."""
    return _make_graph(ndim=3, with_pos=True, with_track_id=True)


@pytest.fixture
def graph_2d_with_computed_features() -> nx.DiGraph:
    """Graph with all computed features - for SolutionTracks with segmentation."""
    return _make_graph(
        ndim=3, with_pos=True, with_track_id=True, with_area=True, with_iou=True
    )


@pytest.fixture
def graph_3d_with_position() -> nx.DiGraph:
    """Graph with 3D positions - for Tracks without segmentation."""
    return _make_graph(ndim=4, with_pos=True)


@pytest.fixture
def graph_3d_with_track_id() -> nx.DiGraph:
    """Graph with 3D positions and track_id - for SolutionTracks without segmentation."""
    return _make_graph(ndim=4, with_pos=True, with_track_id=True)


@pytest.fixture
def graph_3d_with_computed_features() -> nx.DiGraph:
    """Graph with all computed features - for SolutionTracks with segmentation."""
    return _make_graph(
        ndim=4, with_pos=True, with_track_id=True, with_area=True, with_iou=True
    )


@pytest.fixture
def get_tracks(get_graph, get_segmentation) -> Callable[..., "Tracks | SolutionTracks"]:
    """Factory fixture to create Tracks or SolutionTracks instances.

    Returns a factory function that can be called with:
        ndim: 3 for 2D spatial + time, 4 for 3D spatial + time
        with_seg: Whether to include segmentation
        is_solution: Whether to return SolutionTracks instead of Tracks

    Example:
        tracks = get_tracks(ndim=3, with_seg=True, is_solution=True)

    Note:
        Uses a pre-built FeatureDict to avoid recomputing features that already
        exist in the test graph fixtures.
    """
    from funtracks.data_model import SolutionTracks, Tracks
    from funtracks.features import Area, FeatureDict, IoU, Position, Time, TrackletID

    def _make_tracks(
        ndim: int,
        with_seg: bool = True,
        is_solution: bool = False,
    ) -> Tracks | SolutionTracks:
        # Determine axis names based on ndim
        axis_names = ["z", "y", "x"] if ndim == 4 else ["y", "x"]

        # Determine which graph to use based on requirements
        if with_seg:
            # With segmentation: use fully computed features (pos + track_id + area + iou)
            graph = get_graph(ndim=ndim, with_features="computed")
            seg = get_segmentation(ndim=ndim)
        else:
            # Without segmentation
            if is_solution:
                # SolutionTracks needs track_id: use graph with pos + track_id
                graph = get_graph(ndim=ndim, with_features="track_id")
            else:
                # Regular Tracks: use graph with just pos
                graph = get_graph(ndim=ndim, with_features="position")
            seg = None

        # Build FeatureDict based on what exists in the graph
        features_dict = {
            "t": Time(),
            "pos": Position(axes=axis_names),
        }

        if with_seg:
            # Graph has pre-computed features (area, iou, track_id)
            features_dict["area"] = Area(ndim=ndim)
            features_dict["iou"] = IoU()
            features_dict["track_id"] = TrackletID()
        elif is_solution:
            # SolutionTracks without seg: has track_id but not area/iou
            features_dict["track_id"] = TrackletID()

        feature_dict = FeatureDict(
            features=features_dict,
            time_key="t",
            position_key="pos",
            tracklet_key="track_id" if (with_seg or is_solution) else None,
        )

        # Create the appropriate Tracks type with pre-built FeatureDict
        if is_solution:
            return SolutionTracks(
                graph,
                segmentation=seg,
                ndim=ndim,
                features=feature_dict,
            )
        else:
            return Tracks(
                graph,
                segmentation=seg,
                ndim=ndim,
                features=feature_dict,
            )

    return _make_tracks


@pytest.fixture
def graph_2d_list() -> nx.DiGraph:
    graph = nx.DiGraph()
    nodes = [
        (
            1,
            {
                "y": 100,
                "x": 50,
                "t": 0,
                "area": 1245,
                "track_id": 1,
            },
        ),
        (
            2,
            {
                "y": 20,
                "x": 100,
                "t": 1,
                "area": 500,
                "track_id": 2,
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
def segmentation_3d() -> "NDArray[np.int32]":
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
def get_graph(request) -> Callable[..., nx.DiGraph]:
    """Factory fixture to get graph by ndim and feature level.

    Args:
        ndim: 3 for 2D spatial + time, 4 for 3D spatial + time
        with_features: Feature level to include:
            - "clean": time only
            - "position": time + pos
            - "track_id": time + pos + track_id (for SolutionTracks without seg)
            - "computed": time + pos + track_id + area + iou (full features)

    Returns:
        A deep copy of the requested graph

    Example:
        graph = get_graph(ndim=3, with_features="track_id")
    """

    def _get_graph(ndim: int, with_features: str = "clean") -> nx.DiGraph:
        if with_features == "clean":
            graph = request.getfixturevalue("graph_clean")
        elif with_features == "position":
            if ndim == 3:
                graph = request.getfixturevalue("graph_2d_with_position")
            else:  # ndim == 4
                graph = request.getfixturevalue("graph_3d_with_position")
        elif with_features == "track_id":
            if ndim == 3:
                graph = request.getfixturevalue("graph_2d_with_track_id")
            else:  # ndim == 4
                graph = request.getfixturevalue("graph_3d_with_track_id")
        elif with_features == "computed":
            if ndim == 3:
                graph = request.getfixturevalue("graph_2d_with_computed_features")
            else:  # ndim == 4
                graph = request.getfixturevalue("graph_3d_with_computed_features")
        else:
            raise ValueError(
                f"with_features must be 'clean', 'position', 'track_id', or 'computed', "
                f"got {with_features}"
            )

        # Return a deep copy to avoid fixture pollution
        return copy.deepcopy(graph)

    return _get_graph


@pytest.fixture
def get_segmentation(request) -> Callable[..., "NDArray[np.int32]"]:
    """Factory fixture to get segmentation by ndim.

    Args:
        ndim: 3 for 2D spatial + time, 4 for 3D spatial + time

    Returns:
        The segmentation array (not copied since it's not typically modified)

    Example:
        seg = get_segmentation(ndim=3)
    """

    def _get_segmentation(ndim: int) -> "NDArray[np.int32]":
        if ndim == 3:
            return request.getfixturevalue("segmentation_2d")
        else:  # ndim == 4
            return request.getfixturevalue("segmentation_3d")

    return _get_segmentation
