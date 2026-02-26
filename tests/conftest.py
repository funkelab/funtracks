from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest
import tracksdata as td
from skimage.draw import disk
from tracksdata.nodes._mask import Mask

from funtracks.utils.tracksdata_utils import (
    create_empty_graphview_graph,
)

if TYPE_CHECKING:
    from typing import Any

    from funtracks.data_model import SolutionTracks, Tracks

# Feature list constants for consistent test usage
# WITH_SEG means segmentation stored as mask/bbox node attributes
FEATURES_WITH_SEG = ["pos", "area", "iou", "mask", "bbox"]
FEATURES_NO_SEG = ["pos"]
SOLUTION_FEATURES_WITH_SEG = ["pos", "area", "iou", "track_id", "mask", "bbox"]
SOLUTION_FEATURES_NO_SEG = ["pos", "track_id"]


def make_2d_disk_mask(center=(50, 50), radius=20) -> Mask:
    """Create a 2D disk mask with bounding box.

    Args:
        center: Center coordinates (y, x)
        radius: Radius of the disk

    Returns:
        tracksdata Mask object with boolean mask and bbox
    """
    radius_actual = radius - 1
    mask_shape = (2 * radius - 1, 2 * radius - 1)
    rr, cc = disk(center=(radius_actual, radius_actual), radius=radius, shape=mask_shape)
    mask_disk = np.zeros(mask_shape, dtype="bool")
    mask_disk[rr, cc] = True
    return Mask(
        mask_disk,
        bbox=np.array(
            [
                center[0] - radius_actual,
                center[1] - radius_actual,
                center[0] + radius_actual + 1,
                center[1] + radius_actual + 1,
            ]
        ),
    )


def make_3d_sphere_mask(center=(50, 50, 50), radius=20) -> Mask:
    """Create a 3D sphere mask with bounding box.

    Args:
        center: Center coordinates (z, y, x)
        radius: Radius of the sphere

    Returns:
        tracksdata Mask object with boolean mask and bbox
    """
    mask_shape = (2 * radius + 1, 2 * radius + 1, 2 * radius + 1)
    mask_sphere = sphere(center=(radius, radius, radius), radius=radius, shape=mask_shape)
    return Mask(
        mask_sphere,
        bbox=np.array(
            [
                center[0] - radius,
                center[1] - radius,
                center[2] - radius,
                center[0] + radius + 1,
                center[1] + radius + 1,
                center[2] + radius + 1,
            ]
        ),
    )


def make_2d_square_mask(start_corner=(0, 0), width=4) -> Mask:
    """Create a 2D square mask with bounding box.

    Args:
        start_corner: Top-left corner coordinates (y, x)
        width: Width and height of the square

    Returns:
        tracksdata Mask object with boolean mask and bbox
    """
    mask_shape = (width, width)
    mask_square = np.ones(mask_shape, dtype="bool")
    return Mask(
        mask_square,
        bbox=np.array(
            [
                start_corner[0],
                start_corner[1],
                start_corner[0] + width,
                start_corner[1] + width,
            ]
        ),
    )


def make_3d_cube_mask(start_corner=(0, 0, 0), width=4) -> Mask:
    """Create a 3D cube mask with bounding box.

    Args:
        start_corner: Corner coordinates (z, y, x)
        width: Width, height, and depth of the cube

    Returns:
        tracksdata Mask object with boolean mask and bbox
    """
    mask_shape = (width, width, width)
    mask_cube = np.ones(mask_shape, dtype="bool")
    return Mask(
        mask_cube,
        bbox=np.array(
            [
                start_corner[0],
                start_corner[1],
                start_corner[2],
                start_corner[0] + width,
                start_corner[1] + width,
                start_corner[2] + width,
            ]
        ),
    )


def _make_graph(
    *,
    ndim: int = 3,
    with_pos: bool = False,
    with_track_id: bool = False,
    with_area: bool = False,
    with_iou: bool = False,
    with_masks: bool = False,
    database: str | None = None,
) -> td.graph.GraphView:
    """Generate a test graph with configurable features.

    Args:
        ndim: 3 for 2D spatial + time, 4 for 3D spatial + time
        with_pos: Include position attribute
        with_track_id: Include track_id attribute
        with_area: Include area attribute (requires with_pos=True)
        with_iou: Include iou edge attribute (requires with_area=True)
        with_masks: Include mask and bbox node attributes
        database: Database path for SQLGraph (if None, uses default)

    Returns:
        A graph with the requested features
    """

    node_attributes = []
    edge_attributes = []
    if with_pos:
        node_attributes.append("pos")
    if with_track_id:
        node_attributes.append("track_id")
        node_attributes.append("lineage_id")
    if with_area:
        node_attributes.append("area")
    if with_iou:
        edge_attributes.append("iou")
    if with_masks:
        node_attributes.append(td.DEFAULT_ATTR_KEYS.MASK)
        node_attributes.append(td.DEFAULT_ATTR_KEYS.BBOX)

    graph = create_empty_graphview_graph(
        node_attributes=node_attributes,
        edge_attributes=edge_attributes,
        database=database,
        position_attrs=["pos"] if with_pos else None,
        ndim=ndim,
    )

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

    # Mask data (matches segmentation structure)
    segmentation_shape: tuple[int, ...]
    if ndim == 3:  # 2D spatial
        masks = {
            1: make_2d_disk_mask(center=(50, 50), radius=20),
            2: make_2d_disk_mask(center=(20, 80), radius=10),
            3: make_2d_disk_mask(center=(60, 45), radius=15),
            4: make_2d_square_mask(start_corner=(0, 0), width=4),
            5: make_2d_square_mask(start_corner=(0, 0), width=4),
            6: make_2d_square_mask(start_corner=(96, 96), width=4),
        }
        segmentation_shape = (5, 100, 100)
    else:  # 3D spatial
        masks = {
            1: make_3d_sphere_mask(center=(50, 50, 50), radius=20),
            2: make_3d_sphere_mask(center=(20, 50, 80), radius=10),
            3: make_3d_sphere_mask(center=(60, 50, 45), radius=15),
            4: make_3d_cube_mask(start_corner=(0, 0, 0), width=4),
            5: make_3d_cube_mask(start_corner=(0, 0, 0), width=4),
            6: make_3d_cube_mask(start_corner=(96, 96, 96), width=4),
        }
        segmentation_shape = (5, 100, 100, 100)
    # Lineage IDs
    lineage_ids = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 2}

    # Build nodes with requested features
    nodes_id_list = []
    nodes_attrs_list = []
    for node_id, attrs in base_nodes:
        node_attrs: dict[str, Any] = dict(attrs)  # Start with time
        node_attrs["solution"] = 1
        if with_pos:
            # TODO: don't hardcode "pos" and other column names
            node_attrs["pos"] = positions[node_id]
        if with_track_id:
            node_attrs["track_id"] = track_ids[node_id]
            node_attrs["lineage_id"] = lineage_ids[node_id]
        if with_area:
            node_attrs["area"] = float(areas[node_id])
            # I think this is necessary, to keep the dtype the same,
            # in case the scale are not integers
        if with_masks:
            mask = masks[node_id]
            node_attrs[td.DEFAULT_ATTR_KEYS.MASK] = mask
            node_attrs[td.DEFAULT_ATTR_KEYS.BBOX] = mask.bbox
        nodes_id_list.append(node_id)
        nodes_attrs_list.append(node_attrs)

    edges = [
        {"source_id": 1, "target_id": 2, "solution": 1},
        {"source_id": 1, "target_id": 3, "solution": 1},
        {"source_id": 3, "target_id": 4, "solution": 1},
        {"source_id": 4, "target_id": 5, "solution": 1},
    ]

    graph.bulk_add_nodes(nodes=nodes_attrs_list, indices=nodes_id_list)
    graph.bulk_add_edges(edges)
    if with_masks:
        graph.update_metadata(segmentation_shape=segmentation_shape)

    # Add IOUs to edges if requested
    if with_iou:
        for edge, iou in ious.items():
            if graph.has_edge(edge[0], edge[1]):
                edge_id = graph.edge_id(edge[0], edge[1])
                graph.update_edge_attrs(attrs={"iou": iou}, edge_ids=[edge_id])

    return graph


@pytest.fixture
def graph_clean(tmp_path) -> td.graph.GraphView:
    """Base graph with only time - no positions or computed features."""
    db_path = str(tmp_path / "graph_clean.db")
    return _make_graph(ndim=3, database=db_path)


@pytest.fixture
def graph_2d_with_position(tmp_path) -> td.graph.GraphView:
    """Graph with 2D positions - for Tracks without segmentation."""
    db_path = str(tmp_path / "graph_2d_position.db")
    return _make_graph(ndim=3, with_pos=True, database=db_path)


@pytest.fixture
def graph_2d_with_track_id(tmp_path) -> td.graph.GraphView:
    """Graph with 2D positions and track_id - for SolutionTracks without segmentation."""
    db_path = str(tmp_path / "graph_2d_track_id.db")
    return _make_graph(ndim=3, with_pos=True, with_track_id=True, database=db_path)


@pytest.fixture
def graph_2d_with_segmentation(tmp_path) -> td.graph.GraphView:
    """Graph with segmentation (masks/bboxes) and all computed features."""
    db_path = str(tmp_path / "graph_2d_segmentation.db")
    return _make_graph(
        ndim=3,
        with_pos=True,
        with_track_id=True,
        with_area=True,
        with_iou=True,
        with_masks=True,
        database=db_path,
    )


@pytest.fixture
def graph_3d_with_position(tmp_path) -> td.graph.GraphView:
    """Graph with 3D positions - for Tracks without segmentation."""
    db_path = str(tmp_path / "graph_3d_position.db")
    return _make_graph(ndim=4, with_pos=True, database=db_path)


@pytest.fixture
def graph_3d_with_track_id(tmp_path) -> td.graph.GraphView:
    """Graph with 3D positions and track_id - for SolutionTracks without segmentation."""
    db_path = str(tmp_path / "graph_3d_track_id.db")
    return _make_graph(ndim=4, with_pos=True, with_track_id=True, database=db_path)


@pytest.fixture
def graph_3d_with_segmentation(tmp_path) -> td.graph.GraphView:
    """Graph with segmentation (masks/bboxes) and all computed features."""
    db_path = str(tmp_path / "graph_3d_segmentation.db")
    return _make_graph(
        ndim=4,
        with_pos=True,
        with_track_id=True,
        with_area=True,
        with_iou=True,
        with_masks=True,
        database=db_path,
    )


@pytest.fixture
def get_tracks(get_graph) -> Callable[..., "Tracks | SolutionTracks"]:
    """Factory fixture to create Tracks or SolutionTracks instances.

    Returns a factory function that can be called with:
        ndim: 3 for 2D spatial + time, 4 for 3D spatial + time
        with_seg: Whether to include segmentation (mask/bbox as node attributes)
        is_solution: Whether to return SolutionTracks instead of Tracks

    Example:
        tracks = get_tracks(ndim=3, with_seg=True, is_solution=True)

    Note:
        Uses a pre-built FeatureDict to avoid recomputing features that already
        exist in the test graph fixtures.
    """
    from funtracks.data_model import SolutionTracks, Tracks
    from funtracks.features import (
        Area,
        FeatureDict,
        IoU,
        LineageID,
        Position,
        Time,
        TrackletID,
    )

    def _make_tracks(
        ndim: int,
        with_seg: bool = True,
        is_solution: bool = False,
    ) -> Tracks | SolutionTracks:
        # Determine axis names based on ndim
        axis_names = ["z", "y", "x"] if ndim == 4 else ["y", "x"]

        # Determine which graph to use based on requirements
        if with_seg:
            # With segmentation: use graph with mask/bbox node attrs
            # and all computed features
            graph = get_graph(ndim=ndim, with_features="segmentation")
        else:
            # Without segmentation
            if is_solution:
                # SolutionTracks needs track_id: use graph with pos + track_id
                graph = get_graph(ndim=ndim, with_features="track_id")
            else:
                # Regular Tracks: use graph with just pos
                graph = get_graph(ndim=ndim, with_features="position")

        # Build FeatureDict based on what exists in the graph
        features_dict = {
            "t": Time(),
            "pos": Position(axes=axis_names),
        }

        if with_seg:
            # Graph has pre-computed features (area, iou, track_id, mask, bbox)
            features_dict["area"] = Area(ndim=ndim)
            features_dict["iou"] = IoU()
            features_dict["track_id"] = TrackletID()
            features_dict["lineage_id"] = LineageID()
        elif is_solution:
            # SolutionTracks without seg: has track_id but not area/iou/mask/bbox
            features_dict["track_id"] = TrackletID()
            features_dict["lineage_id"] = LineageID()

        feature_dict = FeatureDict(
            features=features_dict,
            time_key="t",
            position_key="pos",
            tracklet_key="track_id" if (with_seg or is_solution) else None,
            lineage_key="lineage_id" if is_solution else None,
        )

        # Create the appropriate Tracks type with pre-built FeatureDict
        if is_solution:
            return SolutionTracks(
                graph,
                ndim=ndim,
                features=feature_dict,
            )
        else:
            return Tracks(
                graph,
                ndim=ndim,
                features=feature_dict,
            )

    return _make_tracks


@pytest.fixture
def graph_2d_list(tmp_path) -> td.graph.GraphView:
    db_path = str(tmp_path / "graph_2d_list.db")
    graph = create_empty_graphview_graph(database=db_path)

    nodes = [
        {
            "y": 100,
            "x": 50,
            "t": 0,
            "area": 1245,
            "track_id": 1,
            "lineage_id": 1,
        },
        {
            "y": 20,
            "x": 100,
            "t": 1,
            "area": 500,
            "track_id": 2,
            "lineage_id": 2,
        },
    ]
    graph.add_node_attr_key("y", default_value=0.0, dtype=pl.Float64)
    graph.add_node_attr_key("x", default_value=0.0, dtype=pl.Float64)
    graph.add_node_attr_key("area", default_value=0.0, dtype=pl.Float64)
    graph.add_node_attr_key("track_id", default_value=0.0, dtype=pl.Float64)

    graph.bulk_add_nodes(nodes=nodes, indices=[1, 2])
    return graph


def sphere(center, radius, shape):
    assert len(center) == len(shape)
    indices = np.moveaxis(np.indices(shape), 0, -1)  # last dim is the index
    distance = np.linalg.norm(np.subtract(indices, np.asarray(center)), axis=-1)
    mask = distance <= radius
    return mask


@pytest.fixture
def get_graph(request) -> Callable[..., td.graph.GraphView]:
    """Factory fixture to get graph by ndim and feature level.

    Args:
        ndim: 3 for 2D spatial + time, 4 for 3D spatial + time
        with_features: Feature level to include:
            - "clean": time only
            - "position": time + pos
            - "track_id": time + pos + track_id and lineage_id (for SolutionTracks
                without seg)
            - "segmentation": time + pos + track_id + area + iou + mask + bbox

    Returns:
        A deep copy of the requested graph

    Example:
        graph = get_graph(ndim=3, with_features="segmentation")
    """

    def _get_graph(ndim: int, with_features: str = "clean") -> td.graph.GraphView:
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
        elif with_features == "segmentation":
            if ndim == 3:
                graph = request.getfixturevalue("graph_2d_with_segmentation")
            else:  # ndim == 4
                graph = request.getfixturevalue("graph_3d_with_segmentation")
        else:
            raise ValueError(
                f"with_features must be 'clean', 'position', 'track_id', "
                f"or 'segmentation', got {with_features}"
            )

        # Deepcopy alternative for tracksdata graph
        return graph.detach().filter().subgraph()

    return _get_graph
