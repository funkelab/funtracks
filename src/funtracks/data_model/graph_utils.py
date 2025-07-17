from typing import Any, Iterable

import dask.array as da
import networkx as nx
import numpy as np
from scipy.spatial import KDTree
from tqdm import tqdm
from funtracks.features.regionprops_extended import regionprops_extended
from funtracks.features.feature_set import FeatureSet
from funtracks.features.regionprops_annotator import RPFeature
from funtracks.data_model.compute_ious import _get_iou_dict
from funtracks.data_model.graph_attributes import EdgeAttr

import networkx as nx
from motile import TrackGraph

def nodes_from_segmentation(
    segmentation: np.ndarray,
    features: FeatureSet,
    intensity_image: np.ndarray | None = None,
    scale: list[float] | None = None,
) -> tuple[nx.DiGraph, dict[int, list[Any]]]:
    """Extract candidate nodes from a segmentation. Returns a networkx graph
    with only nodes, and also a dictionary from frames to node_ids for
    efficient edge adding.

    Each node will have the following attributes (named as in NodeAttrs):
        - time
        - position
        - additional measurement features, which may include:
            - area/volume
            - number of pixels (not scaled by scaling)
            - ellipsoid major/minor/semi-minor axes
            - circularity/sphericity
            - perimeter/surface area
            - intensity (if an intensity image was provided)

    Args:
        segmentation (np.ndarray): A numpy array with integer labels and dimensions
            (t, [z], y, x). Labels must be unique across time, and the label
            will be used as the node id. If the labels are not unique, preprocess
            with motile_toolbox.utils.ensure_unqiue_ids before calling this function.
        features: FeatureSet : A set of node features, including time and position, and 
            optional RPFeatures to compute. 
        intensity_image (np.ndarray): A numpy array from which to compute the region mean intensity measurements.
        scale (list[float] | None, optional): The scale of the segmentation data in all
            dimensions (including time, which should have a dummy 1 value).
            Will be used to rescale the point locations and attribute computations.
            Defaults to None, which implies the data is isotropic.

    Returns:
        tuple[nx.DiGraph, dict[int, list[Any]]]: A candidate graph with only nodes,
            and a mapping from time frames to node ids.
    """

    cand_graph = nx.DiGraph()
    # also construct a dictionary from time frame to node_id for efficiency
    node_frame_dict: dict[int, list[Any]] = {}

    if scale is None:
        scale = [
            1,
        ] * segmentation.ndim
    else:
        assert (
            len(scale) == segmentation.ndim
        ), f"Scale {scale} should have {segmentation.ndim} dims"

    for t in tqdm(range(len(segmentation))):
        segs = segmentation[t]
        nodes_in_frame = []
        if intensity_image is not None:
            if isinstance(intensity_image, da.core.Array):
                props = regionprops_extended(
                    segs,
                    spacing=tuple(scale[1:]),
                    intensity_image=intensity_image[t].compute(),
                )
            else:
                props = regionprops_extended(
                    segs, spacing=tuple(scale[1:]), intensity_image=intensity_image[t]
                )
        else:
            props = regionprops_extended(segs, spacing=tuple(scale[1:]))
        for regionprop in props:
            node_id = regionprop.label
            attrs = {features.time.key: t}
            attrs[features.position.key] = regionprop['centroid']

            regionprop_features = [f for f in features._features if isinstance(f, RPFeature)]
            for feature in regionprop_features:
                attrs[feature.key] = regionprop[feature.regionprops_name]
           
            cand_graph.add_node(node_id, **attrs)
            nodes_in_frame.append(node_id)
        if nodes_in_frame:
            if t not in node_frame_dict:
                node_frame_dict[t] = []
            node_frame_dict[t].extend(nodes_in_frame)
    return cand_graph, node_frame_dict


def nodes_from_points_list(
    points_list: np.ndarray,
    time_key: str,
    pos_key: str,
    scale: list[float] | None = None,
) -> tuple[nx.DiGraph, dict[int, list[Any]]]:
    """Extract candidate nodes from a list of points. Uses the index of the
    point in the list as its unique id.
    Returns a networkx graph with only nodes, and also a dictionary from frames to
    node_ids for efficient edge adding.

    Args:
        points_list (np.ndarray): An NxD numpy array with N points and D
            (3 or 4) dimensions. Dimensions should be in order (t, [z], y, x).
        time_key [str]: key of the time attribute
        pos_key [str]: key of the spatial position attribute      
        scale (list[float] | None, optional): Amount to scale the points in each
            dimension (including time). Only needed if the provided points are in
            "voxel" coordinates instead of world coordinates. Defaults to None, which
            implies the data is isotropic.

    Returns:
        tuple[nx.DiGraph, dict[int, list[Any]]]: A candidate graph with only nodes,
            and a mapping from time frames to node ids.
    """

    cand_graph = nx.DiGraph()
    # also construct a dictionary from time frame to node_id for efficiency
    node_frame_dict: dict[int, list[Any]] = {}

    # scale points
    if scale is not None:
        assert (
            len(scale) == points_list.shape[1]
        ), f"Cannot scale points with {points_list.shape[1]} dims by factor {scale}"
        points_list = points_list * np.array(scale)

    # add points to graph
    for i, point in enumerate(points_list):
        # assume t, [z], y, x
        t = point[0]
        pos = list(point[1:])
        node_id = i
        attrs = {
            time_key: t,
            pos_key: pos,
        }
        cand_graph.add_node(node_id, **attrs)
        if t not in node_frame_dict:
            node_frame_dict[t] = []
        node_frame_dict[t].append(node_id)
    return cand_graph, node_frame_dict


def _compute_node_frame_dict(cand_graph: nx.DiGraph, time_key: str = "time") -> dict[int, list[Any]]:
    """Compute dictionary from time frames to node ids for candidate graph.

    Args:
        cand_graph (nx.DiGraph): A networkx graph
        time_key [str]: key of the time attribute

    Returns:
        dict[int, list[Any]]: A mapping from time frames to lists of node ids.
    """
    node_frame_dict: dict[int, list[Any]] = {}
    for node, data in cand_graph.nodes(data=True):
        t = data[time_key]
        if t not in node_frame_dict:
            node_frame_dict[t] = []
        node_frame_dict[t].append(node)
    return node_frame_dict


def create_kdtree(cand_graph: nx.DiGraph, node_ids: Iterable[Any], pos_key: str = "pos") -> KDTree:
    """Create a kdtree with the given nodes from the candidate graph.
    Will fail if provided node ids are not in the candidate graph.

    Args:
        cand_graph (nx.DiGraph): A candidate graph
        node_ids (Iterable[Any]): The nodes within the candidate graph to
            include in the KDTree. Useful for limiting to one time frame.
        pos_key [str]: key of the spatial position attribute     

    Returns:
        KDTree: A KDTree containing the positions of the given nodes.
    """
    positions = [cand_graph.nodes[node][pos_key] for node in node_ids]
    return KDTree(positions)


def add_cand_edges(
    cand_graph: nx.DiGraph,
    time_key: str,
    pos_key: str,
    max_edge_distance: float,
    node_frame_dict: None | dict[int, list[Any]] = None,
) -> None:
    """Add candidate edges to a candidate graph by connecting all nodes in adjacent
    frames that are closer than max_edge_distance. Also adds attributes to the edges.

    Args:
        cand_graph (nx.DiGraph): Candidate graph with only nodes populated. Will
            be modified in-place to add edges.
        time_key [str]: key of the time attribute
        pos_key [str]: key of the spatial position attribute            
        max_edge_distance (float): Maximum distance that objects can travel between
            frames. All nodes within this distance in adjacent frames will by connected
            with a candidate edge.
        node_frame_dict (dict[int, list[Any]] | None, optional): A mapping from frames
            to node ids. If not provided, it will be computed from cand_graph. Defaults
            to None.
    """

    if not node_frame_dict:
        node_frame_dict = _compute_node_frame_dict(cand_graph, time_key)

    frames = sorted(node_frame_dict.keys())
    prev_node_ids = node_frame_dict[frames[0]]
    prev_kdtree = create_kdtree(cand_graph, prev_node_ids, pos_key)
    for frame in tqdm(frames):
        if frame + 1 not in node_frame_dict:
            continue
        next_node_ids = node_frame_dict[frame + 1]
        next_kdtree = create_kdtree(cand_graph, next_node_ids, pos_key)

        matched_indices = prev_kdtree.query_ball_tree(next_kdtree, max_edge_distance)
        print('node frame dict', node_frame_dict)

        for prev_node_id, next_node_indices in zip(
            prev_node_ids, matched_indices, strict=False
        ):
            for next_node_index in next_node_indices:
                next_node_id = next_node_ids[next_node_index]
                print('adding an edge between node', prev_node_id, 'and', next_node_id)
                cand_graph.add_edge(prev_node_id, next_node_id)

        prev_node_ids = next_node_ids
        prev_kdtree = next_kdtree


def compute_graph_from_seg(
    segmentation: np.ndarray,
    features: FeatureSet,
    max_edge_distance: float,
    iou: bool = False,
    intensity_image: np.ndarray | None = None,
    scale: list[float] | None = None,
) -> nx.DiGraph:
    """Construct a candidate graph from a segmentation array. Nodes are placed at the
    centroid of each segmentation and edges are added for all nodes in adjacent frames
    within max_edge_distance.

    Args:
        segmentation (np.ndarray): A numpy array with integer labels and dimensions
            (t, [z], y, x).
        max_edge_distance (float): Maximum distance that objects can travel between
            frames. All nodes with centroids within this distance in adjacent frames
            will by connected with a candidate edge.
        iou (bool, optional): Whether to include IOU on the candidate graph.
            Defaults to False.
        intensity_image (np.array, optional): the intensity image to be used if intensity_mean is among the features to be measured.
        scale (list[float] | None, optional): The scale of the segmentation data.
            Will be used to rescale the point locations and attribute computations.
            Defaults to None, which implies the data is isotropic.
        features (list[str] | None = []): A list of features (regionprops) to measure.

    Returns:
        nx.DiGraph: A candidate graph that can be passed to the motile solver
    """

    # add nodes
    cand_graph, node_frame_dict = nodes_from_segmentation(
        segmentation, features=features, intensity_image=intensity_image, scale=scale
    )

    # add edges
    add_cand_edges(cand_graph,
        time_key = features.time.key,
        pos_key = features.position.key,
        max_edge_distance=max_edge_distance,
        node_frame_dict=node_frame_dict,
    )
    if iou:
        # Scale does not matter to IOU, because both numerator and denominator
        # are scaled by the anisotropy.
        add_iou(cand_graph, segmentation, node_frame_dict)

    return cand_graph


def compute_graph_from_points_list(
    points_list: np.ndarray,
    features: FeatureSet,
    max_edge_distance: float,
    scale: list[float] | None = None,
) -> nx.DiGraph:
    """Construct a candidate graph from a points list.

    Args:
        points_list (np.ndarray): An NxD numpy array with N points and D
            (3 or 4) dimensions. Dimensions should be in order  (t, [z], y, x).
        max_edge_distance (float): Maximum distance that objects can travel between
            frames. All nodes with centroids within this distance in adjacent frames
            will by connected with a candidate edge.
        scale (list[float] | None, optional): Amount to scale the points in each
            dimension. Only needed if the provided points are in "voxel" coordinates
            instead of world coordinates. Defaults to None, which implies the data is
            isotropic.

    Returns:
        nx.DiGraph: A candidate graph that can be passed to the motile solver.
    """
    # add nodes
    cand_graph, node_frame_dict = nodes_from_points_list(points_list, time_key = features.time.key, pos_key = features.position.key, scale=scale)

    # add edges
    add_cand_edges(
        cand_graph,
        time_key = features.time.key,
        pos_key = features.position.key,
        max_edge_distance=max_edge_distance,
        node_frame_dict=node_frame_dict,
    )
    return cand_graph

def graph_to_nx(graph: TrackGraph) -> nx.DiGraph:
    """Convert a motile TrackGraph into a networkx DiGraph.

    Args:
        graph (TrackGraph): TrackGraph to be converted to networkx

    Returns:
        nx.DiGraph: Directed networkx graph with same nodes, edges, and attributes.
    """
    nx_graph = nx.DiGraph()
    nodes_list = list(graph.nodes.items())
    nx_graph.add_nodes_from(nodes_list)
    edges_list = [
        (edge_id[0], edge_id[1], data) for edge_id, data in graph.edges.items()
    ]
    nx_graph.add_edges_from(edges_list)
    return nx_graph

def add_iou(
    cand_graph: nx.DiGraph,
    segmentation: np.ndarray,
    node_frame_dict: dict[int, list[int]] | None = None,
    multiseg=False,
) -> None:
    """Add IOU to the candidate graph.

    Args:
        cand_graph (nx.DiGraph): Candidate graph with nodes and edges already populated
        segmentation (np.ndarray): segmentation that was used to create cand_graph.
            Has shape ([h], t, [z], y, x), where h is the number of hypotheses if
            multiseg is True.
        node_frame_dict(dict[int, list[Any]] | None, optional): A mapping from
            time frames to nodes in that frame. Will be computed if not provided,
            but can be provided for efficiency (e.g. after running
            nodes_from_segmentation). Defaults to None.
        multiseg (bool): Flag indicating if the given segmentation is actually multiple
            stacked segmentations. Defaults to False.
    """
    if node_frame_dict is None:
        node_frame_dict = _compute_node_frame_dict(cand_graph)
    frames = sorted(node_frame_dict.keys())
    ious = _get_iou_dict(segmentation, multiseg=multiseg)
    for frame in tqdm(frames):
        if frame + 1 not in node_frame_dict.keys():
            continue
        next_nodes = node_frame_dict[frame + 1]
        for node_id in node_frame_dict[frame]:
            for next_id in next_nodes:
                iou = ious.get(node_id, {}).get(next_id, 0)
                if (node_id, next_id) in cand_graph.edges:
                    cand_graph.edges[(node_id, next_id)][EdgeAttr.IOU.value] = iou