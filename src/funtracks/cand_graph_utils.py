import logging

import funlib.persistence as fp
import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

from funtracks.features._base import Feature, FeatureType
from funtracks.features.measurement_features import Intensity
from funtracks.features.regionprops_extended import regionprops_extended
import dask.array as da
from .cand_graph import CandGraph
from .features.feature_set import FeatureSet
from .nx_graph import NxGraph
from .params import CandGraphParams

logger = logging.getLogger(__name__)

def graph_from_points(
    points_data: pd.DataFrame, column_mapping: dict[str:str], feature_set: FeatureSet, cand_graph_params: CandGraphParams
) -> nx.DiGraph:
    """Create a graph from points data, representing t(z)yx coordinates
    Args:
        points_data (pd.DataFrame): dataframe holding the point t, (z), y, x coordinates
        column_mapping (dict[str: str]): dictionary mapping each dimension to a column in
         the dataframe
    Returns:
        nx.DiGraph with nodes only.
    """
    graph = nx.DiGraph()
    for _id, row in points_data.iterrows():
        if "z" in column_mapping:
            pos = [
                row.get(column_mapping["z"], None),
                row.get(column_mapping["y"], None),
                row.get(column_mapping["x"], None),
            ]
        else:
            pos = [row.get(column_mapping["y"], None), row.get(column_mapping["x"], None)]

        t = row.get(column_mapping["t"], None)

        attrs = {
            "t": int(t),
            "pos": pos,
        }

        graph.add_node(_id + 1, **attrs)

    return CandGraph(NxGraph, graph, feature_set, cand_graph_params)

def graph_from_df(
    df: pd.DataFrame,
    segmentation: fp.Array | None,
    intensity_image: list[fp.Array] | None,
    ndim: int,
    n_channels: int,
    scaling: list[float],
    features: list[dict[str : Feature | str | bool]],
    cand_graph_params: CandGraphParams,
) -> nx.DiGraph:
    """Construct a nx.DiGraph from a pd.DataFrame, and optionally computes attributes from
    a list of features and adds them to the nodes.

    Args:
        df (pd.DataFrame): dataframe holding the tracks data.
        segmentation (fp.Array | None): segmentation data
        intensity_image (list[fp.Array] | None): intensity image list, one per channel
        ndim (int): number of dimensions of the dataset, 3,4,5
        n_channels (int): the size of the channel dimension.
        scaling (list[float]): spatial calibration for (z), y, x dimensions.
        features (list[dict[str: Feature|str|bool]]): Features to be measured.
            - feature (funtracks.Feature)
            - include (bool): whether to include this feature on the graph
            - from_column (str | None): optional dataframe column from which to take the
            measurement (instead of recomputing)
        cand_graph_params (CandGraphParams)

    Returns:
        nx.DiGraph with nodes and edges, and computed features on the nodes.

    """

    seg = segmentation is not None
    feature_set = FeatureSet(ndim=ndim, seg=seg, pos_attr="pos", time_attr="t")
    for feature in features:
        if isinstance(feature["feature"], Intensity):
            feature["feature"].value_names = (
                "Intensity"
                if n_channels == 1
                else [f"Intensity_chan{chan}" for chan in range(n_channels)]
            )
        if feature['include']:
            feature_set.add_feature(feature["feature"])  # add the Feature instance from
        # the feature dict to the feature_set

    graph = nx.DiGraph()
    for _, row in tqdm(df.iterrows(), total=len(df)):
        _id = int(row["id"])
        parent_id = row["parent_id"]
        if "z" in df.columns:
            pos = [row["z"], row["y"], row["x"]]
        else:
            pos = [row["y"], row["x"]]

        attrs = {
            "t": int(row["t"]),
            "pos": pos,
        }

        # add additional features that should be recomputed
        features_to_recompute = [
            f["feature"]
            for f in features
            if f["include"]
            and f["feature"].computed
            and f["feature"].feature_type == FeatureType.NODE
            and f["from_column"] is None
            and f["feature"].regionprops_name is not None
        ]

        if len(features_to_recompute) > 0:
            t = int(row["t"])            
            if intensity_image is not None:
                int_stack = []
                for int_img in intensity_image:
                    if isinstance(int_img[t], da.Array):
                        int_stack.append(int_img[t].compute())
                    else: 
                        int_stack.append(int_img[t])
                intensity = np.stack(int_stack, axis=-1) # regionprops wants channel dim to be
                # last
            else:
                intensity = None

            # compute the feature
            seg = segmentation[t].compute() if isinstance(segmentation, da.Array) else segmentation[t]
            props = regionprops_extended(
                (seg == _id).astype(np.uint8),
                intensity_image=intensity,
                spacing=scaling,
            )
            if props:
                regionprop = props[0]
                for feature in features_to_recompute:
                    value = getattr(regionprop, feature.regionprops_name)
                    if isinstance(value, tuple):
                        value = list(value)
                    attrs[feature.attr_name] = value

        # optionally import extra features directly from the table, without recomputing
        features_to_import_from_df = [
            f for f in features if f["include"] and f["from_column"] is not None
        ]

        for feature in features_to_import_from_df:
            attrs[feature["feature"].attr_name] = row.get(feature["from_column"])

        # add the node to the graph
        graph.add_node(_id, **attrs)

        # add the edge to the graph, if the node has a parent
        # note: this loading format does not support edge attributes
        if not pd.isna(parent_id) and parent_id != -1:
            assert parent_id in graph.nodes, (
                f"Parent id {parent_id} of node {_id} not in graph yet"
            )
            graph.add_edge(parent_id, _id)

    return CandGraph(NxGraph, graph, feature_set, cand_graph_params)


def graph_from_segmentation(
    segmentation: fp.Array | None,
    intensity_image: list[fp.Array] | None,
    ndim: int,
    n_channels: int,
    scaling: list[float] | None,
    features: list[dict[str : Feature | str | bool]],
    cand_graph_params: CandGraphParams,
) -> nx.DiGraph:
    """Construct a nx.DiGraph from a pd.DataFrame, and optionally computes attributes from
    a list of features and adds them to the nodes.

    Args:
        segmentation (fp.Array | None): segmentation data
        intensity_image (list[fp.Array] | None): intensity image
        ndim (int): number of dimensions of the dataset, 3,4,5
        n_channels (int): the size of the channel dimension.
        scaling (list[float]): spatial calibration for (z), y, x dimensions.
        features (list[dict[str: Feature|str|bool]]): Features to be measured.
            - feature (funtracks.Feature)
            - include (bool): whether to include this feature on the graph
            - from_column (str | None): optional dataframe column from which to take the
            measurement (instead of recomputing)
        cand_graph_params (CandGraphParams).

    Returns:
        nx.DiGraph with nodes and edges, and computed features on the nodes.

    """

    seg = segmentation is not None
    feature_set = FeatureSet(ndim=ndim, seg=seg, pos_attr="pos", time_attr="t")
    for feature in features:
        if isinstance(feature["feature"], Intensity):
            feature["feature"].value_names = (
                "Intensity"
                if n_channels == 1
                else [f"Intensity_chan{chan}" for chan in range(n_channels)]
            )
        if feature['include']:
            feature_set.add_feature(feature["feature"])  # add the Feature instance from
        # the feature dict to the feature_set

    graph = nx.DiGraph()
    for t in tqdm(range(segmentation.shape[0])):
        
        if intensity_image is not None:
            int_stack = []
            for int_img in intensity_image:
                if isinstance(int_img[t], da.Array):
                    int_stack.append(int_img[t].compute())
                else:
                    int_stack.append(int_img[t])
            intensity = np.stack(int_stack, axis=-1) # regionprops wants channel dim to be
            # last
        else:
            intensity = None

        seg = segmentation[t].compute() if isinstance(segmentation, da.Array) else segmentation[t]
        props = regionprops_extended(
                seg,
                intensity_image=intensity,
                spacing=scaling)
        for regionprop in props:
            node_id = regionprop.label
            attrs = {
                feature_set.time.attr_name: t,
            }
            for feature in feature_set.node_features:
                if feature.computed and feature.regionprops_name is not None:
                    # to list gives floats/ints in the case of single items
                    value = getattr(regionprop, feature.regionprops_name)
                    if isinstance(value, tuple):
                        value = list(value)
                    attrs[feature.attr_name] = value
            graph.add_node(node_id, **attrs)

    return CandGraph(NxGraph, graph, feature_set, cand_graph_params)


def nodes_from_segmentation(segmentation: fp.Array, params: CandGraphParams) -> CandGraph:
    """Extract candidate nodes from a segmentation.

    Args:
        segmentation (fp.Array): A funlib persistence array with integer labels and dimensions
            (t, [z], y, x). Labels must be unique across time, and the label
            will be used as the node id.

    Returns:
        tuple[nx.DiGraph, dict[int, list[Any]]]: A candidate graph with only nodes,
            and a mapping from time frames to node ids.
    """
    logger.debug("Extracting nodes from segmentation")
    cand_graph = nx.DiGraph()

    scale = segmentation.voxel_size
    ndim = segmentation.data.ndim
    spacing = tuple(scale[1:]) if scale is not None else None

    features = FeatureSet(ndim, seg=True)
    for t in tqdm(
        range(len(segmentation.data)), desc="Extracting graph nodes from segmentation"
    ):
        segs = segmentation[t]
        props = regionprops_extended(segs, spacing=spacing)
        for regionprop in props:
            node_id = regionprop.label
            attrs = {
                features.time.attr_name: t,
            }
            for feature in features.node_features:
                if feature.computed and feature.regionprops_name is not None:
                    # to list gives floats/ints in the case of single items
                    value = getattr(regionprop, feature.regionprops_name)
                    if isinstance(value, tuple):
                        value = [i.tolist() for i in value]
                    else:
                        value = value.tolist()
                    attrs[feature.attr_name] = value
            cand_graph.add_node(node_id, **attrs)
    return CandGraph(NxGraph, cand_graph, features, params)
