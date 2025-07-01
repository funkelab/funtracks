import logging

import funlib.persistence as fp
import networkx as nx
from skimage.measure import regionprops
from tqdm import tqdm

from .cand_graph import CandGraph
from .features.feature_set import FeatureSet
from .nx_graph import NxGraph
from .params import CandGraphParams

logger = logging.getLogger(__name__)


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
        props = regionprops(segs, spacing=spacing)
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
