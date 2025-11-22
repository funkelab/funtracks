from __future__ import annotations

from typing import TYPE_CHECKING

import dask.array as da
import networkx as nx
import numpy as np

from funtracks.import_export.magic_imread import magic_imread

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import ArrayLike

# Constants from import_from_geff
SEG_KEY = "seg_id"


def lazy_load_segmentation(segmentation_path: Path) -> da.Array:
    """Lazily load segmentation from disk without reading into memory."""
    return magic_imread(segmentation_path, use_dask=True)


def import_segmentation(
    segmentation_path: Path,
    graph: nx.DiGraph,
    relabel: bool = False,
) -> da.Array | np.ndarray:
    """Load segmentation data from file and optionally relabel.

    Args:
        segmentation_path: Path to segmentation data
        graph: NetworkX graph with standard keys
        relabel: If True, relabel segmentation from seg_id to node_id

    Returns:
        Segmentation array, possibly relabeled to match node IDs
    """
    segmentation = magic_imread(segmentation_path, use_dask=True)

    # If the provided segmentation has seg ids that are not identical to node ids,
    # relabel it now.
    if relabel:
        # Extract data from graph (which has standard keys)
        times = []
        ids = []
        seg_ids = []

        for node_id, data in graph.nodes(data=True):
            times.append(data["time"])
            ids.append(node_id)
            seg_ids.append(data[SEG_KEY])

        times = np.array(times)
        ids = np.array(ids)
        seg_ids = np.array(seg_ids)

        if not len(times) == len(ids) == len(seg_ids):
            raise ValueError(
                "Encountered missing values in the seg_id to node id conversion."
            )
        segmentation = relabel_seg_id_to_node_id(times, ids, seg_ids, segmentation)

    return segmentation


def relabel_seg_id_to_node_id(
    times: ArrayLike, ids: ArrayLike, seg_ids: ArrayLike, segmentation: da.Array
) -> np.ndarray:
    """Create a new segmentation where masks are relabeled to match node ids.

    TODO: How does this relate to motile_toolbox.ensure_unique_labels? Just lazy/dask?

    Args:
        times (ArrayLike): array of time points, one per node
        ids (ArrayLike): array of node ids
        seg_ids (ArrayLike): array of segmentation ids, one per node
        segmentation (da.array): A dask array where segmentation label values match the
          "seg_id" values.

    Returns:
        np.ndarray: A numpy array of dtype uint64, similar to the input segmentation
            where each segmentation now has a unique label across time that corresponds
            to the ID of each node.
    """

    new_segmentation = np.zeros(segmentation.shape, dtype=np.uint64)
    for i, node in enumerate(ids):
        mask = segmentation[times[i]].compute() == seg_ids[i]
        new_segmentation[times[i], mask] = node

    return new_segmentation
