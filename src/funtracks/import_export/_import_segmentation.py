"""Segmentation import and relabeling utilities.

This module provides functions for loading segmentation data and relabeling
segmentation masks to match graph node IDs.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import dask.array as da
import networkx as nx
import numpy as np

from funtracks.import_export.magic_imread import magic_imread

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


def read_dims(segmentation: Path | np.ndarray):
    arr = load_segmentation(segmentation)
    return arr.ndim


def load_segmentation(segmentation: Path | np.ndarray | da.Array) -> da.Array:
    """Load segmentation from path or wrap array in dask.

    Args:
        segmentation: Path to segmentation file, numpy array, or dask array

    Returns:
        Dask array containing segmentation data
    """
    if isinstance(segmentation, Path):
        return magic_imread(segmentation, use_dask=True)
    elif isinstance(segmentation, da.Array):
        # Already a dask array
        return segmentation
    else:
        # Wrap numpy array in dask array for consistency
        return da.from_array(segmentation, chunks=segmentation.shape)


def relabel_segmentation(
    seg_array: da.Array | np.ndarray,
    graph: nx.DiGraph,
    node_ids: ArrayLike,
    seg_ids: ArrayLike,
    time_values: ArrayLike,
) -> np.ndarray:
    """Relabel segmentation from seg_id to node_id.

    Handles the case where node_id 0 exists by offsetting all node IDs by 1,
    since 0 is reserved for background in segmentation arrays.

    Args:
        seg_array: Segmentation array (dask or numpy)
        graph: NetworkX graph (modified in-place if node_id 0 exists)
        node_ids: Array of node IDs
        seg_ids: Array of segmentation IDs corresponding to each node
        time_values: Array of time values for each node

    Returns:
        Relabeled segmentation as numpy array with dtype uint64
    """
    # Convert to numpy arrays for processing
    node_ids = np.asarray(node_ids)
    seg_ids = np.asarray(seg_ids)
    time_values = np.asarray(time_values)

    # Compute segmentation if it's a dask array
    computed_seg = seg_array.compute() if isinstance(seg_array, da.Array) else seg_array

    # If node_id 0 exists, we need to offset all labels by 1 since 0 is background
    # in segmentation arrays. We also need to relabel the graph nodes.
    offset = 1 if 0 in node_ids else 0
    if offset:
        mapping = {old_id: old_id + offset for old_id in graph.nodes()}
        nx.relabel_nodes(graph, mapping, copy=False)
        # Update node_ids array to match
        node_ids = node_ids + offset

    # Relabel segmentation: seg_id -> node_id (with offset if needed)
    new_segmentation = np.zeros_like(computed_seg).astype(np.uint64)

    for t in np.unique(time_values):
        # Get nodes at this time point
        mask = time_values == t
        seg_ids_t = seg_ids[mask]
        node_ids_t = node_ids[mask]

        # Create mapping: seg_id -> node_id
        seg_to_node = dict(zip(seg_ids_t, node_ids_t, strict=True))

        # Apply mapping to segmentation at this time point
        for seg_id, node_id in seg_to_node.items():
            new_segmentation[t][computed_seg[t] == seg_id] = node_id

    return new_segmentation


# TODO: export segmentation with check to relabel to track_id
