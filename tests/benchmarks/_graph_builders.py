"""Builders for the synthetic graphs the benchmarks run against.

A plain module rather than conftest.py, because benchmark modules import these builders
directly and a conftest is not importable. The conftest next door holds only fixtures.
"""

import numpy as np
import polars as pl
import tracksdata as td
from skimage.draw import disk
from tracksdata.nodes import Mask

from funtracks.data_model import Tracks
from funtracks.utils.tracksdata_utils import create_empty_graph


def make_tracks(
    n_frames: int,
    cells_per_frame: int,
    frame_shape: tuple[int, int],
    seed: int = 42,
) -> Tracks:
    """Build a synthetic Tracks with segmentation masks.

    Produces a *solution* graph, not a candidate graph: cells are linked 1-to-1 between
    adjacent frames so every node has out-degree <= 1. This matters because the user
    actions assume a tracking solution -- UserDeleteEdge rejects a removal that would
    leave out-degree > 1, and UserAddEdge rejects merges -- so they raise
    InvalidActionError on the multi-successor candidate graphs that
    compute_graph_from_seg produces.

    Args:
        n_frames: Number of time points.
        cells_per_frame: Cells generated per time point.
        frame_shape: Spatial (y, x) shape of each frame.
        seed: Seed for the random generator, for reproducible geometry.

    Returns:
        A Tracks with pos, area, mask and bbox node attrs, iou edge attrs, and a
        writable unmanaged "score" node attr (populated on every node) for
        attribute-update benchmarks.
    """
    rng = np.random.default_rng(seed)
    graph = create_empty_graph(
        node_attributes=[
            "pos",
            "area",
            td.DEFAULT_ATTR_KEYS.MASK,
            td.DEFAULT_ATTR_KEYS.BBOX,
        ],
        node_default_values=[0.0, 0.0, 0.0, 0.0],
        edge_attributes=["iou"],
        position_attrs=["pos"],
        ndim=3,
    )

    nodes: list[dict] = []
    node_ids: list[int] = []
    edges: list[dict] = []
    node_id = 1
    prev_frame_ids: list[int] = []

    for t in range(n_frames):
        frame_ids = []
        for _ in range(cells_per_frame):
            cy = int(rng.integers(25, frame_shape[0] - 25))
            cx = int(rng.integers(25, frame_shape[1] - 25))
            radius = int(rng.integers(8, 15))

            # Mask is stored in its own local frame with an offset bbox, so the
            # disk is drawn centered in a (2r-1, 2r-1) box.
            mask_shape = (2 * radius - 1, 2 * radius - 1)
            rr, cc = disk(
                center=(radius - 1, radius - 1), radius=radius, shape=mask_shape
            )
            mask_arr = np.zeros(mask_shape, dtype=bool)
            mask_arr[rr, cc] = True
            bbox = np.array([cy - radius + 1, cx - radius + 1, cy + radius, cx + radius])

            nodes.append(
                {
                    "t": t,
                    "pos": [float(cy), float(cx)],
                    "area": float(mask_arr.sum()),
                    "solution": True,
                    td.DEFAULT_ATTR_KEYS.MASK: Mask(mask_arr, bbox=bbox),
                    td.DEFAULT_ATTR_KEYS.BBOX: bbox,
                }
            )
            node_ids.append(node_id)
            frame_ids.append(node_id)
            node_id += 1

        # Link 1-to-1 with the previous frame to keep out-degree <= 1.
        for prev_id, cur_id in zip(prev_frame_ids, frame_ids, strict=False):
            edges.append(
                {
                    "source_id": prev_id,
                    "target_id": cur_id,
                    "solution": True,
                    "iou": float(rng.uniform(0.1, 0.9)),
                }
            )
        prev_frame_ids = frame_ids

    # Register "score" before adding the nodes so every node gets a stored value. Adding
    # the key afterwards leaves existing nodes null, and tracksdata applies the default
    # lazily on read (_maybe_fill_null in graph/_rustworkx_graph.py, guarded by
    # s.has_nulls()). That backfill costs two extra polars collects per read until the
    # node is first written, which made the first update of any node ~2x slower than
    # subsequent ones and turned the attribute benchmarks into a measure of null
    # backfill rather than of the update path.
    graph.add_node_attr_key("score", dtype=pl.Float64, default_value=0.0)
    for node in nodes:
        node["score"] = 0.0

    graph.bulk_add_nodes(nodes=nodes, indices=node_ids)
    graph.bulk_add_edges(edges)
    graph._update_metadata(segmentation_shape=(n_frames, *frame_shape))

    return Tracks(
        graph,
        time_attr="t",
        pos_attr="pos",
        tracklet_attr="tracklet_id",
        lineage_attr="lineage_id",
    )
