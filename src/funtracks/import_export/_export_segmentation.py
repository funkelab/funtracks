"""Helper for exporting segmentation data to zarr or tiff, optionally relabeled."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import tifffile
from tracksdata.array import GraphArrayView

from funtracks.utils import setup_zarr_array

if TYPE_CHECKING:
    from pathlib import Path

    from funtracks.data_model.tracks import Tracks


def resolve_relabel_attr(
    tracks: Tracks,
    relabel: Literal["tracklet", "lineage", None],
) -> str | None:
    """Resolve a ``relabel`` option to the corresponding graph attribute key.

    Validates that the resolved key is both configured on the ``Tracks``
    features and present as a node attribute in the graph.

    Returns:
        The attribute key string, or ``None`` when *relabel* is ``None``.

    Raises:
        ValueError: If the requested key is not configured on *tracks* or is
            not present as a node attribute.
    """
    if relabel == "tracklet":
        label_attr = tracks.features.tracklet_key
        if label_attr is None:
            raise ValueError(
                "relabel='tracklet' requested but tracks has no tracklet key."
            )
    elif relabel == "lineage":
        label_attr = tracks.features.lineage_key
        if label_attr is None:
            raise ValueError("relabel='lineage' requested but tracks has no lineage key.")
    else:
        return None

    existing_attrs = tracks.graph.node_attr_keys()
    if label_attr not in existing_attrs:
        raise ValueError(
            f"relabel='{relabel}' resolved to attribute '{label_attr}', "
            f"which is not a node attribute. "
            f"Available attributes: {existing_attrs}"
        )
    return label_attr


def export_segmentation(
    tracks: Tracks,
    output_path: Path,
    file_format: Literal["zarr", "tiff"] = "zarr",
    relabel: Literal["tracklet", "lineage", None] = "tracklet",
    zarr_format: Literal[2, 3] = 2,
    node_ids: set[int] | None = None,
) -> None:
    """Export the segmentation to zarr or tiff, optionally painting cells by
    tracklet or lineage ID.

    Writes frame-by-frame so large datasets are never fully materialized in memory.

    Args:
        tracks: Tracks object with segmentation present.
        output_path: Destination path. For zarr, a directory will be created.
            For tiff, a .tif file will be written.
        file_format: Output format, either "zarr" or "tiff". Defaults to "zarr".
        relabel: How to relabel the segmentation cells.
            "tracklet" (default): paint each cell with its tracklet ID.
            "lineage": paint each cell with its lineage ID.
            None: preserve original segmentation labels (node IDs) as-is.
        zarr_format: Zarr format version. Only used when file_format="zarr".
            Defaults to 2.
        node_ids: Optional subset of node IDs to include. Cells not in this set
            are painted as 0 (background). Has no effect when relabel is None.

    Raises:
        ValueError: If tracks.segmentation is None.
        ValueError: If relabel is requested but the corresponding key is not set
            or not present as a node attribute.
    """
    if tracks.segmentation is None:
        raise ValueError("tracks.segmentation is None — cannot export segmentation.")

    # Resolve the graph attribute key from the relabel option
    label_attr = resolve_relabel_attr(tracks, relabel)

    shape = tracks.segmentation.shape

    graph = (
        tracks.graph.filter(node_ids=list(node_ids)).subgraph()
        if node_ids is not None
        else tracks.graph
    )

    if label_attr is not None:
        view = GraphArrayView(graph, label_attr, shape=shape)

        def get_frame(t: int) -> np.ndarray:
            return np.array(view[t])

        dtype = view.dtype

    elif node_ids is not None:
        seg = tracks.segmentation
        allowed = np.array(list(node_ids))

        def get_frame(t: int) -> np.ndarray:
            frame = np.array(seg[t])
            return np.where(np.isin(frame, allowed), frame, 0)

        dtype = seg.dtype

    else:
        seg = tracks.segmentation

        def get_frame(t: int) -> np.ndarray:
            return np.array(seg[t])

        dtype = seg.dtype

    if file_format == "zarr":
        chunks: tuple[int, ...] = (1, *tuple(min(512, d) for d in shape[1:]))
        z = setup_zarr_array(
            output_path,
            zarr_format=zarr_format,
            shape=shape,
            dtype=dtype,
            chunks=chunks,
        )
        for t in range(shape[0]):
            z[t] = get_frame(t)

    else:  # tiff
        # Cast to int32 for broad compatibility (e.g. Fiji does not display int64)
        tiff_dtype = np.int32
        with tifffile.TiffWriter(output_path, bigtiff=True) as tif:
            for t in range(shape[0]):
                tif.write(
                    get_frame(t).astype(tiff_dtype), contiguous=True
                )  # compression may prevent contiguous writing
