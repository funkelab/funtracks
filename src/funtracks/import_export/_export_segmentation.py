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


def export_segmentation(
    tracks: Tracks,
    output_path: Path,
    file_format: Literal["zarr", "tiff"] = "zarr",
    label_attr: str | None = "track_id",
    zarr_format: Literal[2, 3] = 2,
    node_ids: set[int] | None = None,
) -> None:
    """Export the segmentation to zarr or tiff, optionally painting cells by a
    node attribute.

    Writes frame-by-frame so large datasets are never fully materialized in memory.

    Args:
        tracks: Tracks object with segmentation present.
        output_path: Destination path. For zarr, a directory will be created.
            For tiff, a .tif file will be written.
        file_format: Output format, either "zarr" or "tiff". Defaults to "zarr".
        label_attr: Node attribute used to paint cell labels. When set, each cell is
            painted with the value of this attribute (e.g. "track_id"). When None,
            the original segmentation labels (node IDs) are preserved as-is.
            Defaults to "track_id".
        zarr_format: Zarr format version. Only used when file_format="zarr".
            Defaults to 2.
        node_ids: Optional subset of node IDs to include. Cells not in this set
            are painted as 0 (background). Has no effect when label_attr is None.

    Raises:
        ValueError: If tracks.segmentation is None.
        ValueError: If label_attr is specified but not present as a node attribute.
    """
    if tracks.segmentation is None:
        raise ValueError("tracks.segmentation is None — cannot export segmentation.")

    if label_attr is not None:
        existing_attrs = tracks.graph.node_attrs().columns
        if label_attr not in existing_attrs:
            raise ValueError(
                f"label_attr '{label_attr}' is not a node attribute. "
                f"Available attributes: {existing_attrs}"
            )

    shape = tracks.segmentation.shape

    if label_attr is not None:
        graph = (
            tracks.graph.filter(node_ids=list(node_ids)).subgraph()
            if node_ids is not None
            else tracks.graph
        )
        view = GraphArrayView(graph, label_attr, shape=shape)

        def get_frame(t: int) -> np.ndarray:
            return np.array(view[t])

        dtype = view.dtype
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
        with tifffile.TiffWriter(output_path, bigtiff=False) as tif:
            for t in range(shape[0]):
                tif.write(get_frame(t).astype(tiff_dtype), contiguous=True)
