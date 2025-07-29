from __future__ import annotations

from typing import (
    TYPE_CHECKING,
)

import geff
import networkx as nx
import numpy as np
import zarr
from geff.affine import Affine
from geff.metadata_schema import GeffMetadata

from funtracks.data_model.graph_attributes import NodeAttr

if TYPE_CHECKING:
    from pathlib import Path

    from funtracks.data_model.tracks import Tracks


def export_to_geff(tracks: Tracks, directory: Path, overwrite: bool = False):
    """Export the Tracks nxgraph to geff.

    Args:
        tracks (Tracks): Tracks object containing a graph to save.
        directory (Path): Destination directory for saving the Zarr.
        overwrite (bool): If True, allows writing into a non-empty directory.

    Raises:
        ValueError: If the path is invalid, parent doesn't exist, is not a directory,
                    or if the directory is not empty and overwrite is False.
    """
    directory = directory.resolve(strict=False)

    # Ensure parent directory exists
    parent = directory.parent
    if not parent.exists():
        raise ValueError(f"Parent directory {parent} does not exist.")

    # Check target directory
    if directory.exists():
        if not directory.is_dir():
            raise ValueError(f"Provided path {directory} exists but is not a directory.")
        if any(directory.iterdir()) and not overwrite:
            raise ValueError(
                f"Directory {directory} is not empty. Use overwrite=True to allow export."
            )
    else:
        # Create dir
        directory.mkdir()

    # update the graph to split the position into separate attrs, if they are currently
    # together in a list
    if isinstance(tracks.pos_attr, str):
        graph = split_position_attr(tracks)
        axis_names = (
            [tracks.time_attr, "y", "x"]
            if tracks.ndim == 3
            else [tracks.time_attr, "z", "y", "x"]
        )
    else:
        graph = tracks.graph
        axis_names = list(tracks.pos_attr)
        axis_names.insert(0, tracks.time_attr)

    axis_types = (
        ["time", "space", "space"]
        if tracks.ndim == 3
        else ["time", "space", "space", "space"]
    )

    # calculate affine matrix
    if tracks.scale is None:
        tracks.scale = (1.0,) * tracks.ndim
    linear_matrix = np.diag(tracks.scale)
    offset = 0.0  # no offset or translation
    affine = Affine.from_matrix_offset(linear_matrix, offset)

    # Create metadata and add the affine matrix. Axes will be added automatically.
    metadata = GeffMetadata(
        geff_version=geff.__version__,
        directed=isinstance(graph, nx.DiGraph),
        affine=affine,
    )

    # Save segmentation if present
    if tracks.segmentation is not None:
        seg_path = directory / "segmentation"
        seg_path.mkdir(exist_ok=True)
        zarr.save_array(str(seg_path), np.asarray(tracks.segmentation))
        metadata.related_objects = [
            {
                "path": "../segmentation",
                "type": "labels",
                "label_prop": NodeAttr.SEG_ID.value,
            }
        ]

    # Save the graph in a 'tracks' folder
    tracks_path = directory / "tracks"
    tracks_path.mkdir(exist_ok=True)
    geff.write_nx(
        graph=graph,
        store=tracks_path,
        metadata=metadata,
        axis_names=axis_names,
        axis_types=axis_types,
    )


def split_position_attr(tracks: Tracks) -> nx.DiGraph:
    """Spread the spatial coordinates to separate node attrs in order to export to geff
    format.

    Args:
        tracks (funtracks.data_model.Tracks): tracks object holding the graph to be
          converted.

    Returns:
        nx.DiGraph with a separate positional attribute for each coordinate.

    """
    new_graph = tracks.graph.copy()

    for _, attrs in new_graph.nodes(data=True):
        pos = attrs.pop(tracks.pos_attr)

        if len(pos) == 2:
            attrs["y"] = pos[0]
            attrs["x"] = pos[1]
        elif len(pos) == 3:
            attrs["z"] = pos[0]
            attrs["y"] = pos[1]
            attrs["x"] = pos[2]

    return new_graph
