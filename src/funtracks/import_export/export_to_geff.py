from __future__ import annotations

import shutil
from typing import (
    TYPE_CHECKING,
)

import geff
import geff_spec
import networkx as nx
import numpy as np
import zarr
from geff_spec import GeffMetadata

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
        shutil.rmtree(directory)  # remove directory since overwriting in a non-empty zarr
        # dir may trigger geff warnings.

    # Create dir
    directory.mkdir()

    # update the graph to split the position into separate attrs, if they are currently
    # together in a list
    graph, axis_names = split_position_attr(tracks)
    if axis_names is None:
        axis_names = []
    axis_names.insert(0, tracks.features.time_key)
    if axis_names is not None:
        axis_types = (
            ["time", "space", "space"]
            if tracks.ndim == 3
            else ["time", "space", "space", "space"]
        )
    else:
        axis_types = None
    if tracks.scale is None:
        tracks.scale = (1.0,) * tracks.ndim

    metadata = GeffMetadata(
        geff_version=geff_spec.__version__,
        directed=isinstance(graph, nx.DiGraph),
        node_props_metadata={},
        edge_props_metadata={},
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
                "label_prop": "seg_id",
            }
            # TODO: I don't think we necessarily have a seg id in our tracks
        ]

    # Save the graph in a 'tracks' folder
    tracks_path = directory / "tracks"
    tracks_path.mkdir(exist_ok=True)
    geff.write(
        graph=graph,
        store=tracks_path,
        metadata=metadata,
        axis_names=axis_names,
        axis_types=axis_types,
        axis_scales=tracks.scale,
        overwrite=overwrite,
    )


def split_position_attr(tracks: Tracks) -> tuple[nx.DiGraph, list[str] | None]:
    """Spread the spatial coordinates to separate node attrs in order to export to geff
    format.

    Args:
        tracks (funtracks.data_model.Tracks): tracks object holding the graph to be
          converted.

    Returns:
        tuple[nx.DiGraph, list[str]]: graph with a separate positional attribute for each
            coordinate, and the axis names used to store the separate attributes

    """
    pos_key = tracks.features.position_key

    if isinstance(pos_key, str):
        # Position is stored as a single attribute, need to split
        new_graph = tracks.graph.copy()
        new_keys = ["y", "x"]
        if tracks.ndim == 4:
            new_keys.insert(0, "z")
        for _, attrs in new_graph.nodes(data=True):
            pos = attrs.pop(pos_key)
            for i in range(len(new_keys)):
                attrs[new_keys[i]] = pos[i]

        return new_graph, new_keys
    elif pos_key is not None:
        # Position is already split into separate attributes
        return tracks.graph, list(pos_key)
    else:
        return tracks.graph, None
