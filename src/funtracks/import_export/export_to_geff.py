from __future__ import annotations

from typing import (
    TYPE_CHECKING,
)

import geff

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

    axis_names = ["y", "x"] if tracks.ndim == 3 else ["z", "y", "x"]
    geff.write_nx(tracks.graph, directory, tracks.pos_attr, axis_names)
