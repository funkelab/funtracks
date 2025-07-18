from __future__ import annotations
from typing import (
    TYPE_CHECKING,
)
import geff 

if TYPE_CHECKING:
    from pathlib import Path
    from funtracks.data_model.tracks import Tracks

def export_to_geff(tracks: Tracks, directory: Path):
    """Export the Tracks nxgraph to geff.
    Args: 
        tracks (Tracks): tracks object containing a graph to save.
        directory (Path): destination for saving the zarr.
    """
    
    if directory.exists():
        if not directory.is_dir():
            raise ValueError(f"Provided path {directory} is not a directory.")
        if any(directory.iterdir()):
            raise ValueError(f"Directory {directory} is not empty.")
    else:
        raise ValueError(f"Directory {directory} does not exist.")
    
    axis_names = ["y", "x"] if tracks.ndim == 3 else ["z", "y", "x"]
    geff.write_nx(tracks.graph, directory, tracks.pos_attr, axis_names)