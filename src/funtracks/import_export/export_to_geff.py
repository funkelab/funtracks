from __future__ import annotations
from typing import (
    TYPE_CHECKING,
)
import geff 

if TYPE_CHECKING:
    from pathlib import Path
    from funtracks.data_model.tracks import Tracks

def export_to_geff(tracks: Tracks, directory: Path):
    axis_names = ["y", "x"] if len(tracks.scale) == 3 else ["z", "y", "x"]
    geff.write_nx(tracks.graph, directory, tracks.pos_attr, axis_names)