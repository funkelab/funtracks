from __future__ import annotations

import ast
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
from geff._typing import InMemoryGeff
from geff_spec.utils import (
    add_or_update_props_metadata,
    create_or_update_metadata,
    create_props_metadata,
)

from funtracks.data_model.graph_attributes import NodeAttr

from .._tracks_builder import TracksBuilder

if TYPE_CHECKING:
    from funtracks.data_model.solution_tracks import SolutionTracks


class CSVTracksBuilder(TracksBuilder):
    """Builder for importing tracks from CSV/DataFrame format."""

    def __init__(self):
        """Initialize CSV builder with CSV-specific required features."""
        super().__init__()
        self.required_features.extend(["id", "parent_id"])

    def read_header(self, source_path: Path) -> None:
        """Read CSV column names.

        Args:
            source_path: Path to CSV file
        """
        df = pd.read_csv(source_path, nrows=0)
        self.importable_node_props = df.columns.tolist()
        self.importable_edge_props = []  # CSV has no edge properties

    def load_source(
        self,
        source_path: Path,
        name_map: dict[str, str],
        node_features: dict[str, bool] | None = None,
    ) -> None:
        """Load CSV and convert to InMemoryGeff format.

        Args:
            source_path: Path to CSV file
            name_map: Maps standard keys to CSV column names
            node_features: Optional features dict for backward compatibility
        """
        # Read CSV
        df = pd.read_csv(source_path)

        # For backward compatibility, extend name_map with node_features
        # Only add features that should be loaded (recompute=False)
        extended_name_map = dict(name_map)
        if node_features is not None:
            for feature_key, recompute in node_features.items():
                if feature_key not in extended_name_map and not recompute:
                    # Assume feature name in CSV matches standard key
                    extended_name_map[feature_key] = feature_key

        # Apply extended_name_map: rename CSV columns to standard keys
        rename_map = {
            csv_col: std_key
            for std_key, csv_col in extended_name_map.items()
            if csv_col is not None and csv_col in df.columns
        }
        df = df.rename(columns=rename_map)

        # Only keep columns that were in extended_name_map
        columns_to_keep = [
            std_key
            for std_key, csv_col in extended_name_map.items()
            if csv_col in rename_map.values() or std_key in df.columns
        ]
        df = df[columns_to_keep]

        # Convert NaN to None
        df = df.map(lambda x: None if pd.isna(x) else x)

        # Handle type conversions - lists stored as strings like "[1, 2, 3]"
        for col in df.columns:
            if col not in name_map:  # custom attributes
                df[col] = df[col].apply(
                    lambda x: ast.literal_eval(x)
                    if isinstance(x, str) and x.startswith("[") and x.endswith("]")
                    else x
                )

        # Determine dimensionality
        self.position_attr = ["z", "y", "x"] if "z" in df.columns else ["y", "x"]
        self.ndim = len(self.position_attr) + 1  # +1 for time

        # Convert DataFrame to InMemoryGeff format
        df_dict = df.to_dict(orient="list")
        node_ids = np.array(df_dict.pop("id"))
        parent_ids = df_dict.pop("parent_id")

        # Build position property from individual coordinate columns
        if "z" in df_dict:
            pos_values = np.array(
                [
                    list(p)
                    for p in zip(
                        df_dict.pop("z"),
                        df_dict.pop("y"),
                        df_dict.pop("x"),
                        strict=True,
                    )
                ]
            )
        else:
            pos_values = np.array(
                [list(p) for p in zip(df_dict.pop("y"), df_dict.pop("x"), strict=True)]
            )

        # Build node_props with GEFF-compatible structure
        node_props: dict[str, dict[str, np.ndarray | None]] = {
            NodeAttr.POS.value: {"values": pos_values, "missing": None}
        }
        for prop_name, values in df_dict.items():
            node_props[prop_name] = {"values": np.array(values), "missing": None}

        # Extract edge IDs from parent_id column
        edge_tuples = [
            (int(parent_id), int(child_id))
            for parent_id, child_id in zip(parent_ids, node_ids, strict=True)
            if not pd.isna(parent_id) and parent_id != -1
        ]
        edge_ids = np.array(edge_tuples)

        # CSV format doesn't support edge attributes
        edge_props: dict[str, dict[str, np.ndarray | None]] = {}

        # Create minimal GeffMetadata
        metadata = create_or_update_metadata(metadata=None, is_directed=True)

        # Create metadata for all node properties
        node_props_metadata = [
            create_props_metadata(identifier=prop_name, prop_data=prop_data)
            for prop_name, prop_data in node_props.items()
        ]
        metadata = add_or_update_props_metadata(
            metadata, node_props_metadata, c_type="node"
        )

        # Set track_node_props if we have track_id or lineage_id
        track_node_props = {}
        if "track_id" in node_props:
            track_node_props["tracklet"] = "track_id"
        if "lineage_id" in node_props:
            track_node_props["lineage"] = "lineage_id"
        if track_node_props:
            metadata.track_node_props = track_node_props

        # Build InMemoryGeff structure (cast dict to InMemoryGeff type)
        self.in_memory_geff = cast(
            InMemoryGeff,
            {
                "metadata": metadata,
                "node_ids": node_ids,
                "edge_ids": edge_ids,
                "node_props": node_props,
                "edge_props": edge_props,
            },
        )


def import_from_csv(
    csv_path: Path,
    name_map: dict[str, str] | None = None,
    segmentation_path: Path | None = None,
    scale: list[float] | None = None,
    node_features: dict[str, bool] | None = None,
) -> SolutionTracks:
    """Import tracks from CSV file.

    Loads tracking data from CSV format with columns:
    time, [z], y, x, id, parent_id, [seg_id], ...

    Args:
        csv_path: Path to CSV file
        name_map: Optional mapping from standard keys to CSV column names.
                  If None, will auto-infer column mappings using fuzzy matching.
                  Example: {"time": "t", "x": "x_coord", "y": "y_coord"}
        segmentation_path: Optional path to segmentation array
                           (numpy, tiff, or zarr)
        scale: Optional spatial scale including time dimension
               (e.g., [1.0, 1.0, 0.5, 0.5])
        node_features: Optional dict mapping feature names to recompute flag.
                      False = load from CSV, True = recompute from segmentation
                      Example: {"area": False, "circularity": True}

    Returns:
        SolutionTracks object with graph and optional segmentation

    Example:
        >>> tracks = import_from_csv(
        ...     "tracks.csv",
        ...     name_map={"time": "frame", "x": "x_pos", "y": "y_pos"},
        ...     segmentation_path="seg.tif"
        ... )
    """
    builder = CSVTracksBuilder()

    if name_map is None:
        # Auto-infer name mapping from CSV headers
        builder.prepare(csv_path)
    else:
        # Use provided name mapping
        builder.read_header(csv_path)
        builder.name_map = name_map

    return builder.build(
        csv_path,
        segmentation_path,
        scale=scale,
        node_features=node_features,
    )
