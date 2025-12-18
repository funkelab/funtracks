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

from .._tracks_builder import TracksBuilder, flatten_name_map

if TYPE_CHECKING:
    from funtracks.data_model.solution_tracks import SolutionTracks


def _ensure_integer_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure that the 'id' column in the dataframe contains integer values.

    Args:
        df: A pandas dataframe with columns named "id" and "parent_id"

    Returns:
        pd.DataFrame: The same dataframe with the ids remapped to be unique integers.
            Parent id column is also remapped.
    """
    if not pd.api.types.is_integer_dtype(df["id"]):
        unique_ids = df["id"].unique()
        id_mapping = {
            original_id: new_id for new_id, original_id in enumerate(unique_ids, start=1)
        }
        df["id"] = df["id"].map(id_mapping)
        df["parent_id"] = df["parent_id"].map(id_mapping).astype(pd.Int64Dtype())

    return df


class CSVTracksBuilder(TracksBuilder):
    """Builder for importing tracks from CSV/DataFrame format."""

    def __init__(self):
        """Initialize CSV builder with CSV-specific required features."""
        super().__init__()
        self.required_features.extend(["id", "parent_id"])

    def read_header(self, source: Path | pd.DataFrame) -> None:
        """Read CSV column names.

        Args:
            source: Path to CSV file or DataFrame
        """
        df = pd.read_csv(source, nrows=0) if isinstance(source, Path) else source
        self.importable_node_props = df.columns.tolist()
        self.importable_edge_props = []  # CSV has no edge properties

    def load_source(
        self,
        source: Path | pd.DataFrame,
        node_name_map: dict[str, str | list[str]],
        node_features: dict[str, bool] | None = None,
    ) -> None:
        """Load CSV and convert to InMemoryGeff format.

        Args:
            source: Path to CSV file or DataFrame
            node_name_map: Maps standard keys to CSV column names
            node_features: Optional features dict for backward compatibility
        """
        # Read CSV or use provided DataFrame
        df = (
            pd.read_csv(source)
            if isinstance(source, Path)
            else source.copy()  # Make a copy to avoid modifying original
        )

        # Validate that 'id' column contains unique values
        if "id" in df.columns and not df["id"].is_unique:
            raise ValueError("The 'id' column must contain unique values")

        # Ensure integer IDs (convert string IDs to integers if needed)
        if "id" in df.columns and "parent_id" in df.columns:
            df = _ensure_integer_ids(df)

        # For backward compatibility, extend node_name_map with node_features
        # Only add features that should be loaded (recompute=False)
        extended_name_map = dict(node_name_map)
        if node_features is not None:
            for feature_key, recompute in node_features.items():
                if feature_key not in extended_name_map and not recompute:
                    # Assume feature name in CSV matches standard key
                    extended_name_map[feature_key] = feature_key

        # Build a new DataFrame with standard key names, copying data for each mapping.
        # Multi-value features keep original names (combining happens in TracksBuilder)
        new_df_data = {}
        for target_key, source_col in flatten_name_map(extended_name_map):
            if source_col in df.columns and target_key not in new_df_data:
                new_df_data[target_key] = df[source_col].copy()
        df = pd.DataFrame(new_df_data)

        # Convert NaN to None
        df = df.map(lambda x: None if pd.isna(x) else x)

        # Handle type conversions - lists stored as strings like "[1, 2, 3]"
        for col in df.columns:
            if col not in node_name_map:  # custom attributes
                df[col] = df[col].apply(
                    lambda x: ast.literal_eval(x)
                    if isinstance(x, str) and x.startswith("[") and x.endswith("]")
                    else x
                )

        # Determine dimensionality from position mapping (if not already set)
        if self.ndim is None:
            pos_mapping = node_name_map.get("pos", [])
            if isinstance(pos_mapping, list):
                self.ndim = len(pos_mapping) + 1  # +1 for time
            else:
                # Fallback for legacy separate position keys
                self.ndim = 4 if "z" in df.columns else 3

        # Convert DataFrame to InMemoryGeff format
        df_dict = df.to_dict(orient="list")
        node_ids = np.array(df_dict.pop("id"))
        parent_ids = df_dict.pop("parent_id")

        # Build node_props with GEFF-compatible structure
        # Store position coordinates as individual attributes (z, y, x)
        node_props: dict[str, dict[str, np.ndarray | None]] = {}
        for prop_name, values in df_dict.items():
            node_props[prop_name] = {"values": np.array(values), "missing": None}

        # Extract edge IDs from parent_id column
        edge_tuples = [
            (int(parent_id), int(child_id))
            for parent_id, child_id in zip(parent_ids, node_ids, strict=True)
            if not pd.isna(parent_id) and parent_id != -1
        ]
        # Ensure edge_ids has shape (n, 2) even when empty
        if edge_tuples:
            edge_ids = np.array(edge_tuples)
        else:
            edge_ids = np.empty((0, 2), dtype=np.int64)

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


def tracks_from_df(
    df: pd.DataFrame,
    segmentation: np.ndarray | None = None,
    scale: list[float] | None = None,
    features: dict[str, str] | None = None,
    node_name_map: dict[str, str | list[str]] | None = None,
    name_map: dict[str, str | list[str]] | None = None,  # deprecated
) -> SolutionTracks:
    """Import tracks from pandas DataFrame (motile_tracker-compatible API).

    Turns a pandas DataFrame with columns:
        time, [z], y, x, id, parent_id, [seg_id], [optional custom attr 1], ...
    into a SolutionTracks object.

    Cells without a parent_id will have an empty string or a -1 for the parent_id.

    Args:
        df: A pandas DataFrame containing columns
            time, [z], y, x, id, parent_id, [seg_id], [optional custom attr 1], ...
        segmentation: An optional accompanying segmentation.
            If provided, assumes that the seg_id column in the dataframe exists and
            corresponds to the label ids in the segmentation array. Defaults to None.
        scale: The scale of the segmentation (including the time dimension).
            Defaults to None.
        features: Dict mapping measurement attributes (area, volume) to value that
            specifies a column from which to import. If value equals "Recompute",
            recompute these values instead of importing them from a column.
            Example: {"Area": "area"} loads from column "area"
                     {"Area": "Recompute"} recomputes from segmentation
            Defaults to None.
        node_name_map: Optional mapping from standard funtracks keys to DataFrame
            column names: {standard_key: column_name}.
            For example: {"time": "t", "pos": ["y", "x"], "seg_id": "label"}
            - Keys are standard funtracks attribute names (e.g., "time", "pos", "seg_id")
            - Values are column names from the DataFrame (e.g., "t", "label")
            - For multi-value features like position, use a list: {"pos": ["y", "x"]}
            If None, column names are auto-inferred using fuzzy matching.
        name_map: Deprecated. Use node_name_map instead.

    Returns:
        SolutionTracks: a solution tracks object

    Raises:
        ValueError: if the segmentation IDs in the dataframe do not match the provided
            segmentation

    Example:
        >>> tracks = tracks_from_df(df, segmentation=seg, scale=[1.0, 1.0, 0.5, 0.5])
    """
    from warnings import warn

    # Handle deprecated name_map parameter
    if name_map is not None:
        warn(
            "name_map is deprecated, use node_name_map instead",
            DeprecationWarning,
            stacklevel=2,
        )
        if node_name_map is None:
            node_name_map = name_map

    # Convert features dict from motile_tracker format to funtracks format
    node_features = None
    if features is not None:
        node_features = {}
        # Convert feature keys to lowercase for consistency
        features = {key.lower(): val for key, val in features.items()}
        for feature_key, feature_value in features.items():
            if feature_value == "Recompute":
                # Recompute from segmentation
                node_features[feature_key] = True
            else:
                # Load from column specified by feature_value
                node_features[feature_value] = False

    builder = CSVTracksBuilder()

    if node_name_map is not None:
        builder.read_header(df)
        builder.node_name_map = node_name_map
    else:
        # Auto-infer name mapping from DataFrame columns
        builder.prepare(df)

    return builder.build(
        df,
        segmentation,
        scale=scale,
        node_features=node_features,
    )
