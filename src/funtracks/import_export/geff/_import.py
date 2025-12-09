from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from geff._typing import InMemoryGeff
from geff.core_io._base_read import read_to_memory

from .._tracks_builder import TracksBuilder

if TYPE_CHECKING:
    from pathlib import Path

    from funtracks.data_model.solution_tracks import SolutionTracks


# defining constants here because they are only used in the context of import
TRACK_KEY = "track_id"
SEG_KEY = "seg_id"


def import_graph_from_geff(
    directory: Path,
    node_name_map: dict[str, str | list[str]],
    edge_name_map: dict[str, str | list[str]] | None = None,
) -> tuple[InMemoryGeff, list[str], int]:
    """Load GEFF data and rename property keys to standard names.

    All property keys are renamed before NetworkX graph construction.

    Args:
        directory: Path to GEFF data directory or zarr store
        node_name_map: Maps standard keys to GEFF node property names.
            Required: "time", "y", "x" (and "z" for 3D)
            Optional: "track_id", "seg_id", "lineage_id", custom features
            Example: {"time": "t", "circularity": "circ"}.
            Only properties included here will be loaded.
        edge_name_map: Maps standard keys to GEFF edge property names.
            If None, all edge properties loaded with original names.
            If provided, only specified properties loaded and renamed.
            Example: {"iou": "overlap"}

    Returns:
        (in_memory_geff, position_attr, ndims) where in_memory_geff has
        all properties renamed to standard keys

    Raises:
        ValueError: If node_name_map contains None or duplicate values
    """
    # Build filter of which node properties to load from GEFF
    # Handle both single string values and lists of strings (multi-value features)
    node_prop_filter: set[str] = set()
    for prop in node_name_map.values():
        if prop is not None:
            if isinstance(prop, list):
                node_prop_filter.update(prop)
            else:
                node_prop_filter.add(prop)

    # Build filter of which edge properties to load from GEFF
    # Handle both single string values and lists of strings (multi-value features)
    edge_prop_filter: list[str] | None = None
    if edge_name_map is not None:
        edge_prop_filter = []
        for prop in edge_name_map.values():
            if isinstance(prop, list):
                edge_prop_filter.extend(prop)
            else:
                edge_prop_filter.append(prop)

    in_memory_geff = read_to_memory(
        directory,
        node_props=list(node_prop_filter),
        edge_props=edge_prop_filter,
    )

    # Validate spatiotemporal keys (before renaming, checking GEFF keys)
    # Handle composite "pos" mapping for position coordinates
    spatio_temporal_keys = ["time"]
    if "pos" in node_name_map:
        # Composite position: "pos" -> ["y", "x"] or ["z", "y", "x"]
        spatio_temporal_keys.append("pos")
    else:
        # Legacy separate position keys (for backward compatibility)
        spatio_temporal_keys.extend([k for k in ("z", "y", "x") if k in node_name_map])

    spatio_temporal_map = {
        key: node_name_map[key] for key in spatio_temporal_keys if key in node_name_map
    }
    if any(v is None for v in spatio_temporal_map.values()):
        raise ValueError(
            "The node_name_map cannot contain None values. Please provide a valid "
            "mapping for all required fields."
        )

    # Rename node property keys from custom (GEFF) to standard using node_name_map
    # Handle duplicate mappings (e.g., seg_id -> "node_id", id -> "node_id") by
    # copying data for each standard key that maps to the same GEFF property.
    # For multi-value features (list of column names), combine into single property.
    node_props = in_memory_geff["node_props"]
    renamed_node_props = {}
    for std_key, geff_key in node_name_map.items():
        if geff_key is None:
            continue
        # Multi-value features have a list of column names
        if isinstance(geff_key, list):
            # Combine multiple columns into a single multi-value property
            # Check all columns exist
            missing_cols = [c for c in geff_key if c not in node_props]
            if missing_cols:
                continue  # Skip if any columns are missing
            # Stack column values into 2D array (n_nodes, num_values)
            col_arrays = [node_props[c]["values"] for c in geff_key]
            combined = np.column_stack(col_arrays)
            renamed_node_props[std_key] = {
                "values": combined,
                "missing": None,  # TODO: handle missing values properly
            }
        elif geff_key in node_props:
            prop_data = node_props[geff_key]
            # Copy values to avoid aliasing when multiple keys map to same source
            renamed_node_props[std_key] = {
                "values": prop_data["values"].copy(),
                "missing": prop_data.get("missing"),
            }
    in_memory_geff["node_props"] = renamed_node_props

    # Rename edge property keys from custom (GEFF) to standard using edge_name_map
    # Handle duplicate mappings by copying data for each standard key.
    # For multi-value features (list of column names), combine into single property.
    if edge_name_map is not None:
        edge_props = in_memory_geff["edge_props"]
        renamed_edge_props = {}
        for std_key, geff_key in edge_name_map.items():
            if geff_key is None:
                continue
            # Multi-value features have a list of column names
            if isinstance(geff_key, list):
                # Combine multiple columns into a single multi-value property
                missing_cols = [c for c in geff_key if c not in edge_props]
                if missing_cols:
                    continue  # Skip if any columns are missing
                col_arrays = [edge_props[c]["values"] for c in geff_key]
                combined = np.column_stack(col_arrays)
                renamed_edge_props[std_key] = {
                    "values": combined,
                    "missing": None,
                }
            elif geff_key in edge_props:
                prop_data = edge_props[geff_key]
                # Copy values to avoid aliasing when multiple keys map to same source
                renamed_edge_props[std_key] = {
                    "values": prop_data["values"].copy(),
                    "missing": prop_data.get("missing"),
                }
        in_memory_geff["edge_props"] = renamed_edge_props

    # Extract position and compute dimensions (now using standard keys)
    # Handle composite "pos" mapping for position coordinates
    if "pos" in node_name_map:
        # Composite position: "pos" -> ["y", "x"] or ["z", "y", "x"]
        pos_mapping = node_name_map["pos"]
        if isinstance(pos_mapping, list):
            position_attr = pos_mapping  # e.g., ["y", "x"]
            ndims = len(pos_mapping) + 1  # +1 for time
        else:
            # Single value (shouldn't happen, but fallback)
            position_attr = [pos_mapping]
            ndims = 2
    else:
        # Legacy separate position keys (for backward compatibility)
        position_attr = [k for k in ("z", "y", "x") if k in node_name_map]
        ndims = len(position_attr) + 1

    return in_memory_geff, position_attr, ndims


class GeffTracksBuilder(TracksBuilder):
    """Builder for importing tracks from GEFF format."""

    def read_header(self, source_path: Path) -> None:
        """Read GEFF property names without loading arrays.

        Args:
            source_path: Path to GEFF zarr store
        """
        from geff_spec import GeffMetadata

        metadata = GeffMetadata.read(source_path)

        # Extract property names from metadata
        self.importable_node_props = list(metadata.node_props_metadata.keys())
        self.importable_edge_props = list(metadata.edge_props_metadata.keys())

    def load_source(
        self,
        source_path: Path,
        node_name_map: dict[str, str | list[str]],
        node_features: dict[str, bool] | None = None,
    ) -> None:
        """Load GEFF data and convert to InMemoryGeff format.

        Args:
            source_path: Path to GEFF zarr store
            node_name_map: Maps standard keys to GEFF property names
            node_features: Optional features dict (handled by import_graph_from_geff)
        """
        # For backward compatibility, extend node_name_map with node_features
        # Only add features that should be loaded (recompute=False)
        extended_name_map = dict(node_name_map)
        if node_features is not None:
            for feature_key, recompute in node_features.items():
                if feature_key not in extended_name_map and not recompute:
                    # Assume feature name in GEFF matches standard key
                    extended_name_map[feature_key] = feature_key

        # Load GEFF data with renamed properties (returns InMemoryGeff with standard keys)
        self.in_memory_geff, self.position_attr, self.ndim = import_graph_from_geff(
            source_path, extended_name_map, edge_name_map=self.edge_name_map
        )


def import_from_geff(
    directory: Path,
    node_name_map: dict[str, str | list[str]] | None = None,
    segmentation_path: Path | None = None,
    scale: list[float] | None = None,
    node_features: dict[str, bool] | None = None,
    edge_features: dict[str, bool] | None = None,
    extra_features: dict[str, bool] | None = None,
    edge_name_map: dict[str, str | list[str]] | None = None,
    edge_prop_filter: list[str] | None = None,
    name_map: dict[str, str | list[str]] | None = None,  # deprecated
) -> SolutionTracks:
    """Import tracks from GEFF format.

    Args:
        directory: Path to GEFF zarr store
        node_name_map: Maps standard keys to GEFF property names
        segmentation_path: Optional path to segmentation data
        scale: Optional spatial scale
        node_features: Optional node features to enable/load
        edge_features: Optional edge features to enable/load
        extra_features: (Deprecated) Use node_features instead. Kept for
            backward compatibility.
        edge_name_map: Optional mapping for edge property names
        edge_prop_filter: (Deprecated) Use edge_name_map instead. Kept for
            backward compatibility.
        name_map: Deprecated. Use node_name_map instead.

    Returns:
        SolutionTracks object

    Raises:
        ValueError: If both node_features and extra_features are provided
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

    # Handle backward compatibility: extra_features -> node_features
    if extra_features is not None and node_features is not None:
        raise ValueError(
            "Cannot specify both 'node_features' and 'extra_features'. "
            "Please use 'node_features' (extra_features is deprecated)."
        )
    if extra_features is not None:
        node_features = extra_features

    # Handle backward compatibility: edge_prop_filter -> edge_name_map
    # edge_prop_filter was a list of property names to load
    # edge_name_map is a dict mapping standard keys to GEFF property names
    # If edge_prop_filter is provided, convert it to edge_name_map format
    if edge_prop_filter is not None and edge_name_map is None:
        # Map each property to itself (no renaming, just filtering)
        edge_name_map = {prop: prop for prop in edge_prop_filter}

    # Filter out None values and "None" strings from node_name_map
    # (e.g., {"lineage_id": None} or {"lineage_id": "None"})
    if node_name_map is not None:
        node_name_map = {
            k: v for k, v in node_name_map.items() if v is not None and v != "None"
        }

    # Filter edge_name_map as well
    if edge_name_map is not None:
        edge_name_map = {
            k: v for k, v in edge_name_map.items() if v is not None and v != "None"
        }

    builder = GeffTracksBuilder()
    builder.prepare(directory)
    if node_name_map is not None:
        builder.node_name_map = node_name_map
    if edge_name_map is not None:
        builder.edge_name_map = edge_name_map
    return builder.build(
        directory,
        segmentation_path,
        scale=scale,
        node_features=node_features,
        edge_features=edge_features,
    )
