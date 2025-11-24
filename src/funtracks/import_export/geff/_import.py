from __future__ import annotations

from typing import TYPE_CHECKING

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
    node_name_map: dict[str, str],
    edge_name_map: dict[str, str] | None = None,
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
    node_prop_filter: set[str] = {
        prop for _, prop in node_name_map.items() if prop is not None
    }

    # Build filter of which edge properties to load from GEFF
    edge_prop_filter = None if edge_name_map is None else list(edge_name_map.values())

    in_memory_geff = read_to_memory(
        directory,
        node_props=list(node_prop_filter),
        edge_props=edge_prop_filter,
    )

    # Validate spatiotemporal keys (before renaming, checking GEFF keys)
    spatio_temporal_keys = ["time", "z", "y", "x"]
    spatio_temporal_map = {
        key: node_name_map[key] for key in spatio_temporal_keys if key in node_name_map
    }
    if any(v is None for v in spatio_temporal_map.values()):
        raise ValueError(
            "The node_name_map cannot contain None values. Please provide a valid "
            "mapping for all required fields."
        )
    if len(set(spatio_temporal_map.values())) != len(spatio_temporal_map.values()):
        raise ValueError(
            "The node_name_map cannot contain duplicate values. Please provide a unique "
            "mapping for each required field."
        )

    # Rename node property keys from custom (GEFF) to standard using node_name_map
    # Build reverse mapping: GEFF key -> standard key
    node_geff_to_standard = {v: k for k, v in node_name_map.items() if v is not None}

    node_props = in_memory_geff["node_props"]
    renamed_node_props = {}
    for geff_key, prop_data in node_props.items():
        standard_key = node_geff_to_standard.get(geff_key, geff_key)
        renamed_node_props[standard_key] = prop_data
    in_memory_geff["node_props"] = renamed_node_props

    # Rename edge property keys from custom (GEFF) to standard using edge_name_map
    if edge_name_map is not None:
        edge_geff_to_standard = {v: k for k, v in edge_name_map.items() if v is not None}

        edge_props = in_memory_geff["edge_props"]
        renamed_edge_props = {}
        for geff_key, prop_data in edge_props.items():
            standard_key = edge_geff_to_standard.get(geff_key, geff_key)
            renamed_edge_props[standard_key] = prop_data
        in_memory_geff["edge_props"] = renamed_edge_props

    # Extract time and position attributes (now using standard keys)
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
        name_map: dict[str, str],
        node_features: dict[str, bool] | None = None,
    ) -> None:
        """Load GEFF data and convert to InMemoryGeff format.

        Args:
            source_path: Path to GEFF zarr store
            name_map: Maps standard keys to GEFF property names
            node_features: Optional features dict (handled by import_graph_from_geff)
        """
        # For backward compatibility, extend name_map with node_features
        # Only add features that should be loaded (recompute=False)
        extended_name_map = dict(name_map)
        if node_features is not None:
            for feature_key, recompute in node_features.items():
                if feature_key not in extended_name_map and not recompute:
                    # Assume feature name in GEFF matches standard key
                    extended_name_map[feature_key] = feature_key

        # Load GEFF data with renamed properties (returns InMemoryGeff with standard keys)
        self.in_memory_geff, self.position_attr, self.ndim = import_graph_from_geff(
            source_path, extended_name_map
        )


def import_from_geff(
    directory: Path,
    name_map: dict[str, str],
    segmentation_path: Path | None = None,
    scale: list[float] | None = None,
    node_features: dict[str, bool] | None = None,
    edge_name_map: dict[str, str] | None = None,
) -> SolutionTracks:
    builder = GeffTracksBuilder()
    builder.prepare(directory)
    if name_map is None and edge_name_map is not None:
        name_map = edge_name_map
    elif name_map is not None and edge_name_map is not None:
        name_map.update(edge_name_map)
    builder.name_map = name_map
    return builder.build(
        directory, segmentation_path, scale=scale, node_features=node_features
    )
