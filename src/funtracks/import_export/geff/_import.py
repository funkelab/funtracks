from __future__ import annotations

from typing import TYPE_CHECKING

from geff._typing import InMemoryGeff
from geff.core_io._base_read import read_to_memory

from .._tracks_builder import TracksBuilder, flatten_name_map

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
        node_name_map: Mapping from standard funtracks keys to GEFF property names:
            {standard_key: geff_property_name}.
            For example: {"time": "t", "pos": ["y", "x"], "seg_id": "label"}
            - Keys are standard funtracks attribute names (e.g., "time", "pos")
            - Values are property names from the GEFF store (e.g., "t", "label")
            - For multi-value features like position, use a list: {"pos": ["y", "x"]}
            Required keys: "time", "pos" (with spatial coordinates)
            Optional: "seg_id", "tracklet_id", "lineage_id", custom features
            Only properties included here will be loaded.
        edge_name_map: Mapping from standard funtracks keys to GEFF edge property names.
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

    # Rename node properties: copy from source keys to target keys
    # Multi-value features keep original names (combining happens in TracksBuilder)
    node_props = in_memory_geff["node_props"]
    renamed_node_props = {}
    for target_key, source_key in flatten_name_map(node_name_map):
        if source_key in node_props and target_key not in renamed_node_props:
            prop_data = node_props[source_key]
            renamed_node_props[target_key] = {
                "values": prop_data["values"].copy(),
                "missing": prop_data.get("missing"),
            }
    in_memory_geff["node_props"] = renamed_node_props

    # Rename edge properties similarly
    if edge_name_map is not None:
        edge_props = in_memory_geff["edge_props"]
        renamed_edge_props = {}
        for target_key, source_key in flatten_name_map(edge_name_map):
            if source_key in edge_props and target_key not in renamed_edge_props:
                prop_data = edge_props[source_key]
                renamed_edge_props[target_key] = {
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
            # Single value: pos is stored as ndarray property in GEFF
            # Infer spatial dims from the array shape
            position_attr = [pos_mapping]
            pos_array = renamed_node_props["pos"]["values"]
            ndims = pos_array.shape[1] + 1 if pos_array.ndim == 2 else 2
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
        self.in_memory_geff, self.position_attr, ndim = import_graph_from_geff(
            source_path, extended_name_map, edge_name_map=self.edge_name_map
        )
        # Only set ndim if not already set from segmentation
        if self.ndim is None:
            self.ndim = ndim


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
        node_name_map: Optional mapping from standard funtracks keys to GEFF
            property names: {standard_key: geff_property_name}.
            For example: {"time": "t", "pos": ["y", "x"], "seg_id": "label"}
            - Keys are standard funtracks attribute names (e.g., "time", "pos")
            - Values are property names from the GEFF store (e.g., "t", "label")
            - For multi-value features like position, use a list: {"pos": ["y", "x"]}
            If None, property names are auto-inferred using fuzzy matching.
        segmentation_path: Optional path to segmentation data
        scale: Optional spatial scale
        node_features: Optional node features to enable/load
        edge_features: Optional edge features to enable/load
        extra_features: (Deprecated) Use node_features instead. Kept for
            backward compatibility.
        edge_name_map: Optional mapping from standard funtracks keys to GEFF
            edge property names. Example: {"iou": "overlap"}
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
