"""Builder pattern for importing tracks from various formats.

This module provides a unified interface for constructing SolutionTracks objects
from different data sources (GEFF, CSV, etc.) while sharing common validation
and construction logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import geff
import networkx as nx
import numpy as np
from geff._typing import InMemoryGeff

from funtracks.data_model.graph_attributes import NodeAttr
from funtracks.data_model.solution_tracks import SolutionTracks
from funtracks.features import Feature
from funtracks.import_export._import_segmentation import (
    load_segmentation,
    read_dims,
    relabel_segmentation,
)
from funtracks.import_export._name_mapping import (
    infer_edge_name_map,
    infer_node_name_map,
)
from funtracks.import_export._utils import (
    get_default_key_to_feature_mapping,
    infer_dtype_from_array,
)
from funtracks.import_export._validation import (
    validate_edge_name_map,
    validate_feature_key_collisions,
    validate_in_memory_geff,
    validate_node_name_map,
    validate_spatial_dims,
)

if TYPE_CHECKING:
    import pandas as pd
    from numpy.typing import ArrayLike


def flatten_name_map(
    name_map: dict[str, str | list[str]],
) -> list[tuple[str, str]]:
    """Flatten a name_map to a list of (target_key, source_key) tuples.

    For single-value mappings like {"time": "t"}, returns [("time", "t")] (renaming).
    For multi-value mappings like {"pos": ["y", "x"]}, returns [("y", "y"), ("x", "x")]
    (keep original names for later combining by TracksBuilder._combine_multi_value_props).

    Args:
        name_map: Mapping from standard keys to source property names

    Returns:
        List of (target_key, source_key) tuples
    """
    result = []
    for std_key, source in name_map.items():
        if source is None:
            continue
        if isinstance(source, list):
            # Multi-value: keep original column names (combining happens later)
            for col in source:
                result.append((col, col))
        else:
            # Single-value: rename from source to standard key
            result.append((std_key, source))
    return result


# defining constants here because they are only used in the context of import
TRACK_KEY = "track_id"
SEG_KEY = "seg_id"


class TracksBuilder(ABC):
    """Abstract builder for importing tracks from various formats.

    Defines the construction steps that all format-specific builders must implement,
    along with common logic shared across formats.
    """

    TIME_ATTR = "time"

    def __init__(self):
        """Initialize builder state."""
        # State transferred between steps
        self.in_memory_geff: InMemoryGeff | None = None
        self.ndim: int | None = None

        # Name maps: {standard_key -> source_property_name(s)}
        # Keys are standard funtracks attribute names (e.g., "time", "pos", "seg_id")
        # Values are property names from the source data (e.g., "t", ["y", "x"], "label")
        # For multi-value features like position, the value is a list of source columns
        self.node_name_map: dict[str, str | list[str]] = {}
        self.edge_name_map: dict[str, str | list[str]] | None = None

        # Available properties in the source data (populated by read_header)
        self.importable_node_props: list[str] = []
        self.importable_edge_props: list[str] = []

        # Builder configuration
        self.required_features = ["time"]
        self.available_computed_features = get_default_key_to_feature_mapping(
            self.ndim, display_name=False
        )

    @property
    def axis_names(self) -> list[str]:
        """Position attribute names derived from ndim.

        Returns ["z", "y", "x"] for 3D (ndim=4) or ["y", "x"] for 2D (ndim=3).
        If ndim is None, returns ["z", "y", "x"] as default.
        """
        if self.ndim is None or self.ndim == 4:
            return ["z", "y", "x"]
        return ["y", "x"]

    @abstractmethod
    def read_header(self, source: Path | pd.DataFrame) -> None:
        """Read metadata/headers from source without loading data.

        Should populate self.importable_node_props and
        self.importable_edge_props with property/column names.

        Args:
            source: Path to data source (zarr store, CSV file, etc.) or DataFrame
        """

    def infer_node_name_map(self) -> dict[str, str | list[str]]:
        """Infer node_name_map by matching source properties to standard keys.

        The node_name_map maps standard funtracks keys to source property names:
            {standard_key: source_property_name}

        For example: {"time": "t", "pos": ["y", "x"], "seg_id": "label"}
        - "time", "pos", "seg_id" are standard funtracks keys
        - "t", "y", "x", "label" are property names from the source data

        Uses difflib fuzzy matching with the following priority:
        1. Exact matches to standard keys (time, seg_id, etc.)
        2. Fuzzy matches to standard keys (case-insensitive, 40% similarity cutoff)
        3. Exact matches to feature display names/value_names (including position z/y/x)
        4. Fuzzy matches to feature display names (case-insensitive, 40% cutoff)
        5. Remaining properties map to themselves (custom properties)

        Position attributes (z, y, x) are matched via Position feature's value_names,
        resulting in a composite mapping like {"pos": ["z", "y", "x"]}.

        Returns:
            Inferred node_name_map mapping standard keys to source property names

        Raises:
            ValueError: If required features cannot be inferred
        """
        return infer_node_name_map(
            self.importable_node_props,
            self.required_features,
            self.available_computed_features,
        )

    def infer_edge_name_map(self) -> dict[str, str | list[str]]:
        """Infer edge_name_map by matching source properties to standard keys.

        The edge_name_map maps standard funtracks keys to source property names:
            {standard_key: source_property_name}

        For example: {"iou": "overlap"}
        - "iou" is the standard funtracks key
        - "overlap" is the property name from the source data

        Uses difflib fuzzy matching with the following priority:
        1. Exact matches to edge feature default keys
        2. Fuzzy matches to edge feature default keys (case-insensitive, 40%
           similarity cutoff)
        3. Exact matches to edge feature display names
        4. Fuzzy matches to edge feature display names (case-insensitive,
           40% cutoff)
        5. Remaining properties map to themselves (custom properties)

        Returns:
            Inferred edge_name_map mapping standard keys to source property names
        """
        if not self.importable_edge_props:
            return {}

        return infer_edge_name_map(
            self.importable_edge_props, self.available_computed_features
        )

    def _preprocess_name_map(self) -> None:
        """Preprocess name maps to remove empty lists and None values.

        Empty lists and None values have no semantic meaning and should be
        treated the same as being absent from the mapping.

        Modifies self.node_name_map and self.edge_name_map in place.
        """
        # Remove empty lists and None values from node_name_map
        keys_to_remove = [
            k for k, v in self.node_name_map.items() if v is None or v == []
        ]
        for k in keys_to_remove:
            del self.node_name_map[k]

        # Remove empty lists and None values from edge_name_map
        if self.edge_name_map is not None:
            keys_to_remove = [
                k for k, v in self.edge_name_map.items() if v is None or v == []
            ]
            for k in keys_to_remove:
                del self.edge_name_map[k]

    def validate_name_map(self, has_segmentation: bool = False) -> None:
        """Validate that node_name_map and edge_name_map contain valid mappings.

        Checks for nodes:
        - No None values in required mappings
        - All required_features are mapped
        - Position ("pos") is mapped to coordinate columns (unless segmentation provided)
        - All mapped properties exist in importable_node_props
        - Features with spatial_dims=True have correct number of list elements

        Checks for edges:
        - All mapped edge properties exist in importable_edge_props

        Checks for both:
        - No feature key collisions between node and edge features

        Note: Array shapes for spatial_dims features are validated after loading
        via validate_spatial_dims().

        Args:
            has_segmentation: If True, position can be computed from segmentation
                and is not required in name_map

        Raises:
            ValueError: If validation fails
        """
        # Preprocess: remove empty lists from name maps
        self._preprocess_name_map()

        # Validate node_name_map (includes spatial_dims validation for list mappings)
        validate_node_name_map(
            self.node_name_map,
            self.importable_node_props,
            self.required_features,
            available_features=self.available_computed_features,
            ndim=self.ndim,
            has_segmentation=has_segmentation,
        )

        # Validate edge_name_map if provided (includes spatial_dims validation)
        if self.edge_name_map is not None:
            validate_edge_name_map(
                self.edge_name_map,
                self.importable_edge_props,
                available_features=self.available_computed_features,
                ndim=self.ndim,
            )

        # Check for feature key collisions between nodes and edges
        validate_feature_key_collisions(self.node_name_map, self.edge_name_map)

    def prepare(
        self,
        source: Path | pd.DataFrame,
        segmentation: Path | ArrayLike | None = None,
    ) -> None:
        """Prepare for building by reading headers and inferring name maps.

        This method reads the data source headers/metadata and automatically
        infers both node_name_map and edge_name_map. After calling this, you can
        inspect and modify self.node_name_map and self.edge_name_map before calling
        build().

        Args:
            source: Path to data source or DataFrame
            segmentation: Optional path to segmentation or array to infer ndim

        Example:
            >>> builder = CSVTracksBuilder()
            >>> builder.prepare("data.csv")
            >>> # Optionally modify the inferred mappings
            >>> builder.node_name_map["circularity"] = "circ"
            >>> builder.edge_name_map["iou"] = "overlap"
            >>> tracks = builder.build("data.csv", segmentation_path="seg.tif")
        """
        self.read_header(source)
        if segmentation is not None:
            self.ndim = read_dims(segmentation)
        self.node_name_map = self.infer_node_name_map()
        self.edge_name_map = self.infer_edge_name_map()

    @abstractmethod
    def load_source(
        self,
        source: Path | pd.DataFrame,
        node_name_map: dict[str, str | list[str]],
        node_features: dict[str, bool] | None = None,
    ) -> None:
        """Load data from source file and convert to InMemoryGeff format.

        Should populate self.in_memory_geff with all properties using standard keys.

        Args:
            source: Path to data source (zarr store, CSV file, etc.) or DataFrame
            node_name_map: Maps standard keys to source property names
            node_features: Optional features dict for backward compatibility
        """

    def _combine_multi_value_props(
        self,
        props: dict,
        name_map: dict[str, str | list[str]],
    ) -> None:
        """Combine multi-value feature columns into single properties.

        For features mapped to a list of columns (e.g., "pos": ["y", "x"]),
        combines those columns into a single property with stacked values.

        Args:
            props: Property dict from InMemoryGeff (node_props or edge_props)
            name_map: Mapping from standard keys to source property names

        Modifies props in place.
        """
        for std_key, source_cols in name_map.items():
            if not isinstance(source_cols, list) or len(source_cols) == 0:
                continue
            # Check all source columns exist
            missing_cols = [c for c in source_cols if c not in props]
            if missing_cols:
                continue  # Skip if any columns are missing
            # Stack column values into 2D array
            col_arrays = [props[c]["values"] for c in source_cols]
            combined = np.column_stack(col_arrays)
            # Combine missing arrays with OR (missing if any component is missing)
            missing_arrays = [props[c].get("missing") for c in source_cols]
            if any(m is not None for m in missing_arrays):
                combined_missing = np.zeros(len(combined), dtype=np.bool_)
                for m in missing_arrays:
                    if m is not None:
                        combined_missing |= m
            else:
                combined_missing = None
            props[std_key] = {
                "values": combined,
                "missing": combined_missing,
            }
            # Remove the individual source columns
            for c in source_cols:
                if c in props and c != std_key:
                    del props[c]

    def validate(self) -> None:
        """Validate the loaded InMemoryGeff data.

        Common validation logic shared across all formats.
        Validates:
        - Graph structure (unique nodes, valid edges, etc.)
        - Spatial_dims features have correct array shapes
        - Optional properties (lineage_id, track_id) - removed with warning if invalid

        Raises:
            ValueError: If required validation fails
        """
        if self.in_memory_geff is None:
            raise ValueError("No data loaded. Call load_source() first.")

        # Validate spatial_dims features have correct array shapes
        validate_spatial_dims(
            self.in_memory_geff,
            self.available_computed_features,
            ndim=self.ndim,
        )

        # Validate graph structure and optional properties
        validate_in_memory_geff(self.in_memory_geff)

    def construct_graph(self) -> nx.DiGraph:
        """Construct NetworkX graph from validated InMemoryGeff data.

        Common logic shared across all formats.

        Returns:
            NetworkX DiGraph with standard keys

        Raises:
            ValueError: If data not loaded or validated
        """
        if self.in_memory_geff is None:
            raise ValueError("No data loaded. Call load_source() first.")
        return geff.construct(**self.in_memory_geff)

    def handle_segmentation(
        self,
        graph: nx.DiGraph,
        segmentation: Path | np.ndarray | None,
        scale: list[float] | None,
    ) -> tuple[np.ndarray | None, list[float] | None]:
        """Load, validate, and optionally relabel segmentation.

        Common logic shared across all formats.

        Args:
            graph: Constructed NetworkX graph for validation
            segmentation: Path to segmentation data or pre-loaded segmentation array
            scale: Spatial scale for coordinate transformation

        Returns:
            Tuple of (segmentation array, scale) or (None, scale)

        Raises:
            ValueError: If segmentation validation fails
        """
        if segmentation is None:
            return None, scale

        if self.in_memory_geff is None:
            raise ValueError("No data loaded. Call load_source() first.")

        # Load segmentation from path or wrap array
        seg_array = load_segmentation(segmentation)

        # Validate dimensions match graph
        if seg_array.ndim != self.ndim:
            raise ValueError(
                f"Segmentation has {seg_array.ndim} dimensions but graph has "
                f"{self.ndim} dimensions"
            )

        # Default scale to 1.0 for each axis if not provided
        if scale is None:
            scale = [1.0] * self.ndim

        # Validate segmentation matches graph (only if position is loaded)
        # If position is not in graph, it will be computed from segmentation
        sample_node = next(iter(graph.nodes()))
        has_position = "pos" in graph.nodes[sample_node]
        if has_position:
            from funtracks.import_export._validation import validate_graph_seg_match

            validate_graph_seg_match(graph, seg_array, scale, self.axis_names)

        # Check if relabeling is needed (seg_id != node_id)
        node_props = self.in_memory_geff["node_props"]
        if "seg_id" not in node_props:
            # No seg_id property, assume segmentation labels match node IDs
            return seg_array.compute(), scale

        node_ids = self.in_memory_geff["node_ids"]
        seg_ids = node_props["seg_id"]["values"]

        # Check if any seg_id differs from node_id
        if np.array_equal(seg_ids, node_ids):
            # No relabeling needed
            return seg_array.compute(), scale

        # Relabel segmentation: seg_id -> node_id
        time_values = node_props[NodeAttr.TIME.value]["values"]
        new_segmentation = relabel_segmentation(
            seg_array, graph, node_ids, seg_ids, time_values
        )

        return new_segmentation, scale

    def enable_features(
        self,
        tracks: SolutionTracks,
        features: dict[str, bool] | None,
        feature_type: Literal["node", "edge"] = "node",
    ) -> None:
        """Enable and register features on tracks object.

        Common logic shared across all formats for both node and edge features.

        Args:
            tracks: SolutionTracks object to add features to
            features: Dict mapping feature names to recompute flags
            feature_type: Type of features ("node" or "edge")
        """
        if features is None:
            return

        if self.in_memory_geff is None:
            raise ValueError("No data loaded. Call load_source() first.")

        # Get the appropriate props dict based on feature_type
        props = (
            self.in_memory_geff["node_props"]
            if feature_type == "node"
            else self.in_memory_geff["edge_props"]
        )

        # Validate requested features before enabling
        invalid_features = []
        for key, recompute in features.items():
            if recompute:
                # Features to compute must exist in annotators
                if key not in tracks.annotators.all_features:
                    invalid_features.append(key)
            else:
                # Features to load must exist in props
                if key not in props:
                    invalid_features.append(key)

        if invalid_features:
            available_computed = list(tracks.annotators.all_features.keys())
            available_geff = list(props.keys())
            raise KeyError(
                f"{feature_type.capitalize()} features not available: "
                f"{invalid_features}. "
                f"Available computed features: {available_computed}. "
                f"Available {feature_type} properties: {available_geff}"
            )

        # Separate into features that exist in annotators vs static features
        annotator_features = {
            key: recompute
            for key, recompute in features.items()
            if key in tracks.annotators.all_features
        }

        # Enable annotator features with appropriate recompute flag
        for key, recompute in annotator_features.items():
            tracks.enable_features([key], recompute=recompute)

        # Register static features (features not in annotator registry)
        static_keys = [key for key in features if key not in annotator_features]
        static_features: dict[str, Feature] = {}
        for key in static_keys:
            static_features[key] = Feature(
                display_name=key,
                feature_type=feature_type,
                value_type=infer_dtype_from_array(props[key]["values"]),
                num_values=1,
                required=False,
                default_value=None,
            )
        tracks.features.update(static_features)

    def build(
        self,
        source: Path | pd.DataFrame,
        segmentation: Path | np.ndarray | None = None,
        scale: list[float] | None = None,
        node_features: dict[str, bool] | None = None,
        edge_features: dict[str, bool] | None = None,
    ) -> SolutionTracks:
        """Orchestrate the full construction process.

        Args:
            source: Path to data source or DataFrame
            segmentation: Optional path to segmentation or pre-loaded segmentation array
            scale: Optional spatial scale
            node_features: Optional node features to enable/load
            edge_features: Optional edge features to enable/load

        Returns:
            Fully constructed SolutionTracks object

        Raises:
            ValueError: If self.node_name_map is not set or validation fails

        Example:
            >>> # Using prepare() to auto-infer node_name_map
            >>> builder = CSVTracksBuilder()
            >>> builder.prepare("data.csv")
            >>> tracks = builder.build("data.csv")
            >>>
            >>> # Or set node_name_map manually
            >>> builder = CSVTracksBuilder()
            >>> builder.read_header("data.csv")
            >>> builder.node_name_map = {"time": "t", "x": "x", "y": "y", "id": "id"}
            >>> tracks = builder.build("data.csv")
        """
        # Validate we have a node_name_map
        if not self.node_name_map:
            raise ValueError(
                "self.node_name_map must be set before calling build(). "
                "Call prepare() to auto-infer or set manually."
            )

        # Set ndim early - from segmentation if provided, otherwise from position mapping
        # This value is used for all subsequent validation and should not change
        if self.ndim is None:
            if segmentation is not None:
                self.ndim = read_dims(segmentation)
            elif "pos" in self.node_name_map:
                pos_mapping = self.node_name_map["pos"]
                if isinstance(pos_mapping, list):
                    self.ndim = len(pos_mapping) + 1  # +1 for time

            # Regenerate available_computed_features with correct ndim
            if self.ndim is not None:
                self.available_computed_features = get_default_key_to_feature_mapping(
                    self.ndim, display_name=False
                )

        # Validate node_name_map is complete and valid
        self.validate_name_map(has_segmentation=segmentation is not None)

        # 1. Load source data to InMemoryGeff
        self.load_source(source, self.node_name_map, node_features)
        if self.in_memory_geff is None:
            raise ValueError("load_source() must populate self.in_memory_geff")

        # 2. Combine multi-value feature columns
        self._combine_multi_value_props(
            self.in_memory_geff["node_props"], self.node_name_map
        )
        if self.edge_name_map is not None:
            self._combine_multi_value_props(
                self.in_memory_geff["edge_props"], self.edge_name_map
            )

        # 3. Validate InMemoryGeff (includes spatial_dims array shape validation)
        self.validate()

        # 4. Construct graph
        graph = self.construct_graph()

        # 5. Handle segmentation
        segmentation_array, scale = self.handle_segmentation(graph, segmentation, scale)

        # 6. Create SolutionTracks
        tracks = SolutionTracks(
            graph=graph,
            segmentation=segmentation_array,
            pos_attr="pos",
            time_attr=self.TIME_ATTR,
            ndim=self.ndim,
            scale=scale,
        )

        # 7. Enable and register features
        if node_features is not None:
            self.enable_features(tracks, node_features, feature_type="node")
        if edge_features is not None:
            self.enable_features(tracks, edge_features, feature_type="edge")

        return tracks
