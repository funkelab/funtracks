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

if TYPE_CHECKING:
    import pandas as pd
    from numpy.typing import ArrayLike

from funtracks.data_model.graph_attributes import NodeAttr
from funtracks.data_model.solution_tracks import SolutionTracks
from funtracks.features import Feature
from funtracks.import_export._import_segmentation import (
    load_segmentation,
    read_dims,
    relabel_segmentation,
)
from funtracks.import_export._name_mapping import infer_edge_name_map, infer_name_map
from funtracks.import_export._utils import (
    get_default_key_to_feature_mapping,
    infer_dtype_from_array,
)
from funtracks.import_export._validation import (
    validate_in_memory_geff,
    validate_name_map,
)

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
        self.axis_names: list[str] = ["z", "y", "x"]
        self.ndim: int | None = None
        # TODO: self.node_name_map instead of self.name_map
        # TODO: how much of the node/edge stuff can we abstract?
        self.name_map: dict[str, str] = {}
        self.edge_name_map: dict[str, str] | None = None
        self.importable_node_props: list[str] = []
        self.importable_edge_props: list[str] = []

        # Builder configuration
        self.required_features = ["time"]
        self.available_computed_features = get_default_key_to_feature_mapping(
            self.ndim, display_name=False
        )

    @abstractmethod
    def read_header(self, source: Path | pd.DataFrame) -> None:
        """Read metadata/headers from source without loading data.

        Should populate self.importable_node_props and
        self.importable_edge_props with property/column names.

        Args:
            source: Path to data source (zarr store, CSV file, etc.) or DataFrame
        """

    # TODO: infer_node_name_map
    def infer_name_map(self) -> dict[str, str]:
        """Infer name_map by matching importable node properties to standard keys.

        Uses difflib fuzzy matching with the following priority:
        1. Exact matches to standard keys
        2. Fuzzy matches to standard keys (case-insensitive, 40% similarity cutoff)
        3. Exact matches to feature display names
        4. Fuzzy matches to feature display names (case-insensitive, 40% cutoff)
        5. Remaining properties map to themselves (custom properties)

        Returns:
            Inferred name_map (standard_key -> source_property)

        Raises:
            ValueError: If required features cannot be inferred
        """
        return infer_name_map(
            self.importable_node_props,
            self.required_features,
            self.axis_names,
            self.ndim,
            self.available_computed_features,
        )

    def infer_edge_name_map(self) -> dict[str, str]:
        """Infer edge_name_map by matching importable edge properties to standard keys.

        Uses difflib fuzzy matching with the following priority:
        1. Exact matches to edge feature default keys
        2. Fuzzy matches to edge feature default keys (case-insensitive, 40%
           similarity cutoff)
        3. Exact matches to edge feature display names
        4. Fuzzy matches to edge feature display names (case-insensitive,
           40% cutoff)
        5. Remaining properties map to themselves (custom properties)

        Returns:
            Inferred edge_name_map (standard_key -> source_property)
        """
        if not self.importable_edge_props:
            return {}

        return infer_edge_name_map(
            self.importable_edge_props, self.available_computed_features
        )

    # TODO: validate_node_name_map, validate_edge_name_map?
    def validate_name_map(self) -> None:
        """Validate that name_map and edge_name_map contain valid mappings.

        Checks for nodes:
        - No None values in required mappings
        - No duplicate values in required mappings
        - All required_features are mapped
        - All position_attr are mapped (based on ndim)
        - All mapped properties exist in importable_node_props

        Checks for edges:
        - All mapped edge properties exist in importable_edge_props

        Checks for both:
        - No feature key collisions between node and edge features

        Raises:
            ValueError: If validation fails or ndim is not set
        """
        if self.ndim is None:
            raise ValueError("ndim must be set before validating name_map")
        validate_name_map(
            self.name_map,
            self.importable_node_props,
            self.required_features,
            self.axis_names,
            self.ndim,
            edge_name_map=self.edge_name_map,
            importable_edge_props=self.importable_edge_props,
        )

    def prepare(
        self,
        source: Path | pd.DataFrame,
        segmentation: Path | ArrayLike | None = None,
    ) -> None:
        """Prepare for building by reading headers and inferring name maps.

        This method reads the data source headers/metadata and automatically
        infers both name_map and edge_name_map. After calling this, you can
        inspect and modify self.name_map and self.edge_name_map before calling
        build().

        Args:
            source: Path to data source or DataFrame
            segmentation: Optional path to segmentation or array to infer ndim

        Example:
            >>> builder = CSVTracksBuilder()
            >>> builder.prepare("data.csv")
            >>> # Optionally modify the inferred mappings
            >>> builder.name_map["circularity"] = "circ"
            >>> builder.edge_name_map["iou"] = "overlap"
            >>> tracks = builder.build("data.csv", segmentation_path="seg.tif")
        """
        self.read_header(source)
        if segmentation is not None:
            self.ndim = read_dims(segmentation)
        self.name_map = self.infer_name_map()
        self.edge_name_map = self.infer_edge_name_map()

    @abstractmethod
    def load_source(
        self,
        source: Path | pd.DataFrame,
        name_map: dict[str, str],
        node_features: dict[str, bool] | None = None,
    ) -> None:
        """Load data from source file and convert to InMemoryGeff format.

        Should populate self.in_memory_geff with all properties using standard keys.

        Args:
            source: Path to data source (zarr store, CSV file, etc.) or DataFrame
            name_map: Maps standard keys to source property names
            node_features: Optional features dict for backward compatibility
        """

    def validate(self) -> None:
        """Validate the loaded InMemoryGeff data.

        Common validation logic shared across all formats.
        If optional properties (lineage_id, track_id) fail validation,
        they are removed with a warning.

        Raises:
            ValueError: If required validation (graph structure) fails
        """
        if self.in_memory_geff is None:
            raise ValueError("No data loaded. Call load_source() first.")

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

        # Validate segmentation matches graph
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
            ValueError: If self.name_map is not set or validation fails

        Example:
            >>> # Using prepare() to auto-infer name_map
            >>> builder = CSVTracksBuilder()
            >>> builder.prepare("data.csv")
            >>> tracks = builder.build("data.csv")
            >>>
            >>> # Or set name_map manually
            >>> builder = CSVTracksBuilder()
            >>> builder.read_header("data.csv")
            >>> builder.name_map = {"time": "t", "x": "x", "y": "y", "id": "id"}
            >>> tracks = builder.build("data.csv")
        """
        # Validate we have a name_map
        if not self.name_map:
            raise ValueError(
                "self.name_map must be set before calling build(). "
                "Call prepare() to auto-infer or set manually."
            )

        # Validate name_map is complete and valid
        self.validate_name_map()

        # 1. Load source data to InMemoryGeff
        self.load_source(source, self.name_map, node_features)

        # 2. Validate InMemoryGeff
        self.validate()

        # 3. Construct graph
        graph = self.construct_graph()

        # 4. Handle segmentation
        segmentation_array, scale = self.handle_segmentation(graph, segmentation, scale)

        # 5. Create SolutionTracks
        tracks = SolutionTracks(
            graph=graph,
            segmentation=segmentation_array,
            pos_attr=self.axis_names,
            time_attr=self.TIME_ATTR,
            ndim=self.ndim,
            scale=scale,
        )

        # 6. Enable and register features
        if node_features is not None:
            self.enable_features(tracks, node_features, feature_type="node")
        if edge_features is not None:
            self.enable_features(tracks, edge_features, feature_type="edge")

        return tracks
