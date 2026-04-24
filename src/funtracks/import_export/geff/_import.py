from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from geff._typing import InMemoryGeff
from geff.core_io._base_read import read_to_memory

from funtracks.annotators import RegionpropsAnnotator

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

    All property keys are renamed before Tracksdata graph construction.

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
        import warnings

        import zarr as _zarr
        from geff_spec import GeffMetadata

        metadata = GeffMetadata.read(source_path)

        # Extract property names from metadata
        self.importable_node_props = list(metadata.node_props_metadata.keys())
        self.importable_edge_props = list(metadata.edge_props_metadata.keys())

        # Store axes metadata for use in infer_node_name_map
        self._geff_axes = metadata.axes or []

        # Read segmentation_shape written by export_to_geff (stored as an extra
        # zarr attribute alongside the geff metadata).
        # source_path may be a filesystem Path or an in-memory zarr Store,
        # so pass it directly without str() conversion.
        try:
            z = _zarr.open(source_path, mode="r")
            raw = dict(z.attrs).get("segmentation_shape")
        except (OSError, KeyError, ValueError):
            raw = None
        self._segmentation_shape = tuple(raw) if raw is not None else None

        # Warn when masks/bboxes are present but segmentation_shape is absent.
        # This happens with GEFFs written by older funtracks or external tools.
        has_masks = (
            "mask" in self.importable_node_props and "bbox" in self.importable_node_props
        )
        if has_masks and self._segmentation_shape is None:
            warnings.warn(
                "GEFF contains 'mask' and 'bbox' node attributes but no "
                "'segmentation_shape' metadata. The segmentation cannot be "
                "reconstructed. Re-export with an updated version of funtracks "
                "to preserve the segmentation.",
                UserWarning,
                stacklevel=2,
            )

    def infer_node_name_map(self) -> dict[str, str | list[str]]:
        """Derive time and position mapping from geff axes metadata.

        When axes with typed metadata (type="time" / type="space") are present,
        uses them directly instead of falling back to fuzzy string matching, which
        can misassign properties when many non-spatiotemporal properties are present.

        Falls back to the base-class fuzzy matching when axes metadata is absent.

        Returns:
            Inferred node_name_map mapping standard keys to source property names
        """
        import tracksdata as td

        # Tracksdata-internal attributes are added by the builder and should not
        # appear in the node name map (avoids collision with edge-side solution attr).
        internal_attrs = {td.DEFAULT_ATTR_KEYS.SOLUTION}

        geff_axes = getattr(self, "_geff_axes", [])
        if geff_axes:
            time_axes = [ax.name for ax in geff_axes if ax.type == "time"]
            space_axes = [ax.name for ax in geff_axes if ax.type == "space"]

            if time_axes and space_axes:
                axis_props = set(time_axes + space_axes)
                node_name_map: dict[str, str | list[str]] = {
                    "time": time_axes[0],
                    "pos": space_axes,
                }
                # Pass all remaining non-internal properties through unchanged
                for prop in self.importable_node_props:
                    if prop not in axis_props and prop not in internal_attrs:
                        node_name_map[prop] = prop
                return node_name_map

        # Fall back to fuzzy matching when axes metadata is absent or incomplete
        return super().infer_node_name_map()

    def enable_features(
        self,
        tracks: SolutionTracks,
        name_map: dict[str, str | list[str]],
        feature_type: Literal["node", "edge"] = "node",
    ) -> None:
        """Reconstruct embedded segmentation, then delegate to the parent.

        The GEFF format serialises mask data as plain numeric arrays (zarr cannot
        store arbitrary Python objects).  Before enabling node features, this
        override wraps each raw array back into a
        :class:`tracksdata.nodes.Mask` instance and, when a
        ``segmentation_shape`` was written to the zarr attrs during export,
        reconstructs the dense segmentation as a
        :class:`tracksdata.array.GraphArrayView` and registers a
        :class:`~funtracks.annotators.RegionpropsAnnotator`.

        Doing this inside ``enable_features()`` — rather than after
        ``super().build()`` returns — means the annotator is present when the
        parent processes the name map, so multi-value features such as
        ``ellipse_axis_radii`` are registered with their correct metadata
        (``num_values=2``, ``spatial_dims=True``, ``value_names``) rather than
        being silently downgraded to a static scalar feature.
        """
        if feature_type == "node":
            self._setup_embedded_segmentation(tracks)
        super().enable_features(tracks, name_map, feature_type=feature_type)

    def _setup_embedded_segmentation(self, tracks: SolutionTracks) -> None:
        """Reconstruct Mask objects and segmentation from embedded GEFF data.

        Idempotent: guards on ``tracks.segmentation is None`` and the absence of
        an existing :class:`~funtracks.annotators.RegionpropsAnnotator`.
        """
        import tracksdata as td
        from tracksdata.array import GraphArrayView
        from tracksdata.nodes import Mask

        mask_key = td.DEFAULT_ATTR_KEYS.MASK
        bbox_key = td.DEFAULT_ATTR_KEYS.BBOX
        graph = tracks.graph

        if (
            mask_key not in graph.node_attr_keys()
            or bbox_key not in graph.node_attr_keys()
        ):
            return

        df = graph.node_attrs(attr_keys=[mask_key, bbox_key])
        node_ids = list(graph.node_ids())
        nodes_to_update = []
        new_masks = []
        for node_id, mask_val, bbox_val in zip(
            node_ids, df[mask_key], df[bbox_key], strict=True
        ):
            if not isinstance(mask_val, Mask):
                nodes_to_update.append(node_id)
                new_masks.append(Mask(mask_val.astype(bool), bbox=bbox_val))

        if nodes_to_update:
            graph.update_node_attrs(
                attrs={mask_key: new_masks},
                node_ids=nodes_to_update,
            )

        seg_shape = getattr(self, "_segmentation_shape", None)
        if seg_shape is not None and tracks.segmentation is None:
            graph._update_metadata(segmentation_shape=seg_shape)
            tracks.segmentation = GraphArrayView(
                graph=graph,
                shape=seg_shape,
                attr_key="node_id",
                offset=0,
            )
            if not any(isinstance(a, RegionpropsAnnotator) for a in tracks.annotators):
                pos_key = (
                    tracks.features.position_key
                    if isinstance(tracks.features.position_key, str)
                    else None
                )
                tracks.annotators.append(RegionpropsAnnotator(tracks, pos_key=pos_key))
                tracks._setup_core_computed_features()

    def load_source(
        self,
        source_path: Path,
        node_name_map: dict[str, str | list[str]],
    ) -> None:
        """Load GEFF data and convert to InMemoryGeff format.

        Args:
            source_path: Path to GEFF zarr store
            node_name_map: Maps standard keys to GEFF property names
        """
        # Load GEFF data with renamed properties (returns InMemoryGeff with standard keys)
        self.in_memory_geff, self.position_attr, ndim = import_graph_from_geff(
            source_path, node_name_map, edge_name_map=self.edge_name_map
        )
        # Only set ndim if not already set from segmentation
        if self.ndim is None:
            self.ndim = ndim


def import_from_geff(
    directory: Path,
    node_name_map: dict[str, str | list[str]] | None = None,
    segmentation_path: Path | None = None,
    scale: list[float] | None = None,
    edge_name_map: dict[str, str | list[str]] | None = None,
    database: str | None = None,
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
        edge_name_map: Optional mapping from standard funtracks keys to GEFF
            edge property names. Example: {"iou": "overlap"}
        database: Optional path to a SQLite database file for backing storage.
            If None (default), an in-memory/temp graph is used.

    Returns:
        SolutionTracks object
    """
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
        node_name_map=node_name_map,
        database=database,
    )
