from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import tracksdata as td
import zarr
from geff._typing import InMemoryGeff
from geff.core_io._base_read import read_to_memory
from geff_spec import GeffMetadata
from tracksdata.nodes import Mask

from .._tracks_builder import TracksBuilder, flatten_name_map

if TYPE_CHECKING:
    from pathlib import Path

    from tracksdata.io._geff import StoreLike

    from funtracks.data_model.tracks import Tracks


# defining constants here because they are only used in the context of import
SEG_KEY = "seg_id"


def read_segmentation_shape(
    source: StoreLike, metadata: GeffMetadata | None = None
) -> tuple[int, ...] | None:
    """Read the segmentation shape recorded in a GEFF store, if any.

    The shape lives in the tracksdata graph metadata (``extra.tracksdata.shape``
    in the GEFF metadata) written by ``export_to_geff``/tracksdata. A legacy fallback
    is implemented for GEFFs written by old versions, checking for the top-level zarr
    attribute "segmentation_shape".

    Args:
        source: Path to a GEFF store, or an open zarr store/group.
        metadata: The GEFF store's already-parsed metadata, if available, to
            avoid re-reading it from `source`.

    Returns:
        The segmentation shape as a tuple, or None if the store has no
        recorded shape.
    """
    metadata = metadata if metadata is not None else GeffMetadata.read(source)
    graph_metadata = td.io.read_graph_metadata(metadata)
    raw = graph_metadata.get("shape")
    if raw is None:
        # source may be a filesystem Path or an in-memory zarr Store, so pass it
        # directly without str() conversion. zarr.open cannot fail here: metadata
        # was already read from source above, so it is a valid, openable store.
        z = zarr.open(source, mode="r")
        raw = dict(z.attrs).get("segmentation_shape")
    return tuple(raw) if raw is not None else None


def has_embedded_segmentation(source: StoreLike) -> bool:
    """Return True if a GEFF store has embedded segmentation that can be
    reconstructed on import.

    Requires both 'mask' and 'bbox' node properties, and a recorded
    segmentation shape (see `read_segmentation_shape`). When both are present,
    `import_from_geff` will reconstruct the segmentation automatically as a
    GraphArrayView, without needing an external segmentation file.

    Args:
        source: Path to a GEFF store, or an open zarr store/group.
    """
    metadata = GeffMetadata.read(source)
    node_props = metadata.node_props_metadata
    has_masks = "mask" in node_props and "bbox" in node_props
    return has_masks and read_segmentation_shape(source, metadata=metadata) is not None


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
        metadata = GeffMetadata.read(source_path)

        # Extract property names from metadata
        self.importable_node_props = list(metadata.node_props_metadata.keys())
        self.importable_edge_props = list(metadata.edge_props_metadata.keys())

        # Store axes metadata for use in infer_node_name_map
        self._geff_axes = metadata.axes or []

        # Read funtracks FeatureDict from GEFF extra metadata if present
        # This will be passed to Tracks via the base build() method
        funtracks_extra = (metadata.extra or {}).get("funtracks")
        has_funtracks_features = bool(funtracks_extra and "features" in funtracks_extra)
        if funtracks_extra is not None and has_funtracks_features:
            try:
                from funtracks.features import FeatureDict

                self.features = FeatureDict.from_json(funtracks_extra["features"])
            except (KeyError, ValueError, TypeError):
                # If FeatureDict loading fails, features will remain None
                pass

        self._shape = read_segmentation_shape(source_path, metadata=metadata)

        # Backward compat: funtracks used to (incorrectly) write segmentation scale
        # into axes.scale instead of graph.metadata["scale"], while pos was already
        # written in world units. Applying axes.scale to pos for such a file would
        # double-scale already-correct positions. Geffs written that way have a
        # funtracks FeatureDict extra but no funtracks "version" string (the version
        # string is new, added alongside this fix): detect that combination and, for
        # those files only, skip scaling pos and instead treat axes.scale as the
        # segmentation scale.
        self._is_legacy_funtracks_geff = has_funtracks_features and "version" not in (
            funtracks_extra or {}
        )
        self._graph_metadata_scale = td.io.read_graph_metadata(metadata).get("scale")

        # Warn when masks/bboxes are present but the shape is absent.
        # This happens with GEFFs written by older funtracks or external tools.
        has_masks = (
            "mask" in self.importable_node_props and "bbox" in self.importable_node_props
        )
        if has_masks and self._shape is None:
            warnings.warn(
                "GEFF contains 'mask' and 'bbox' node attributes but no shape "
                "metadata. The segmentation cannot be reconstructed. Re-export "
                "with an updated version of funtracks or tracksdata to preserve "
                "the segmentation.",
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
                for prop in self.importable_node_props:
                    if prop not in axis_props:
                        node_name_map[prop] = prop
                return node_name_map

        # Fall back to fuzzy matching when axes metadata is absent or incomplete
        return super().infer_node_name_map()

    def _axes_scale(self) -> list[float] | None:
        """Derive the per-dimension scale declared in the geff axes metadata.

        The result is ordered like ``Tracks.scale``: time first, then the spatial
        axes in the same order as ``self.node_name_map["pos"]`` (falling back to
        the order the axes appear in the file).

        Per the geff spec ``scale`` is optional per axis, so any axis without one
        falls back to 1.0. Returns None when there is no axes metadata, when no
        axis declares a scale at all, or when the axes cannot be lined up with the
        graph's dimensions — in those cases the scale is genuinely unknown, and
        callers should be able to tell that apart from unity.

        Returns:
            Scale per dimension (time first), or None if unknown.
        """
        geff_axes = getattr(self, "_geff_axes", [])
        if not geff_axes:
            return None

        scale_by_name = {ax.name: ax.scale for ax in geff_axes}

        # Reconstruct the dimension order used for the graph, not the file order.
        pos_names = self.node_name_map.get("pos")
        ordered: list[str] = []
        time_name = self.node_name_map.get("time")
        if isinstance(time_name, str):
            ordered.append(time_name)
        if isinstance(pos_names, list):
            ordered.extend(pos_names)
        if not ordered or not all(name in scale_by_name for name in ordered):
            # The name map does not line up with the axes metadata (e.g. position
            # stored as a single ndarray property): fall back to the file order.
            ordered = [ax.name for ax in geff_axes]

        # Only trust the result if it covers exactly the graph's dimensions,
        # otherwise Tracks would reject the mismatched length.
        expected_ndim = self.ndim
        if expected_ndim is None and isinstance(pos_names, list):
            expected_ndim = len(pos_names) + 1
        if expected_ndim is not None and len(ordered) != expected_ndim:
            return None

        scales = [scale_by_name.get(name) for name in ordered]
        if all(s is None for s in scales):
            return None
        return [1.0 if s is None else float(s) for s in scales]

    def apply_points_scale(self) -> None:
        """Multiply ``pos`` by the geff axes scale, so it ends up in world units.

        No-op for legacy funtracks GEFFs (see ``read_header``): their axes.scale
        is actually a mislabeled segmentation scale, and their ``pos`` is already
        in world units, so applying it here would double-scale positions.
        """
        if self._is_legacy_funtracks_geff:
            return
        scale = self._axes_scale()
        if scale is None:
            return
        if self.in_memory_geff is None:
            raise ValueError("No data loaded. Call load_source() first.")
        pos = self.in_memory_geff["node_props"].get("pos")
        if pos is None:
            return
        # scale is [time, *spatial]; pos only holds the spatial dims.
        pos["values"] = pos["values"] * np.asarray(scale[1:], dtype=pos["values"].dtype)

    def infer_segmentation_scale(self) -> list[float] | None:
        """Determine ``Tracks.scale`` (segmentation voxel spacing) for this file.

        Reads ``graph.metadata["scale"]`` (see ``read_header``), except for
        legacy funtracks GEFFs, whose segmentation scale was instead stored
        (mislabeled) in the geff axes -- see ``read_header`` for detection.

        Returns:
            Scale per dimension (time first), or None if unknown.
        """
        if self._is_legacy_funtracks_geff:
            return self._axes_scale()
        raw = self._graph_metadata_scale
        return [float(s) for s in raw] if raw is not None else None

    def construct_graph(
        self,
        node_name_map: dict[str, str | list[str]] | None = None,
        database: str | None = None,
    ) -> td.graph.BaseGraph:
        """Construct graph and prepare embedded segmentation data.

        The GEFF format serialises mask data as plain numeric arrays (zarr
        cannot store arbitrary Python objects).  After the base graph is built,
        this override wraps each raw array back into a
        :class:`tracksdata.nodes.Mask` instance and writes the ``shape`` into
        the graph metadata so that
        :class:`~funtracks.data_model.tracks.Tracks.__init__` can reconstruct
        the segmentation and create the
        :class:`~funtracks.annotators.RegionpropsAnnotator` naturally.
        """
        graph = super().construct_graph(node_name_map, database=database)

        mask_key = td.DEFAULT_ATTR_KEYS.MASK
        bbox_key = td.DEFAULT_ATTR_KEYS.BBOX
        has_mask = mask_key in graph.node_attr_keys()
        has_bbox = bbox_key in graph.node_attr_keys()
        if has_mask != has_bbox:
            raise ValueError(
                f"GEFF graph has only one of '{mask_key}'/'{bbox_key}'; "
                "both are required for mask reconstruction."
            )

        if has_mask:
            # Reconstruct Mask objects from raw numeric arrays
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

        # Write the shape into graph metadata so that Tracks.__init__ can
        # reconstruct the segmentation and the RegionpropsAnnotator is created
        # during _get_annotators().
        if self._shape is not None:
            graph.metadata["shape"] = self._shape
            # DEPRECATED: dual-write for motile_tracker, remove later
            graph.metadata["segmentation_shape"] = self._shape

        return graph

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
) -> Tracks:
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
        scale: Optional segmentation voxel scale (``Tracks.scale``) -- this
            only ever sets the segmentation scale, never the points' scale. If
            None, defaults to the scale recorded in the geff's tracksdata graph
            metadata (see :meth:`GeffTracksBuilder.infer_segmentation_scale`),
            and stays None when the file does not declare one. This is
            independent of the geff axes' own ``scale``, which (per the geff
            spec) describes how to convert stored positions to world units and
            is always applied to ``pos`` on import -- regardless of what is
            passed here -- so that funtracks' world-units invariant holds.
        edge_name_map: Optional mapping from standard funtracks keys to GEFF
            edge property names. Example: {"iou": "overlap"}
        database: Optional path to a SQLite database file for backing storage.
            If None (default), an in-memory/temp graph is used.

    Returns:
        Tracks object
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
    # When a FeatureDict was loaded from the GEFF metadata, the attribute
    # names in the graph already match the FeatureDict keys. A user-provided
    # name_map would rename those columns, making them inconsistent with the
    # stored FeatureDict. So we only apply the user's name_map when no
    # FeatureDict was found (i.e. old/external GEFFs).
    has_feature_dict = getattr(builder, "features", None) is not None
    if has_feature_dict and (node_name_map is not None or edge_name_map is not None):
        warnings.warn(
            "Ignoring user-provided name_map because a FeatureDict was "
            "loaded from the GEFF metadata. The stored FeatureDict already "
            "defines the attribute names.",
            UserWarning,
            stacklevel=2,
        )
    if node_name_map is not None and not has_feature_dict:
        builder.node_name_map = node_name_map
    if edge_name_map is not None and not has_feature_dict:
        builder.edge_name_map = edge_name_map

    # An explicit scale always wins; otherwise honour what the file's metadata says.
    if scale is None:
        scale = builder.infer_segmentation_scale()

    return builder.build(
        directory,
        segmentation_path,
        scale=scale,
        node_name_map=builder.node_name_map,
        database=database,
    )
