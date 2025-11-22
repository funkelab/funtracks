from __future__ import annotations

from typing import TYPE_CHECKING

import dask.array as da
import geff
from geff.core_io._base_read import read_to_memory
from geff.validate.segmentation import axes_match_seg_dims, has_valid_seg_id
from geff.validate.tracks import validate_lineages, validate_tracklets

from funtracks.features import Feature
from funtracks.import_export._import_segmentation import (
    import_segmentation,
    lazy_load_segmentation,
)
from funtracks.import_export._utils import _infer_dtype_from_array
from funtracks.import_export._validation import validate_graph_seg_match

if TYPE_CHECKING:
    from pathlib import Path

    from geff._typing import InMemoryGeff

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


def import_from_geff(
    directory: Path,
    name_map: dict[str, str],
    segmentation_path: Path | None = None,
    scale: list[float] | None = None,
    node_features: dict[str, bool] | None = None,
    edge_name_map: dict[str, str] | None = None,
) -> SolutionTracks:
    """Load Tracks from a geff directory. Takes a name_map to map graph attributes
    (spatial dimensions and optional track and lineage ids) to tracks attributes.
    Optionally takes a path to segmentation data, and verifies if the segmentation data
    matches with the graph data. If a scaling tuple is provided, it will be used to scale
    the spatial coordinates on the graph (world coordinates) to pixel coordinates when
    checking if segmentation data matches the graph data. If no scale is provided, the
    geff metadata will be queried for a scale, if it is not present, no scaling will be
    applied. Optional extra features, present as node properties in the geff, can be
    included by providing a dictionary with keys as the feature names and values as
    booleans indicating whether to they should be recomputed (currently only supported for
    the 'area' feature), or loaded as static node attributes.

    Args:
        directory (Path): path to the geff tracks data or its parent folder.
        name_map (dict[str,str]): dictionary mapping standard keys to GEFF property names.
            Required fields:
                - time: temporal coordinate
                - y, x: spatial coordinates (required)
                - z: spatial coordinate (required for 4D data)
            Optional fields:
                - seg_id: segmentation ID (if segmentation is provided)
                - tracklet_id: tracklet identifier (for solutions)
                - lineage_id: lineage identifier (for solutions)
            Any additional properties:
                - Can include any other properties from the GEFF file
                - Both computed features (area, circularity, etc.) and static features
                - Maps standard feature keys to their names in the GEFF file
        segmentation_path (Path | None = None): path to segmentation data.
        scale (list[float]): scaling information (pixel to world coordinates).
        node_features (dict[str, bool] | None=None):
            Optional dict mapping feature names to recompute flags.
            Keys are standard feature names, values indicate whether to recompute (True)
            or use existing values from GEFF (False).
            Features not in name_map will be auto-added assuming they map to themselves.
        edge_name_map (dict[str, str] | None): Maps standard keys to GEFF edge
            property names. If None, all edge properties loaded with original names.
            If provided, only specified properties loaded and renamed.
            Example: {"iou": "overlap"}
    Returns:
        Tracks based on the geff graph and segmentation, if provided.
    """
    # For backward compatibility, add node_features keys to name_map if not present
    # Only add features that should be loaded from GEFF (recompute=False)
    # Assumes feature name in GEFF matches the standard key
    extended_name_map = dict(name_map)
    if node_features is not None:
        for feature_key, recompute in node_features.items():
            if feature_key not in extended_name_map and not recompute:
                extended_name_map[feature_key] = feature_key

    # Load and validate GEFF data (returns InMemoryGeff with standard keys)
    in_memory_geff, position_attr, ndims = import_graph_from_geff(
        directory, extended_name_map, edge_name_map=edge_name_map
    )

    metadata = dict(in_memory_geff["metadata"])
    node_ids = in_memory_geff["node_ids"]
    node_props = in_memory_geff["node_props"]  # Now has standard keys!
    edge_ids = in_memory_geff["edge_ids"]
    edge_props = in_memory_geff["edge_props"]
    segmentation = None
    time_attr = "time"  # Using standard key directly

    # if no scale is provided, load from metadata if available.
    if scale is None:
        scale = list([1.0] * ndims)
        axes = metadata.get("axes")
        if axes:
            lookup = {a.name.lower(): (a.scale or 1) for a in axes}
            scale[-1], scale[-2] = lookup.get("x", 1), lookup.get("y", 1)
            if "z" in lookup:
                scale[-3] = lookup.get("z", 1)

    # Check if a track_id was provided, and if it is valid keep it.
    # If it is not provided or invalid, it will be computed again.
    if TRACK_KEY in node_props:
        # if track id is present, it is a solution graph
        valid_track_ids, errors = validate_tracklets(
            node_ids=node_ids,
            edge_ids=edge_ids,
            tracklet_ids=node_props[TRACK_KEY]["values"],
        )
        if not valid_track_ids:
            # Remove invalid track_id from node properties
            node_props.pop(TRACK_KEY)

    # Check if a lineage_id was provided, and if it is valid keep it.
    # If invalid, remove it from node properties.
    if "lineage_id" in node_props:
        valid_lineages, errors = validate_lineages(
            node_ids=node_ids,
            edge_ids=edge_ids,
            lineage_ids=node_props["lineage_id"]["values"],
        )
        if not valid_lineages:
            # Remove invalid lineage_id from node properties
            node_props.pop("lineage_id")

    # All pre-checks have passed, load the graph now.
    graph = geff.construct(
        metadata=in_memory_geff["metadata"],
        node_ids=in_memory_geff["node_ids"],
        edge_ids=in_memory_geff["edge_ids"],
        node_props=node_props,
        edge_props=edge_props,
    )

    # Try to load the segmentation data, if it was provided.
    if segmentation_path is not None:
        seg_reference = lazy_load_segmentation(segmentation_path)

        # GEFF-specific validation: Check axes metadata matches segmentation dimensions
        axes_valid, errors = axes_match_seg_dims(in_memory_geff, seg_reference)
        if not axes_valid:
            error_msg = "Axes in the geff do not match segmentation:\n"
            error_msg += "\n".join(f"- {e}" for e in errors)
            raise ValueError(error_msg)

        # GEFF-specific validation: Check seg_ids are valid integers
        seg_id_valid, errors = has_valid_seg_id(in_memory_geff, seg_id=SEG_KEY)
        if not seg_id_valid:
            error_msg = "Invalid seg_id values in geff:\n"
            error_msg += "\n".join(f"- {e}" for e in errors)
            raise ValueError(error_msg)

        # Generic validation: check graph matches segmentation and if relabeling needed
        relabel = validate_graph_seg_match(graph, seg_reference, scale, position_attr)
        segmentation = import_segmentation(segmentation_path, graph, relabel=relabel)
    # Put segmentation data in memory now.
    if segmentation is not None and isinstance(segmentation, da.Array):
        segmentation = segmentation.compute()

    # Import at runtime to avoid circular dependency
    from funtracks.data_model.solution_tracks import SolutionTracks

    # Create the tracks.
    tracks = SolutionTracks(
        graph=graph,
        segmentation=segmentation,
        pos_attr=position_attr,
        time_attr=time_attr,
        ndim=ndims,
        scale=scale,
    )

    # Enable features from node_features (both computed and loaded)
    if node_features is not None:
        # Validate requested features before enabling
        invalid_features = []
        for key, recompute in node_features.items():
            if recompute:
                # Features to compute must exist in annotators
                if key not in tracks.annotators.all_features:
                    invalid_features.append(key)
            else:
                # Features to load must exist in node_props
                if key not in node_props:
                    invalid_features.append(key)

        if invalid_features:
            available_computed = list(tracks.annotators.all_features.keys())
            available_geff = list(node_props.keys())
            raise KeyError(
                f"Features not available: {invalid_features}. "
                f"Available computed features: {available_computed}. "
                f"Available GEFF properties: {available_geff}"
            )

        # Separate into features that exist in annotators vs static features
        annotator_features = {
            key: recompute
            for key, recompute in node_features.items()
            if key in tracks.annotators.all_features
        }

        # Enable annotator features with appropriate recompute flag
        for key, recompute in annotator_features.items():
            tracks.enable_features([key], recompute=recompute)

    # Register static features (features not in annotator registry)
    static_keys = [
        key for key in extended_name_map if key not in tracks.annotators.all_features
    ]
    static_features: dict[str, Feature] = {}
    for key in static_keys:
        static_features[key] = Feature(
            display_name=key,
            feature_type="node",
            value_type=_infer_dtype_from_array(node_props[key]["values"]),
            num_values=1,
            required=False,
            default_value=None,
        )
    tracks.features.update(static_features)

    # Handle edge features similarly if edge_name_map was provided
    if edge_name_map is not None:
        edge_static_features: dict[str, Feature] = {}
        for key in edge_name_map:
            if key in edge_props:
                edge_static_features[key] = Feature(
                    display_name=key,
                    feature_type="edge",
                    value_type=_infer_dtype_from_array(edge_props[key]["values"]),
                    num_values=1,
                    required=False,
                    default_value=None,
                )
        tracks.features.update(edge_static_features)

    return tracks
