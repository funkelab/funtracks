from __future__ import annotations

from typing import TYPE_CHECKING, cast

import dask.array as da
import geff
from geff.core_io._base_read import read_to_memory
from geff.validate.tracks import validate_lineages, validate_tracklets

from funtracks.features import Feature, ValueType
from funtracks.import_export._import_segmentation import (
    import_segmentation,
    lazy_load_segmentation,
)
from funtracks.import_export._register_computed_features import (
    register_computed_features,
)
from funtracks.import_export._utils import (
    _get_default_key_to_display_name_mapping,
    _infer_dtype_from_array,
)
from funtracks.import_export._validation import validate_graph_seg_match

if TYPE_CHECKING:
    from pathlib import Path

    from geff._typing import InMemoryGeff

    from funtracks.data_model.solution_tracks import SolutionTracks
    from funtracks.import_export._types import (
        ImportedComputedFeature,
        ImportedNodeFeature,
    )


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
            Example: {"time": "t", "circularity": "circ"}
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
        directory, node_props=list(node_prop_filter), edge_props=edge_prop_filter
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
    node_features: list[ImportedNodeFeature] | dict[str, bool] | None = None,
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
        node_features (list[ImportedNodeFeature] | dict[str, bool] | None=None):
            optional features to include in the Tracks object. Can be either:
            - list[ImportedNodeFeature]: Recommended format with full feature metadata
            - dict[str, bool]: DEPRECATED - maps feature name to recompute flag
        edge_name_map (dict[str, str] | None): Maps standard keys to GEFF edge
            property names. If None, all edge properties loaded with original names.
            If provided, only specified properties loaded and renamed.
            Example: {"iou": "overlap"}
    Returns:
        Tracks based on the geff graph and segmentation, if provided.
    """

    # Issue deprecation warning for dict[str, bool] format (conversion happens later)
    if node_features is not None and isinstance(node_features, dict):
        import warnings

        warnings.warn(
            "Passing node_features as dict[str, bool] is deprecated. "
            "Use list[ImportedNodeFeature] instead. "
            "See ImportedNodeFeature documentation for the new format.",
            DeprecationWarning,
            stacklevel=2,
        )

    # Load and validate GEFF data (returns InMemoryGeff with standard keys)
    in_memory_geff, position_attr, ndims = import_graph_from_geff(
        directory, name_map, edge_name_map=edge_name_map
    )

    metadata = dict(in_memory_geff["metadata"])
    node_ids = in_memory_geff["node_ids"]
    node_props = in_memory_geff["node_props"]  # Now has standard keys!
    edge_ids = in_memory_geff["edge_ids"]
    edge_props = in_memory_geff["edge_props"]
    segmentation = None
    time_attr = "time"  # Using standard key directly

    # Rebuild node_prop_filter for later use (now using standard keys)
    node_prop_filter: set[str] = {
        standard_key for standard_key in name_map if standard_key is not None
    }
    if node_features is not None:
        if isinstance(node_features, dict):
            node_prop_filter.update(
                key
                for key, should_recompute in node_features.items()
                if not should_recompute
            )
        else:
            node_prop_filter.update(
                str(feature["prop_name"])
                for feature in node_features
                if feature["feature"] is None or not bool(feature["recompute"])
            )

    # Convert dict[str, bool] format to list[ImportedNodeFeature] now that we have
    # geff data
    if node_features is not None and isinstance(node_features, dict):
        from funtracks.annotators._regionprops_annotator import DEFAULT_POS_KEY

        # Get the mapping and add position key based on ndim
        key_to_display = _get_default_key_to_display_name_mapping(ndim=ndims)
        position_display_name = (
            tuple(position_attr) if len(position_attr) > 1 else (position_attr[0],)
        )
        key_to_display[DEFAULT_POS_KEY] = position_display_name

        converted_features: list[ImportedNodeFeature] = []
        for key, should_recompute in node_features.items():
            # Check if it's a known computed feature
            if key in key_to_display:
                display_name = key_to_display[key]
                dtype = cast(ValueType, "float")  # Placeholder for computed features
            # Check if it exists in geff data (static feature)
            elif key in node_props:
                display_name = None
                dtype = _infer_dtype_from_array(node_props[key]["values"])
            else:
                raise ValueError(
                    f"Unknown feature '{key}' - not found in geff data or known "
                    "computed features. Available features in geff: "
                    f"{list(node_props.keys())}"
                )

            converted_features.append(
                {
                    "prop_name": key,
                    "feature": display_name,
                    "recompute": should_recompute,
                    "dtype": dtype,
                }
            )
        node_features = converted_features

    # if no scale is provided, load from metadata if available.
    if scale is None:
        scale = list([1.0] * ndims)
        axes = metadata.get("axes")
        if axes:
            lookup = {a.name.lower(): (a.scale or 1) for a in axes}
            scale[-1], scale[-2] = lookup.get("x", 1), lookup.get("y", 1)
            if "z" in lookup:
                scale[-3] = lookup.get("z", 1)

    # Check if a track_id was provided, and if it is valid keep it in the
    # node_prop_filter. If it is not provided or invalid, it will be computed again.
    if TRACK_KEY in node_props:
        # if track id is present, it is a solution graph
        valid_track_ids, errors = validate_tracklets(
            node_ids=node_ids,
            edge_ids=edge_ids,
            tracklet_ids=node_props[TRACK_KEY]["values"],
        )
        if not valid_track_ids:
            # Remove invalid track_id from properties to load
            node_prop_filter.discard(TRACK_KEY)

    # Check if a lineage_id was provided, and if it is valid keep it in the
    # node_prop_filter. If invalid, remove it from properties to load.
    if "lineage_id" in node_props:
        valid_lineages, errors = validate_lineages(
            node_ids=node_ids,
            edge_ids=edge_ids,
            lineage_ids=node_props["lineage_id"]["values"],
        )
        if not valid_lineages:
            # Remove invalid lineage_id from properties to load
            node_prop_filter.discard("lineage_id")

    # All pre-checks have passed, load the graph now.
    filtered_node_props = {k: v for k, v in node_props.items() if k in node_prop_filter}

    graph = geff.construct(
        metadata=in_memory_geff["metadata"],
        node_ids=in_memory_geff["node_ids"],
        edge_ids=in_memory_geff["edge_ids"],
        node_props=filtered_node_props,
        edge_props=edge_props,
    )

    # Note: No need to relabel track_id anymore since InMemoryGeff
    # already has standard keys

    # Try to load the segmentation data, if it was provided.
    if segmentation_path is not None:
        seg_reference = lazy_load_segmentation(segmentation_path)
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

    static_features: dict[str, Feature] = {}
    computed_features: list[ImportedComputedFeature] = []

    if node_features is not None:
        for import_feat in node_features:
            prop_name = import_feat["prop_name"]
            feat_name = import_feat["feature"]
            if feat_name is None:
                static_features[prop_name] = Feature(
                    display_name=prop_name,
                    feature_type="node",
                    value_type=import_feat["dtype"],
                    num_values=1,
                    required=False,
                    default_value=None,
                )
            else:
                computed_features.append(
                    {
                        "prop_name": prop_name,
                        "feature": feat_name,
                        "recompute": import_feat["recompute"],
                    }
                )

    if len(computed_features) > 0:
        register_computed_features(tracks, computed_features)

    # Add static features to tracks.features
    tracks.features.update(static_features)
    return tracks
