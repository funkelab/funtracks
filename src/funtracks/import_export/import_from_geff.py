from __future__ import annotations

import inspect
from typing import (
    TYPE_CHECKING,
)

import dask.array as da
import geff
import numpy as np
from geff.core_io._base_read import read_to_memory
from geff.validate.segmentation import (
    axes_match_seg_dims,
    has_seg_ids_at_coords,
    has_valid_seg_id,
)
from geff.validate.tracks import validate_lineages, validate_tracklets
from numpy.typing import ArrayLike

from funtracks.annotators._regionprops_annotator import (
    DEFAULT_AREA_KEY,
    DEFAULT_CIRCULARITY_KEY,
    DEFAULT_ELLIPSE_AXIS_KEY,
    DEFAULT_PERIMETER_KEY,
)
from funtracks.features import _regionprops_features
from funtracks.features._feature import Feature
from funtracks.import_export.magic_imread import magic_imread

if TYPE_CHECKING:
    from pathlib import Path

    from geff._typing import InMemoryGeff

from funtracks.data_model.solution_tracks import SolutionTracks

# defining constants here because they are only used in the context of import
TRACK_KEY = "track_id"
SEG_KEY = "seg_id"

regionprop_class_names = [
    name
    for name, func in inspect.getmembers(_regionprops_features, inspect.isfunction)
    if func.__module__ == "funtracks.features._regionprops_features"
]

default_name_map = {
    "Area": DEFAULT_AREA_KEY,
    "Circularity": DEFAULT_CIRCULARITY_KEY,
    "Perimeter": DEFAULT_PERIMETER_KEY,
    "EllipsoidAxes": DEFAULT_ELLIPSE_AXIS_KEY,
}


def get_regionprops_features(
    node_features,
    node_props,
    ndims,
) -> (dict[str, int | float], list[dict]):
    """
    Search for features that should be registered as Regionprop feature, but not
    recomputed. They have to be registered under the correct name for the annotators to
    recognize them.

    Returns:
        regionprop_node_props: dict[str, float | int]: dictionary with default Regionprop
        Feature names as keys and the corresponding values from the geff
        regionprop_feature_list: list[dict]
            List of {prop_name: feature_dict} entries for downstream processing.
    """

    # for each to be checked feature, check if it is 1) a RegionProp Feature and 2) it
    # does not have to be recomputed
    regionprop_node_props = {}
    regionprop_feature_list = []

    for feature in node_features:
        feat_name = feature["feature"]
        prop_name = feature["prop_name"]

        if feat_name in regionprop_class_names and not feature["recompute"]:
            regionprop_func = getattr(_regionprops_features, feat_name, None)
            if regionprop_func is None:
                raise ValueError(f"Regionprop feature '{feat_name}' not found.")

            # Determine correct prop_name (the new key)
            new_prop_name = default_name_map.get(feat_name, prop_name)

            # Call the regionprop function (check if it accepts ndim)
            sig = inspect.signature(regionprop_func)
            if "ndim" in sig.parameters:
                regionprop_feature = regionprop_func(ndim=ndims)
            else:
                regionprop_feature = regionprop_func()

            # Append to the list for downstream use
            regionprop_feature_list.append({new_prop_name: regionprop_feature})

            # rename regionprop_node_props directly here
            regionprop_node_props[new_prop_name] = node_props[feature["prop_name"]]

    return regionprop_node_props, regionprop_feature_list


def relabel_seg_id_to_node_id(
    times: ArrayLike, ids: ArrayLike, seg_ids: ArrayLike, segmentation: da.Array
) -> np.ndarray:
    """Create a new segmentation where masks are relabeled to match node ids.

    TODO: How does this relate to motile_toolbox.ensure_unique_labels? Just lazy/dask?

    Args:
        times (ArrayLike): array of time points, one per node
        ids (ArrayLike): array of node ids
        seg_ids (ArrayLike): array of segmentation ids, one per node
        segmentation (da.array): A dask array where segmentation label values match the
          "seg_id" values.

    Returns:
        np.ndarray: A numpy array of dtype uint64, similar to the input segmentation
            where each segmentation now has a unique label across time that corresponds
            to the ID of each node.
    """

    new_segmentation = np.zeros(segmentation.shape, dtype=np.uint64)
    for i, node in enumerate(ids):
        mask = segmentation[times[i]].compute() == seg_ids[i]
        new_segmentation[times[i], mask] = node

    return new_segmentation


def validate_graph_seg_match(
    in_memory_geff: InMemoryGeff,
    segmentation: ArrayLike,
    name_map: dict[str, str],
    scale: list[float],
    position_attr: list[str],
) -> bool:
    """Validate if the given geff matches the provided segmentation data.

    Raises an error if no valid seg ids are provided, if the metadata axes do not match
    segmentation shape, or if the seg_id value of the last node does not match the pixel
    value at the (scaled) node coordinates. Returns a boolean indicating whether
    relabeling of the segmentation to match it to node id values is required.

    Args:
        in_memory_geff (InMemoryGeff): geff data read into memory
        name_map (dict[str,str]): dictionary mapping required fields to node properties.
        segmentation (ArrayLike): segmentation data.
        scale (list[float]): scaling information (pixel to world coordinates).
        position_attr (list[str]): position keys in the geff tracks data

    Returns:
        bool: True if relabeling from seg_id to node_id is required.
    """

    # check if the axes information in the metadata matches the segmentation
    # dimensions
    axes_match, errors = axes_match_seg_dims(in_memory_geff, segmentation)
    if not axes_match:
        error_msg = "Axes in the geff do not match segmentation:\n" + "\n".join(
            f"- {e}" for e in errors
        )
        raise ValueError(error_msg)

    node_ids = in_memory_geff["node_ids"]
    node_props = in_memory_geff["node_props"]

    # Check if valid seg_ids are provided
    if name_map.get(SEG_KEY) is not None:
        seg_ids_valid, errors = has_valid_seg_id(in_memory_geff, name_map[SEG_KEY])
        if not seg_ids_valid:
            error_msg = "Error in validating the segmentation ids:\n" + "\n".join(
                f"- {e}" for e in errors
            )
            raise ValueError(error_msg)
        seg_id = int(node_props[name_map[SEG_KEY]]["values"][-1])
    else:
        # assign the node id as seg_id instead and check in the next step if this is valid
        seg_id = int(node_ids[-1])

    # Get the coordinates for the last node.
    t = node_props[name_map["time"]]["values"][-1]
    z = node_props[name_map["z"]]["values"][-1] if len(position_attr) == 3 else None
    y = node_props[name_map["y"]]["values"][-1]
    x = node_props[name_map["x"]]["values"][-1]

    coord = []
    coord.append(t)
    if z is not None:
        coord.append(z)
    coord.append(y)
    coord.append(x)

    # Check if the segmentation pixel value at the coordinates of the last node
    # matches the seg id. Since the scale factor was used to convert from pixels to
    # world coordinates, we need to invert this scale factor to get the pixel
    # coordinates.
    seg_id_at_coord, errors = has_seg_ids_at_coords(
        segmentation, [coord], [seg_id], tuple(1 / s for s in scale)
    )
    if not seg_id_at_coord:
        error_msg = "Error testing seg id:\n" + "\n".join(f"- {e}" for e in errors)
        raise ValueError(error_msg)

    return node_ids[-1] != seg_id


def import_from_geff(
    directory: Path,
    name_map: dict[str, str],
    segmentation_path: Path | None = None,
    scale: list[float] | None = None,
    node_features: list[dict[str, str | bool]] | None = None,
    edge_prop_filter: list[str] | None = None,
):
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
        name_map (dict[str,str]): dictionary mapping required fields to node properties.
            Should include:
                time,
                (z),
                y,
                x,
                (seg_id), if a segmentation is provided
                (tracklet_id), optional, if it is a solution
                (lineage_id), optional, if it is a solution
        segmentation_path (Path | None = None): path to segmentation data.
        scale (list[float]): scaling information (pixel to world coordinates).
        node_features list[(dict[str, str | bool]] | None=None): optional features to
            include in the Tracks object. Each to be included feature should be a
            dictionary with the following keys: "prop_name", "feature", "recompute",
            "dtype". The prop_name is the name of the property in the geff, feature is the
            name of the feature, this can either be the class name of a regionprops
            Feature, or it can be 'Group' (will be imported as group) or 'Custom'
            (static feature). Recompute is a boolean that indicates whether to recompute
            the regionprops feature or load it as is.
        edge_prop_filter (list[str]): List of edge properties to include. If None all
        properties will be included.
    Returns:
        Tracks based on the geff graph and segmentation, if provided.
    """

    # Read the GEFF file into memory
    node_prop_filter = [
        prop for key, prop in name_map.items() if name_map[key] is not None
    ]
    if node_features is not None:
        # Only load features from geff that should NOT be computed (value=False)
        # Features with value=True will be computed by annotators after loading
        node_prop_filter.extend(
            [
                feature["prop_name"]
                for feature in node_features
                if not feature["recompute"]
            ]
        )

    in_memory_geff = read_to_memory(
        directory, node_props=node_prop_filter, edge_props=edge_prop_filter
    )
    metadata = dict(in_memory_geff["metadata"])
    node_ids = in_memory_geff["node_ids"]
    node_props = in_memory_geff["node_props"]
    edge_ids = in_memory_geff["edge_ids"]
    edge_props = in_memory_geff["edge_props"]
    node_attrs_to_load_from_geff = []
    segmentation = None

    # Check that the spatiotemporal key mapping does not contain None or duplicate values.
    # It is allowed to not include z, but it is not allowed to include z with a None or
    # duplicate value.
    spatio_temporal_keys = ["time", "z", "y", "x"]
    spatio_temporal_map = {
        key: name_map[key] for key in spatio_temporal_keys if key in name_map
    }
    if any(v is None for v in spatio_temporal_map.values()):
        raise ValueError(
            "The name_map cannot contain None values. Please provide a valid mapping "
            "for all required fields."
        )
    if len(set(spatio_temporal_map.values())) != len(spatio_temporal_map.values()):
        raise ValueError(
            "The name_map cannot contain duplicate values. Please provide a unique "
            "mapping for each required field."
        )

    # Extract the time and position attributes from the name_map, containing and optional
    # z coordinate.
    time_attr = name_map["time"]
    node_attrs_to_load_from_geff.append(name_map["time"])
    position_attr = [name_map[k] for k in ("z", "y", "x") if k in name_map]
    node_attrs_to_load_from_geff.extend(position_attr)
    ndims = len(position_attr) + 1

    # if no scale is provided, load from metadata if available.
    if scale is None:
        scale = list([1.0] * ndims)
        axes = metadata.get("axes", [])
        lookup = {a.name.lower(): (a.scale or 1) for a in axes}
        scale[-1], scale[-2] = lookup.get("x", 1), lookup.get("y", 1)
        if "z" in lookup:
            scale[-3] = lookup.get("z", 1)

    # Check if a track_id was provided, and if it is valid add it to list of selected
    # attributes. If it is not provided, it will be computed again.
    if name_map.get(TRACK_KEY) is not None:
        # if track id is present, it is a solution graph
        valid_track_ids, errors = validate_tracklets(
            node_ids=node_ids,
            edge_ids=edge_ids,
            tracklet_ids=node_props[name_map[TRACK_KEY]]["values"],
        )
        if valid_track_ids:
            node_attrs_to_load_from_geff.append(TRACK_KEY)
    recompute_track_ids = TRACK_KEY not in node_attrs_to_load_from_geff

    # Check if a lineage_id was provided, and if it is valid add it to list of selected
    # attributes. If it is not provided, it will be a static feature (for now).
    if name_map.get("lineage_id") is not None:
        valid_lineages, errors = validate_lineages(
            node_ids=node_ids,
            edge_ids=edge_ids,
            lineage_ids=node_props[name_map["lineage_id"]]["values"],
        )
        if valid_lineages:
            node_attrs_to_load_from_geff.append(name_map["lineage_id"])

    # Try to load the segmentation data, if it was provided.
    if segmentation_path is not None:
        segmentation = magic_imread(
            segmentation_path, use_dask=True
        )  # change to in memory later

        relabel = validate_graph_seg_match(
            in_memory_geff, segmentation, name_map, scale, position_attr
        )

        # If the provided segmentation has seg ids that are not identical to node ids,
        # relabel it now.
        if relabel:
            times = node_props[name_map["time"]]["values"][:]
            ids = node_ids[:]
            seg_ids = node_props[name_map[SEG_KEY]]["values"][:]

            if not len(times) == len(ids) == len(seg_ids):
                raise ValueError(
                    "Encountered missing values in the seg_id to node id conversion."
                )
            segmentation = relabel_seg_id_to_node_id(times, ids, seg_ids, segmentation)

    # All pre-checks have passed, load the graph now.
    filtered_node_props = {
        k: v for k, v in node_props.items() if k in node_attrs_to_load_from_geff
    }

    # Add optional extra features that were loaded from geff (not computed).
    if node_features is not None:
        regionprop_node_props, regionprop_feature_list = get_regionprops_features(
            node_features=node_features,
            node_props=node_props,
            ndims=ndims,
        )
        filtered_node_props.update(regionprop_node_props)

    graph = geff.construct(
        metadata=in_memory_geff["metadata"],
        node_ids=in_memory_geff["node_ids"],
        edge_ids=in_memory_geff["edge_ids"],
        node_props=filtered_node_props,
        edge_props=edge_props,
    )

    # Relabel track_id attr to TRACK_KEY (unless we should recompute)
    if name_map.get(TRACK_KEY) is not None and not recompute_track_ids:
        for _, data in graph.nodes(data=True):
            try:
                data[TRACK_KEY] = data.pop(name_map[TRACK_KEY])
            except KeyError:
                recompute_track_ids = True
                break

    # Put segmentation data in memory now.
    if segmentation is not None and isinstance(segmentation, da.Array):
        segmentation = segmentation.compute()
    # Create the tracks.
    tracks = SolutionTracks(
        graph=graph,
        segmentation=segmentation,
        pos_attr=position_attr,
        time_attr=time_attr,
        ndim=ndims,
        scale=scale,
    )
    # TODO: properly import/activate static features that were loaded from geff using
    # geff metadata to create the Feature
    # Compute any extra features that were requested but not already loaded from geff

    # Register the regionprops features with the RegionpropsAnnotator
    for feature in regionprop_feature_list:
        key = list(feature.keys())[0]
        value = list(feature.values())[0]
        tracks.features[key] = value
        tracks.annotators.activate_features([key])

    # Register other features (Groups or static Custom features)
    for feature in node_features:
        if feature["feature"] == "Group":
            new_group_feature: Feature = {
                "feature_type": "node",
                "value_type": "bool",
                "num_values": 1,
                "display_name": feature["prop_name"],
                "required": False,
                "default_value": False,
                "is_group": True,
            }
            tracks.features[feature["prop_name"]] = new_group_feature
        elif feature["feature"] == "Custom":
            new_custom_feature: Feature = {
                "feature_type": "node",
                "value_type": feature["dtype"],
                "num_values": 1,
                "display_name": feature["prop_name"],
                "required": False,
                "default_value": None,
                "is_group": False,
            }
            tracks.features[feature["prop_name"]] = new_custom_feature

    # Recompute requested regionprops features
    features_to_compute = [feature for feature in node_features if feature["recompute"]]
    features_to_compute_names = []
    for feature in features_to_compute:
        prop_name = default_name_map[feature["feature"]]
        features_to_compute_names.append(prop_name)

    if features_to_compute:
        tracks.enable_features(features_to_compute_names)

    return tracks
