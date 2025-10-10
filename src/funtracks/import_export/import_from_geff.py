from __future__ import annotations

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

from funtracks.data_model.graph_attributes import NodeAttr
from funtracks.import_export.magic_imread import magic_imread

if TYPE_CHECKING:
    from pathlib import Path

    from geff._typing import InMemoryGeff

from funtracks.data_model.solution_tracks import SolutionTracks


def relabel_seg_id_to_node_id(
    times: ArrayLike, ids: ArrayLike, seg_ids: ArrayLike, segmentation: da.Array
) -> np.ndarray:
    """Relabel the segmentation from seg_id to unique node id.
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
    """Validate if the given geff matches the provided segmentation data. Raises a value
    error if no valid seg ids are provided, if the metadata axes do not match the
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
    if name_map.get(NodeAttr.SEG_ID.value) is not None:
        seg_ids_valid, errors = has_valid_seg_id(
            in_memory_geff, name_map[NodeAttr.SEG_ID.value]
        )
        if not seg_ids_valid:
            error_msg = "Error in validating the segmentation ids:\n" + "\n".join(
                f"- {e}" for e in errors
            )
            raise ValueError(error_msg)
        seg_id = int(node_props[name_map[NodeAttr.SEG_ID.value]]["values"][-1])
    else:
        # assign the node id as seg_id instead and check in the next step if this is valid
        seg_id = int(node_ids[-1])

    # Get the coordinates for the last node.
    t = node_props[name_map[NodeAttr.TIME.value]]["values"][-1]
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
    extra_features: dict[str, bool] | None = None,
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
        extra_features (dict[str: bool] | None=None): optional features to include in the
            Tracks object. The keys are the feature names, and the boolean value indicates
            whether to recompute the feature (area) or load it as a static node attribute.
        edge_prop_filter (list[str]): List of edge properties to include. If None all
        properties will be included.
    Returns:
        Tracks based on the geff graph and segmentation, if provided.
    """

    # Read the GEFF file into memory
    node_prop_filter = [
        prop for key, prop in name_map.items() if name_map[key] is not None
    ]
    if extra_features is not None:
        node_prop_filter.extend(list(extra_features.keys()))

    in_memory_geff = read_to_memory(
        directory, node_props=node_prop_filter, edge_props=edge_prop_filter
    )
    metadata = dict(in_memory_geff["metadata"])
    node_ids = in_memory_geff["node_ids"]
    node_props = in_memory_geff["node_props"]
    edge_ids = in_memory_geff["edge_ids"]
    edge_props = in_memory_geff["edge_props"]
    selected_attrs = []
    segmentation = None

    # Check that the spatiotemporal key mapping does not contain None or duplicate values.
    # It is allowed to not include z, but it is not allowed to include z with a None or
    # duplicate value.
    spatio_temporal_keys = [NodeAttr.TIME.value, "z", "y", "x"]
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
    time_attr = name_map[NodeAttr.TIME.value]
    selected_attrs.append(name_map[NodeAttr.TIME.value])
    position_attr = [name_map[k] for k in ("z", "y", "x") if k in name_map]
    selected_attrs.extend(position_attr)
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
    if name_map.get(NodeAttr.TRACK_ID.value) is not None:
        # if track id is present, it is a solution graph
        valid_track_ids, errors = validate_tracklets(
            node_ids=node_ids,
            edge_ids=edge_ids,
            tracklet_ids=node_props[name_map[NodeAttr.TRACK_ID.value]]["values"],
        )
        if valid_track_ids:
            selected_attrs.append(NodeAttr.TRACK_ID.value)
    recompute_track_ids = NodeAttr.TRACK_ID.value not in selected_attrs

    # Check if a lineage_id was provided, and if it is valid add it to list of selected
    # attributes. If it is not provided, it will be a static feature (for now).
    if name_map.get("lineage_id") is not None:
        valid_lineages, errors = validate_lineages(
            node_ids=node_ids,
            edge_ids=edge_ids,
            lineage_ids=node_props[name_map["lineage_id"]]["values"],
        )
        if valid_lineages:
            selected_attrs.append(name_map["lineage_id"])

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
            times = node_props[name_map[NodeAttr.TIME.value]]["values"][:]
            ids = node_ids[:]
            seg_ids = node_props[name_map[NodeAttr.SEG_ID.value]]["values"][:]

            if not len(times) == len(ids) == len(seg_ids):
                raise ValueError(
                    "Encountered missing values in the seg_id to node id conversion."
                )
            segmentation = relabel_seg_id_to_node_id(times, ids, seg_ids, segmentation)

    # Add optional extra features.
    if extra_features is None:
        extra_features = {}
    selected_attrs.extend(extra_features.keys())

    # All pre-checks have passed, load the graph now.
    filtered_node_props = {k: v for k, v in node_props.items() if k in selected_attrs}
    graph = geff.construct(
        metadata=in_memory_geff["metadata"],
        node_ids=in_memory_geff["node_ids"],
        edge_ids=in_memory_geff["edge_ids"],
        node_props=filtered_node_props,
        edge_props=edge_props,
    )

    # Relabel track_id attr to NodeAttr.TRACK_ID.value (unless we should recompute)
    if name_map.get(NodeAttr.TRACK_ID.value) is not None and not recompute_track_ids:
        for _, data in graph.nodes(data=True):
            try:
                data[NodeAttr.TRACK_ID.value] = data.pop(
                    name_map[NodeAttr.TRACK_ID.value]
                )
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
        recompute_track_ids=recompute_track_ids,
    )
    # compute the 'area' attribute if needed
    if tracks.segmentation is not None and extra_features.get("area"):
        nodes = tracks.graph.nodes
        times = tracks.get_times(nodes)
        computed_attrs = [
            tracks._compute_node_attrs(node, time)
            for node, time in zip(nodes, times, strict=True)
        ]
        areas = [attr[NodeAttr.AREA.value] for attr in computed_attrs]
        tracks._set_nodes_attr(nodes, NodeAttr.AREA.value, areas)

    return tracks
