from __future__ import annotations

from typing import (
    TYPE_CHECKING,
)

import geff
import numpy as np
import zarr
from geff.affine import Affine
from geff.validators.segmentation_validators import (
    axes_match_seg_dims,
    has_seg_ids_at_coords,
    has_valid_seg_id,
)
from geff.validators.validators import validate_lineages, validate_tracklets
from numpy.typing import ArrayLike

from funtracks.data_model.graph_attributes import NodeAttr
from funtracks.import_export.magic_imread import magic_imread

if TYPE_CHECKING:
    from pathlib import Path

import dask.array as da

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
    directory: Path,
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
        directory (Path): path to the geff tracks data or its parent folder.
        name_map (dict[str,str]): dictionary mapping required fields to node properties.
        segmentation (ArrayLike): segmentation data.
        scale (list[float]): scaling information (pixel to world coordinates).
        position_attr (list[str]): position keys in the geff tracks data

    Returns:
        bool: True if relabeling from seg_id to node_id is required.
    """

    group = zarr.open_group(directory, mode="r")
    # check if the axes information in the metadata matches the segmentation
    # dimensions
    axes_match, errors = axes_match_seg_dims(directory, segmentation)
    if not axes_match:
        error_msg = "Axes in the geff do not match segmentation:\n" + "\n".join(
            f"- {e}" for e in errors
        )
        raise ValueError(error_msg)

    # Check if valid seg_ids are provided
    if name_map.get(NodeAttr.SEG_ID.value) is not None:
        seg_ids_valid, errors = has_valid_seg_id(
            directory, name_map[NodeAttr.SEG_ID.value]
        )
        if not seg_ids_valid:
            error_msg = "Error in validating the segmentation ids:\n" + "\n".join(
                f"- {e}" for e in errors
            )
            raise ValueError(error_msg)
        seg_id = int(
            group["nodes"]["props"][name_map[NodeAttr.SEG_ID.value]]["values"][-1]
        )
    else:
        # assign the node id as seg_id instead and check in the next step if this is
        #  valid.
        seg_id = int(group["nodes"]["ids"][-1])

    # Get the coordinates for the last node.
    t = group["nodes"]["props"][name_map[NodeAttr.TIME.value]]["values"][-1]
    z = (
        group["nodes"]["props"][name_map["z"]]["values"][-1]
        if len(position_attr) == 3
        else None
    )
    y = group["nodes"]["props"][name_map["y"]]["values"][-1]
    x = group["nodes"]["props"][name_map["x"]]["values"][-1]

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

    return group["nodes"]["ids"][-1] != seg_id


def import_from_geff(
    directory: Path,
    name_map: dict[str, str],
    segmentation_path: Path | None = None,
    scale: list[float] | None = None,
    extra_features: dict[str, bool] | None = None,
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

    Returns:
        Tracks based on the geff graph and segmentation, if provided.
    """

    group = zarr.open_group(directory, mode="r")
    metadata = dict(group.attrs)
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

    # if no scale is provided, load from metadata, if available.
    if scale is None:
        affine = metadata.get("affine", {}).get("matrix", None)
        if affine is not None:
            affine = Affine(matrix=affine)
            linear = affine.linear_matrix
            scale = list(np.diag(linear))
        else:
            scale = list([1.0] * ndims)

    # Check if a track_id was provided, and if it is valid add it to list of selected
    # attributes. If it is not provided, it will be computed again.
    if name_map.get(NodeAttr.TRACK_ID.value) is not None:
        # if track id is present, it is a solution graph
        valid_track_ids, errors = validate_tracklets(
            node_ids=group["nodes"]["ids"][:],
            edge_ids=group["edges"]["ids"][:],
            tracklet_ids=group["nodes"]["props"][name_map[NodeAttr.TRACK_ID.value]][
                "values"
            ][:],
        )
        if valid_track_ids:
            selected_attrs.append(NodeAttr.TRACK_ID.value)
    recompute_track_ids = NodeAttr.TRACK_ID.value not in selected_attrs

    # Check if a lineage_id was provided, and if it is valid add it to list of selected
    # attributes. If it is not provided, it will be a static feature (for now).
    if name_map.get("lineage_id") is not None:
        valid_lineages, errors = validate_lineages(
            node_ids=group["nodes"]["ids"],
            edge_ids=group["edges"]["ids"],
            lineage_ids=group["nodes"]["props"][name_map["lineage_id"]]["values"],
        )
        if valid_lineages:
            selected_attrs.append(name_map["lineage_id"])

    # Try to load the segmentation data, if it was provided.
    if segmentation_path is not None:
        segmentation = magic_imread(
            segmentation_path, use_dask=True
        )  # change to in memory later

        relabel = validate_graph_seg_match(
            directory, segmentation, name_map, scale, position_attr
        )

        # If the provided segmentation has seg ids that are not identical to node ids,
        # relabel it now.
        if relabel:
            times = group["nodes"]["props"][name_map[NodeAttr.TIME.value]]["values"][:]
            ids = group["nodes"]["ids"][:]
            seg_ids = group["nodes"]["props"][name_map[NodeAttr.SEG_ID.value]]["values"][
                :
            ]

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
    graph, _ = geff.read_nx(directory, node_props=selected_attrs)

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
        areas = [
            tracks._compute_node_attrs(node, time)[NodeAttr.AREA.value]
            for node, time in zip(nodes, times, strict=True)
        ]
        tracks._set_nodes_attr(nodes, NodeAttr.AREA.value, areas)

    return tracks
