from __future__ import annotations

import itertools
from typing import (
    TYPE_CHECKING,
    Literal,
)

import geff_spec
import numpy as np
import tracksdata as td
from geff_spec import GeffMetadata

from funtracks.utils import remove_tilde, setup_zarr_array, setup_zarr_group

from .._utils import filter_graph_with_ancestors

if TYPE_CHECKING:
    from pathlib import Path

    from funtracks.data_model.tracks import Tracks


def export_to_geff(
    tracks: Tracks,
    directory: Path,
    overwrite: bool = False,
    node_ids: set[int] | None = None,
    zarr_format: Literal[2, 3] = 2,
):
    """Export the Tracks graph to geff.

    Args:
        tracks (Tracks): Tracks object containing a graph to save.
        directory (Path): Destination directory for saving the Zarr.
        overwrite (bool): If True, allows writing into a non-empty directory.
        node_ids (set[int], optional): A set of nodes that should be saved. If
            provided, a valid graph will be constructed that also includes the ancestors
            of the given nodes. All other nodes will NOT be saved.
        zarr_format (Literal[2, 3]): Zarr format version to use. Defaults to 2
            for maximum compatibility.
    """
    directory = remove_tilde(directory)
    directory = directory.resolve(strict=False)

    if node_ids is not None:
        nodes_to_keep = filter_graph_with_ancestors(
            tracks.graph, node_ids
        )  # include the ancestors to make sure the graph is valid and has no missing
        # parent nodes.

    # Create directory as a zarr group
    mode: Literal["w", "w-"] = "w" if overwrite else "w-"
    setup_zarr_group(directory, zarr_format=zarr_format, mode=mode)

    # update the graph to split the position into separate attrs, if they are currently
    # together in a list
    graph, axis_names = split_position_attr(tracks)
    if axis_names is None:
        axis_names = []
    axis_names.insert(0, tracks.features.time_key)
    if axis_names is not None:
        axis_types = (
            ["time", "space", "space"]
            if tracks.ndim == 3
            else ["time", "space", "space", "space"]
        )
    else:
        axis_types = None
    if tracks.scale is None:
        tracks.scale = (1.0,) * tracks.ndim

    # Create axes metadata
    axes = []
    for name, axis_type, scale in zip(axis_names, axis_types, tracks.scale, strict=True):
        axes.append(
            {
                "name": name,
                "type": axis_type,
                "scale": scale,
            }
        )

    metadata = GeffMetadata(
        geff_version=geff_spec.__version__,
        directed=True,
        node_props_metadata={},
        edge_props_metadata={},
        axes=axes,
    )

    # Save segmentation if present

    # TODO: helper function in _export_segmentation file
    if tracks.segmentation is not None:
        seg_path = directory / "segmentation"

        seg_data = tracks.segmentation
        shape = seg_data.shape
        dtype = seg_data.dtype

        # TODO: this probably isn't a good chunk size - time should be 1?
        # TODO: export to tiffs
        chunk_size: tuple[int, ...] = (64, 64, 64)
        chunk_size = tuple(list(chunk_size) + [1] * (len(shape) - len(chunk_size)))
        chunk_size = chunk_size[: len(shape)]

        z = setup_zarr_array(
            seg_path,
            zarr_format=zarr_format,
            shape=shape,
            dtype=dtype,
            chunks=chunk_size,
        )

        if node_ids is not None:
            nodes_to_keep = np.asarray(nodes_to_keep)

            # to avoid having to copy the segmentation array entirely, loop over chunks,
            # and mask out the nodes that should be kept.
            chunk_ranges = [
                range(0, dim, chunk) for dim, chunk in zip(shape, chunk_size, strict=True)
            ]

            for starts in itertools.product(*chunk_ranges):
                slices = tuple(
                    slice(start, min(start + chunk, dim))
                    for start, chunk, dim in zip(starts, chunk_size, shape, strict=True)
                )

                block = seg_data[slices]
                mask = np.isin(block, nodes_to_keep)
                filtered = np.where(mask, block, 0)
                z[slices] = filtered

        else:
            z[:] = seg_data

        metadata.related_objects = [
            {
                "path": "../segmentation",
                "type": "labels",
                "label_prop": "seg_id",
            }
        ]

    # Filter the graph if node_ids is provided
    if node_ids is not None:
        graph = graph.filter(node_ids=nodes_to_keep).subgraph()

    # Save the graph in a 'tracks' folder
    tracks_path = directory / "tracks"
    graph.to_geff(geff_store=tracks_path, geff_metadata=metadata, zarr_format=zarr_format)


def split_position_attr(tracks: Tracks) -> tuple[td.graph.GraphView, list[str] | None]:
    # TODO: this exists in unsqueeze in geff somehow?
    """Spread the spatial coordinates to separate node attrs in order to export to geff
    format.

    Args:
        tracks (funtracks.data_model.Tracks): tracks object holding the graph to be
          converted.

    Returns:
        tuple[td.graph.GraphView, list[str] | None]: graph with a separate positional
            attribute for each coordinate, and the axis names used to store the
            separate attributes

    """
    pos_key = tracks.features.position_key

    if isinstance(pos_key, str):
        # Position is stored as a single attribute, need to split
        new_graph = tracks.graph.detach()
        new_graph = new_graph.filter().subgraph()

        # Register new attribute keys
        new_graph.add_node_attr_key("x", default_value=0.0)
        new_graph.add_node_attr_key("y", default_value=0.0)

        # Get all position values at once
        pos_values = new_graph.node_attrs()["pos"].to_numpy()
        ndim = pos_values.shape[1]

        if ndim == 2:
            new_keys = ["y", "x"]
            new_graph.update_node_attrs(
                attrs={"x": pos_values[:, 1], "y": pos_values[:, 0]},
                node_ids=new_graph.node_ids(),
            )
        elif ndim == 3:
            new_keys = ["z", "y", "x"]
            new_graph.add_node_attr_key("z", default_value=0.0)
            new_graph.update_node_attrs(
                attrs={
                    "x": pos_values[:, 2],
                    "y": pos_values[:, 1],
                    "z": pos_values[:, 0],
                },
                node_ids=new_graph.node_ids(),
            )
        new_graph.remove_node_attr_key(pos_key)
        return new_graph, new_keys
    elif pos_key is not None:
        # Position is already split into separate attributes
        return tracks.graph, list(pos_key)
    else:
        return tracks.graph, None
