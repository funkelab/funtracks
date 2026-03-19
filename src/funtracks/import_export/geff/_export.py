from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Literal,
)

import geff_spec
import polars as pl
import tracksdata as td
from geff_spec import GeffMetadata

from funtracks.utils import remove_tilde, setup_zarr_group

from .._export_segmentation import export_segmentation
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
    save_segmentation: bool = True,
    seg_label_attr: str | None = "track_id",
    seg_file_format: Literal["zarr", "tiff"] = "zarr",
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
        save_segmentation (bool): If True, saves the segmentation array alongside
            the graph when segmentation is present. Defaults to True.
        seg_label_attr (str | None): Node attribute used to paint cell labels in the
            exported segmentation. Defaults to "track_id". When None, original
            segmentation labels (node IDs) are preserved.
        seg_file_format: Output format for the segmentation, either "zarr" or "tiff".
            Defaults to "zarr".
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

    # Save segmentation if present and requested
    if save_segmentation and tracks.segmentation is not None:
        seg_name = "segmentation.tif" if seg_file_format == "tiff" else "segmentation"
        export_segmentation(
            tracks,
            directory / seg_name,
            file_format=seg_file_format,
            label_attr=seg_label_attr,
            zarr_format=zarr_format,
            node_ids=set(nodes_to_keep) if node_ids is not None else None,
        )
        metadata.related_objects = [
            {
                "path": f"../{seg_name}",
                "type": "labels",
                "label_prop": seg_label_attr or "node_id",
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
        new_graph.add_node_attr_key("x", default_value=0.0, dtype=pl.Float64)
        new_graph.add_node_attr_key("y", default_value=0.0, dtype=pl.Float64)

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
            new_graph.add_node_attr_key("z", default_value=0.0, dtype=pl.Float64)
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
