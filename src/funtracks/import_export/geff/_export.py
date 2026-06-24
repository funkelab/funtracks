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

from .._export_segmentation import export_segmentation, resolve_relabel_attr
from .._utils import filter_graph_with_ancestors

if TYPE_CHECKING:
    from pathlib import Path

    from funtracks.data_model.tracks import Tracks


def write_to_geff(
    tracks: Tracks,
    path: Path,
    overwrite: bool = False,
    zarr_format: Literal[2, 3] = 2,
):
    """Write tracks directly to a geff store at the given path.

    Unlike :func:`export_to_geff` (which creates a parent zarr container with
    a ``tracks.geff`` subfolder and optional segmentation), this writes the
    geff store directly to *path*.  Intended for internal save/load workflows
    where the user picks the ``.geff`` path.

    Args:
        tracks: Tracks object containing a graph to save.
        path: Destination path for the geff store.
        overwrite: If True, overwrites an existing store at *path*.
        zarr_format: Zarr format version to use. Defaults to 2.
    """
    path = remove_tilde(path)
    path = path.resolve(strict=False)

    graph, metadata = _build_geff_metadata(tracks, include_features=True)
    graph.to_geff(
        geff_store=path,
        geff_metadata=metadata,
        zarr_format=zarr_format,
        overwrite=overwrite,
    )
    _write_segmentation_shape(path, tracks)


def export_to_geff(
    tracks: Tracks,
    directory: Path,
    overwrite: bool = False,
    node_ids: set[int] | None = None,
    zarr_format: Literal[2, 3] = 2,
    save_segmentation: bool = True,
    seg_relabel: Literal["tracklet", "lineage", None] = "tracklet",
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
        seg_relabel: How to relabel cells in the exported segmentation.
            "tracklet" (default): paint by tracklet ID.
            "lineage": paint by lineage ID.
            None: preserve original labels (node IDs).
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

    # Include the FeatureDict in metadata only for full exports.
    # Subgroup exports do not necessarily have valid tracklet/lineage IDs
    # and thus are not valid SolutionTracks
    graph, metadata = _build_geff_metadata(tracks, include_features=(node_ids is None))

    # Save segmentation if present and requested
    if save_segmentation and tracks.segmentation is not None:
        seg_name = "segmentation.tif" if seg_file_format == "tiff" else "segmentation"
        # Tiff is saved next to (sibling of) the geff directory to avoid napari
        # misidentifying it as zarr when the geff directory has a .zarr extension.
        seg_parent = directory.parent if seg_file_format == "tiff" else directory
        rel_prefix = "../.." if seg_file_format == "tiff" else ".."
        export_segmentation(
            tracks,
            seg_parent / seg_name,
            file_format=seg_file_format,
            relabel=seg_relabel,
            zarr_format=zarr_format,
            node_ids=set(nodes_to_keep) if node_ids is not None else None,
        )
        label_prop = resolve_relabel_attr(tracks, seg_relabel) or "node_id"
        metadata.related_objects = [
            {
                "path": f"{rel_prefix}/{seg_name}",
                "type": "labels",
                "label_prop": label_prop,
            }
        ]

    # Filter the graph if node_ids is provided
    if node_ids is not None:
        graph = graph.filter(node_ids=nodes_to_keep).subgraph()

    # Save the graph in a 'tracks.geff' folder
    tracks_path = directory / "tracks.geff"
    graph.to_geff(geff_store=tracks_path, geff_metadata=metadata, zarr_format=zarr_format)

    _write_segmentation_shape(tracks_path, tracks)


def _build_geff_metadata(
    tracks: Tracks,
    include_features: bool = True,
) -> tuple[td.graph.GraphView, GeffMetadata]:
    """Build the geff metadata and prepare the graph for writing.

    Returns the (possibly modified) graph with split position attributes
    and the GeffMetadata object.
    """
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

    extra: dict = {}
    if include_features:
        extra["funtracks"] = {"features": tracks.features.dump_json()}

    metadata = GeffMetadata(
        geff_version=geff_spec.__version__,
        directed=True,
        node_props_metadata={},
        edge_props_metadata={},
        axes=axes,
        extra=extra,
    )

    return graph, metadata


def _write_segmentation_shape(geff_path: Path, tracks: Tracks) -> None:
    """Write segmentation_shape as an extra zarr attribute when masks are present.

    GeffMetadata has no segmentation_shape field, so it must be stored separately.
    This allows import_from_geff to reconstruct the segmentation (GraphArrayView)
    without requiring an external segmentation file.
    """
    seg_shape = tracks.graph.metadata.get("segmentation_shape")
    if seg_shape is not None:
        import zarr as _zarr

        z = _zarr.open(str(geff_path), mode="a")
        attrs = dict(z.attrs)
        attrs["segmentation_shape"] = list(seg_shape)
        z.attrs.update(attrs)


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
