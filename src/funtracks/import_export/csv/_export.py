"""Export tracks to CSV format."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import pandas as pd

from .._export_segmentation import export_segmentation
from .._utils import filter_graph_with_ancestors

if TYPE_CHECKING:
    from funtracks.data_model.solution_tracks import SolutionTracks


def export_to_csv(
    tracks: SolutionTracks,
    outfile: Path | str,
    color_dict: dict[int, np.ndarray] | None = None,
    node_ids: set[int] | None = None,
    use_display_names: bool = False,
    export_seg: bool = False,
    seg_path: Path | str | None = None,
    seg_label_attr: str | None = "track_id",
    seg_file_format: Literal["zarr", "tiff"] = "zarr",
    zarr_format: Literal[2, 3] = 2,
) -> None:
    """Export tracks to a CSV file.
    TODO: export_all = False for backward compatibility - display names option shouldn't
    change which columns are exported, just using which names

    Exports tracking data to CSV format with columns for node ID, parent ID,
    and all registered features. Optionally also exports the segmentation as zarr or
    tiff. If a color dictionary is provided, it will also export the tracklet colors.

    Args:
        tracks: SolutionTracks object containing the tracking data to export
        outfile: Path to output CSV file
        color_dict: dict[int, np.ndarray], optional. If provided, will be used to save the
            hex colors.
        node_ids: Optional set of node IDs to include. If provided, only these
            nodes and their ancestors will be included in the output.
        use_display_names: If True, use feature display names as column headers.
            If False (default), use raw feature keys for backward compatibility.
        export_seg: Whether to export the segmentation alongside the CSV.
        seg_path: Path to save the segmentation to. Required when export_seg=True.
        seg_label_attr: Node attribute used to paint cell labels in the exported
            segmentation. Defaults to "track_id". When None, original segmentation
            labels (node IDs) are preserved.
        seg_file_format: Output format for the segmentation, either "zarr" or "tiff".
            Defaults to "zarr".
        zarr_format: Zarr format version. Only used when seg_file_format="zarr".
            Defaults to 2.

    Example:
        >>> from funtracks.import_export import export_to_csv
        >>> export_to_csv(tracks, "output.csv")
        >>> # Export with display names
        >>> export_to_csv(tracks, "output.csv", use_display_names=True)
        >>> # Export only specific nodes
        >>> export_to_csv(tracks, "filtered.csv", node_ids={1, 2, 3})
        >>> # Export with segmentation as zarr painted by track_id
        >>> export_to_csv(tracks, "out.csv", export_seg=True, seg_path="seg_zarr")
        >>> # Export with segmentation as tiff, original labels
        >>> export_to_csv(tracks, "out.csv", export_seg=True, seg_path="seg.tif",
        ...               seg_label_attr=None, seg_file_format="tiff")
    """

    def convert_numpy_to_python(value):
        """Convert numpy types to native Python types."""
        if isinstance(value, (np.float64, np.float32, np.float16)):
            return float(value)
        elif isinstance(value, (np.int64, np.int32, np.int16)):
            return int(value)
        elif isinstance(value, (np.bool_, bool)):
            return bool(value)
        return value

    header: list[str] = []
    column_map: dict[str, str | list[str]] = {}

    # Build header - use old hardcoded format for backward compatibility
    if use_display_names:
        header.extend(["ID", "Parent ID"])
        column_map["id"] = "ID"
        column_map["parent_id"] = "Parent ID"
    else:
        # time
        header.append("t")
        column_map["time"] = "t"

        # spatial coordinates
        coords = ["z", "y", "x"] if tracks.ndim == 4 else ["y", "x"]

        header.extend(coords)
        column_map["coords"] = coords

        # identifiers
        header.extend(["id", "parent_id", "track_id"])
        column_map["id"] = "id"
        column_map["parent_id"] = "parent_id"
        column_map["track_id"] = "track_id"

    # For display names mode, build dynamic header from features
    feature_names = []
    if use_display_names:
        for feature_name, feature_dict in tracks.features.items():
            feature_names.append(feature_name)
            num_values = feature_dict.get("num_values", 1)

            if num_values > 1:
                # Multi-value feature: use value_names if available
                value_names = feature_dict.get("value_names")
                if value_names is not None:
                    names = list(value_names)
                else:
                    # Fall back to display_name or feature_name with index suffix
                    base_name = feature_dict.get("display_name", feature_name)
                    if (
                        isinstance(base_name, (list, tuple))
                        and len(base_name) == num_values
                    ):
                        # use list elements
                        names = list(base_name)
                    else:
                        # use a suffix
                        names = [f"{base_name}_{i}" for i in range(num_values)]
                header.extend(names)
            else:
                # Single-value feature: use display_name or feature_name
                names = feature_dict.get("display_name", feature_name)
                header.extend([names])

            column_map[feature_name] = names

    # Determine which nodes to export
    if node_ids is None:
        node_to_keep = tracks.graph.node_ids()
    else:
        node_to_keep = filter_graph_with_ancestors(tracks.graph, node_ids)

    # Write CSV file
    rows: list[dict[str, Any]] = []

    for node_id in node_to_keep:
        parents = list(tracks.graph.predecessors(node_id))
        parent_id = "" if len(parents) == 0 else parents[0]

        row: dict[str, Any]

        row = {}
        row[cast(str, column_map["id"])] = node_id
        row[cast(str, column_map["parent_id"])] = parent_id

        if use_display_names:
            for feature_name in feature_names:
                value = tracks.get_node_attr(node_id, feature_name)
                cols = column_map[feature_name]
                if isinstance(cols, list):
                    if not isinstance(value, (list, tuple)):
                        value = list(value)
                    for col, v in zip(cols, value, strict=True):
                        row[col] = convert_numpy_to_python(v)
                else:
                    row[cols] = convert_numpy_to_python(value)

        else:
            row[cast(str, column_map["time"])] = convert_numpy_to_python(
                tracks.get_time(node_id)
            )

            pos = tracks.get_position(node_id)
            for name, value in zip(column_map["coords"], pos, strict=True):
                row[name] = convert_numpy_to_python(value)

            row[cast(str, column_map["track_id"])] = tracks.get_track_id(node_id)

        rows.append(row)

    df = pd.DataFrame(rows)
    df = df[header]

    # Also add a column with the track ID color
    if color_dict is not None:

        def rgb_to_hex(rgb):
            """Convert [R, G, B] to #RRGGBB."""
            r, g, b = [int(round(c * 255)) for c in rgb[:3]]  # scale and convert to int
            return f"#{r:02x}{g:02x}{b:02x}"

        track_id_to_hex = {}

        for track_id, nodes in tracks.track_id_to_node.items():
            if not nodes:
                continue
            first_node = nodes[0]
            rgb = color_dict[first_node]
            track_id_to_hex[track_id] = rgb_to_hex(rgb)

        df_colors = pd.DataFrame(
            list(track_id_to_hex.items()),  # convert dict to list of (track_id, hex)
            columns=[column_map["track_id"], "Tracklet ID Color"],
        )

        df = pd.merge(df, df_colors, how="left", on=[column_map["track_id"]])

    df.to_csv(outfile, index=False)

    if export_seg:
        if seg_path is None:
            raise ValueError("seg_path must be provided when export_seg=True")
        export_segmentation(
            tracks,
            Path(seg_path),
            file_format=seg_file_format,
            label_attr=seg_label_attr,
            zarr_format=zarr_format,
        )
