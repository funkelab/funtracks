"""Export tracks to CSV format."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from funtracks.import_export.export_utils import filter_graph_with_ancestors

if TYPE_CHECKING:
    from funtracks.data_model.solution_tracks import SolutionTracks


def export_to_csv(
    tracks: SolutionTracks,
    outfile: Path | str,
    node_ids: set[int] | None = None,
) -> None:
    """Export tracks to a CSV file.

    Exports tracking data to CSV format with columns for node ID, parent ID,
    and all registered features. The features are exported using their display names.

    Args:
        tracks: SolutionTracks object containing the tracking data to export
        outfile: Path to output CSV file
        node_ids: Optional set of node IDs to include. If provided, only these
            nodes and their ancestors will be included in the output.

    Example:
        >>> from funtracks.import_export import export_to_csv
        >>> export_to_csv(tracks, "output.csv")
        >>> # Export only specific nodes
        >>> export_to_csv(tracks, "filtered.csv", node_ids={1, 2, 3})
    """

    def convert_numpy_to_python(value):
        """Convert numpy types to native Python types."""
        if isinstance(value, (np.float64, np.float32, np.float16)):
            return float(value)
        elif isinstance(value, (np.int64, np.int32, np.int16)):
            return int(value)
        return value

    # Build header from features
    header = ["ID", "Parent ID"]
    feature_names = []
    for feature_name, feature_dict in tracks.features.items():
        col_name = feature_dict["display_name"]
        feature_names.append(feature_name)
        if isinstance(col_name, (list, tuple)):
            header.extend(col_name)
        else:
            header.append(cast(str, col_name))

    # Determine which nodes to export
    if node_ids is None:
        node_to_keep = tracks.graph.nodes()
    else:
        node_to_keep = filter_graph_with_ancestors(tracks.graph, node_ids)

    # Write CSV file
    with open(outfile, "w") as f:
        f.write(",".join(header))
        for node_id in node_to_keep:
            parents = list(tracks.graph.predecessors(node_id))
            parent_id = "" if len(parents) == 0 else parents[0]
            features: list[Any] = []
            for feature_name in feature_names:
                feature_value = tracks.get_node_attr(node_id, feature_name)
                if isinstance(feature_value, list | tuple):
                    features.extend(feature_value)
                else:
                    features.append(feature_value)
            row = [node_id, parent_id, *features]
            row = [convert_numpy_to_python(value) for value in row]
            f.write("\n")
            f.write(",".join(map(str, row)))
