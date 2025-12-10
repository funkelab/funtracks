"""Export tracks to CSV format."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .._utils import filter_graph_with_ancestors

if TYPE_CHECKING:
    from funtracks.data_model.solution_tracks import SolutionTracks


def export_to_csv(
    tracks: SolutionTracks,
    outfile: Path | str,
    node_ids: set[int] | None = None,
    use_display_names: bool = False,
) -> None:
    """Export tracks to a CSV file.
    TODO: export_all = False for backward compatibility - display names option shouldn't
    change which columns are exported, just using which names

    Exports tracking data to CSV format with columns for node ID, parent ID,
    and all registered features.

    Args:
        tracks: SolutionTracks object containing the tracking data to export
        outfile: Path to output CSV file
        node_ids: Optional set of node IDs to include. If provided, only these
            nodes and their ancestors will be included in the output.
        use_display_names: If True, use feature display names as column headers.
            If False (default), use raw feature keys for backward compatibility.

    Example:
        >>> from funtracks.import_export import export_to_csv
        >>> export_to_csv(tracks, "output.csv")
        >>> # Export with display names
        >>> export_to_csv(tracks, "output.csv", use_display_names=True)
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

    # Build header - use old hardcoded format for backward compatibility
    if use_display_names:
        header = ["ID", "Parent ID"]
    else:
        # Backward compatibility: use old column names
        # Old format: t, [z], y, x, id, parent_id, track_id
        header = ["t"]
        if tracks.ndim == 4:
            header.extend(["z", "y", "x"])
        else:  # ndim == 3
            header.extend(["y", "x"])
        header.extend(["id", "parent_id", "track_id"])

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
                    header.extend(value_names)
                else:
                    # Fall back to display_name or feature_name with index suffix
                    base_name = feature_dict.get("display_name", feature_name)
                    header.extend([f"{base_name}_{i}" for i in range(num_values)])
            else:
                # Single-value feature: use display_name or feature_name
                col_name = feature_dict.get("display_name", feature_name)
                header.append(col_name)

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

            if use_display_names:
                # Dynamic feature export
                features: list[Any] = []
                for feature_name in feature_names:
                    feature_value = tracks.get_node_attr(node_id, feature_name)
                    if isinstance(feature_value, list | tuple):
                        features.extend(feature_value)
                    else:
                        features.append(feature_value)
                row = [node_id, parent_id, *features]
            else:
                # Backward compatibility: hardcoded format matching old behavior
                time = tracks.get_time(node_id)
                position = tracks.get_position(node_id)
                track_id = tracks.get_track_id(node_id)
                row = [time, *position, node_id, parent_id, track_id]

            row = [convert_numpy_to_python(value) for value in row]
            f.write("\n")
            f.write(",".join(map(str, row)))
