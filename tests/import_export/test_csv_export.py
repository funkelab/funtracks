import pytest

from funtracks.import_export import export_to_csv


@pytest.mark.parametrize(
    ("ndim", "expected_header"),
    [
        (3, ["t", "y", "x", "id", "parent_id", "track_id"]),
        (4, ["t", "z", "y", "x", "id", "parent_id", "track_id"]),
    ],
    ids=["2d", "3d"],
)
def test_export_solution_to_csv(get_tracks, tmp_path, ndim, expected_header):
    """Test exporting tracks to CSV."""
    tracks = get_tracks(ndim=ndim, with_seg=False, is_solution=True)
    temp_file = tmp_path / "test_export.csv"
    export_to_csv(tracks, temp_file)

    with open(temp_file) as f:
        lines = f.readlines()

    assert len(lines) == tracks.graph.number_of_nodes() + 1  # add header
    assert lines[0].strip().split(",") == expected_header

    # Check first data line (node 1: t=0, pos=[50, 50] or [50, 50, 50], track_id=1)
    if ndim == 3:
        expected_line1 = ["0", "50", "50", "1", "", "1"]
    else:
        expected_line1 = ["0", "50", "50", "50", "1", "", "1"]
    assert lines[1].strip().split(",") == expected_line1


def test_export_with_display_names(get_tracks, tmp_path):
    """Test exporting with display names."""
    tracks = get_tracks(ndim=3, with_seg=False, is_solution=True)
    temp_file = tmp_path / "test_export_display.csv"
    export_to_csv(tracks, temp_file, use_display_names=True)

    with open(temp_file) as f:
        lines = f.readlines()

    # Should have ID and Parent ID columns
    header = lines[0].strip().split(",")
    assert "ID" in header
    assert "Parent ID" in header


def test_export_filtered_nodes(get_tracks, tmp_path):
    """Test exporting only specific nodes."""
    tracks = get_tracks(ndim=3, with_seg=False, is_solution=True)
    temp_file = tmp_path / "test_export_filtered.csv"

    # Export only nodes 1 and 2 (and their ancestors)
    export_to_csv(tracks, temp_file, node_ids={2})

    with open(temp_file) as f:
        lines = f.readlines()

    # Should have header + node 2 + node 1 (ancestor)
    assert len(lines) == 3  # header + 2 nodes
