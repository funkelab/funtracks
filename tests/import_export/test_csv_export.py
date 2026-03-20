import pytest
import tifffile
import zarr

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

    assert len(lines) == tracks.graph.num_nodes() + 1  # add header
    assert lines[0].strip().split(",") == expected_header

    # Check first data line (node 1: t=0, pos=[50, 50] or [50, 50, 50], track_id=1)
    if ndim == 3:
        expected_line1 = ["0", "50.0", "50.0", "1", "", "1"]
    else:
        expected_line1 = ["0", "50.0", "50.0", "50.0", "1", "", "1"]
    assert lines[1].strip().split(",") == expected_line1


@pytest.mark.parametrize(
    ("ndim", "expected_header"),
    [
        (3, ["t", "y", "x", "id", "parent_id", "track_id"]),
        (4, ["t", "z", "y", "x", "id", "parent_id", "track_id"]),
    ],
    ids=["2d", "3d"],
)
def test_export_solution_to_csv_with_seg_zarr(
    get_tracks, tmp_path, ndim, expected_header
):
    """Test exporting tracks to CSV + segmentation painted by track_id as zarr."""
    tracks = get_tracks(ndim=ndim, with_seg=True, is_solution=True)
    temp_file = tmp_path / "test_export.csv"
    seg_dir = tmp_path / "test_export_seg"
    export_to_csv(tracks, temp_file, export_seg=True, seg_path=seg_dir)

    with open(temp_file) as f:
        lines = f.readlines()

    assert len(lines) == tracks.graph.num_nodes() + 1  # add header
    assert lines[0].strip().split(",") == expected_header

    # check the segmentation zarr
    seg_zarr = zarr.open(str(seg_dir), mode="r")
    assert isinstance(seg_zarr, zarr.Array)
    assert seg_zarr.shape == tracks.segmentation.shape

    # values should be track_ids (not node_ids) — default seg_label_attr="track_id"
    seg_arr = seg_zarr[:]
    unique_vals = set(seg_arr.flatten()) - {0}
    track_ids = set(tracks.graph.node_attrs(attr_keys=["track_id"])["track_id"].to_list())
    assert unique_vals == track_ids


@pytest.mark.parametrize("ndim", [3, 4], ids=["2d", "3d"])
def test_export_solution_to_csv_with_seg_tiff(get_tracks, tmp_path, ndim):
    """Test exporting tracks to CSV + segmentation as tiff painted by track_id."""
    tracks = get_tracks(ndim=ndim, with_seg=True, is_solution=True)
    temp_file = tmp_path / "test_export.csv"
    seg_file = tmp_path / "test_export_seg.tif"
    export_to_csv(
        tracks,
        temp_file,
        export_seg=True,
        seg_path=seg_file,
        seg_file_format="tiff",
    )

    assert seg_file.exists()
    seg_arr = tifffile.imread(str(seg_file))
    assert seg_arr.shape == tracks.segmentation.shape

    # values should be track_ids (not node_ids) — default seg_label_attr="track_id"
    unique_vals = set(seg_arr.flatten()) - {0}
    track_ids = set(tracks.graph.node_attrs(attr_keys=["track_id"])["track_id"].to_list())
    assert unique_vals == track_ids


@pytest.mark.parametrize("ndim", [3, 4], ids=["2d", "3d"])
def test_export_solution_to_csv_with_seg_original_labels(get_tracks, tmp_path, ndim):
    """Test exporting tracks to CSV + segmentation with original (node_id) labels."""
    tracks = get_tracks(ndim=ndim, with_seg=True, is_solution=True)
    temp_file = tmp_path / "test_export.csv"
    seg_dir = tmp_path / "test_export_seg"
    export_to_csv(
        tracks,
        temp_file,
        export_seg=True,
        seg_path=seg_dir,
        seg_label_attr=None,
    )

    seg_zarr = zarr.open(str(seg_dir), mode="r")
    assert isinstance(seg_zarr, zarr.Array)
    assert seg_zarr.shape == tracks.segmentation.shape

    # values should be node_ids (original labels), not track_ids
    seg_arr = seg_zarr[:]
    unique_vals = set(seg_arr.flatten()) - {0}
    node_ids = set(tracks.graph.node_ids())
    assert unique_vals == node_ids


def test_export_with_color_dict(get_tracks, tmp_path):
    """Test exporting with a color_dict adds a Tracklet ID Color column."""
    import numpy as np

    tracks = get_tracks(ndim=3, with_seg=False, is_solution=True)
    temp_file = tmp_path / "test_export_colors.csv"

    # Build a color dict: node_id → [R, G, B] floats in [0, 1]
    node_ids = list(tracks.graph.node_ids())
    color_dict = {
        node_id: np.array([0.1 * (i % 10), 0.5, 0.9])
        for i, node_id in enumerate(node_ids)
    }

    export_to_csv(tracks, temp_file, color_dict=color_dict)

    import pandas as pd

    df = pd.read_csv(temp_file)
    assert "Tracklet ID Color" in df.columns
    # All color values should be valid hex strings
    assert df["Tracklet ID Color"].str.match(r"^#[0-9a-f]{6}$").all()


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
