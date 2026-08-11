import numpy as np
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
    tracks = get_tracks(ndim=ndim, with_seg=False, prefill_track_ids=True)
    temp_file = tmp_path / "test_export.csv"
    export_to_csv(tracks, temp_file)

    with open(temp_file) as f:
        lines = f.readlines()

    assert len(lines) == tracks.graph_solution.num_nodes() + 1  # add header
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
    tracks = get_tracks(ndim=ndim, with_seg=True, prefill_track_ids=True)
    temp_file = tmp_path / "test_export.csv"
    seg_dir = tmp_path / "test_export_seg"
    export_to_csv(tracks, temp_file, export_seg=True, seg_path=seg_dir)

    with open(temp_file) as f:
        lines = f.readlines()

    assert len(lines) == tracks.graph_solution.num_nodes() + 1  # add header
    assert lines[0].strip().split(",") == expected_header

    # check the segmentation zarr
    seg_zarr = zarr.open(str(seg_dir), mode="r")
    assert isinstance(seg_zarr, zarr.Array)
    assert seg_zarr.shape == tracks.segmentation.shape

    # values should be tracklet_ids (not node_ids) — default seg_relabel="tracklet"
    seg_arr = seg_zarr[:]
    unique_vals = set(seg_arr.flatten()) - {0}
    label_key = tracks.features.tracklet_key
    track_ids = set(
        tracks.graph_solution.node_attrs(attr_keys=[label_key])[label_key].to_list()
    )
    assert unique_vals == track_ids


@pytest.mark.parametrize("ndim", [3, 4], ids=["2d", "3d"])
def test_export_solution_to_csv_with_seg_tiff(get_tracks, tmp_path, ndim):
    """Test exporting tracks to CSV + segmentation as tiff painted by tracklet ID."""
    tracks = get_tracks(ndim=ndim, with_seg=True, prefill_track_ids=True)
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

    # values should be tracklet_ids (not node_ids) — default seg_relabel="tracklet"
    unique_vals = set(seg_arr.flatten()) - {0}
    label_key = tracks.features.tracklet_key
    track_ids = set(
        tracks.graph_solution.node_attrs(attr_keys=[label_key])[label_key].to_list()
    )
    assert unique_vals == track_ids


@pytest.mark.parametrize("ndim", [3, 4], ids=["2d", "3d"])
def test_export_solution_to_csv_with_seg_original_labels(get_tracks, tmp_path, ndim):
    """Test exporting tracks to CSV + segmentation with original (node_id) labels."""
    tracks = get_tracks(ndim=ndim, with_seg=True, prefill_track_ids=True)
    temp_file = tmp_path / "test_export.csv"
    seg_dir = tmp_path / "test_export_seg"
    export_to_csv(
        tracks,
        temp_file,
        export_seg=True,
        seg_path=seg_dir,
        seg_relabel=None,
    )

    seg_zarr = zarr.open(str(seg_dir), mode="r")
    assert isinstance(seg_zarr, zarr.Array)
    assert seg_zarr.shape == tracks.segmentation.shape

    # values should be node_ids (original labels), not track_ids
    seg_arr = seg_zarr[:]
    unique_vals = set(seg_arr.flatten()) - {0}
    node_ids = set(tracks.graph_solution.node_ids())
    assert unique_vals == node_ids


def test_export_with_color_dict(get_tracks, tmp_path):
    """Test exporting with a color_dict adds a Tracklet ID Color column."""
    import numpy as np

    tracks = get_tracks(ndim=3, with_seg=False, prefill_track_ids=True)
    temp_file = tmp_path / "test_export_colors.csv"

    # Build a color dict: node_id → [R, G, B] floats in [0, 1]
    node_ids = list(tracks.graph_solution.node_ids())
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
    tracks = get_tracks(ndim=3, with_seg=False, prefill_track_ids=True)
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
    tracks = get_tracks(ndim=3, with_seg=False, prefill_track_ids=True)
    temp_file = tmp_path / "test_export_filtered.csv"

    # Export only nodes 1 and 2 (and their ancestors)
    export_to_csv(tracks, temp_file, node_ids={2})

    with open(temp_file) as f:
        lines = f.readlines()

    # Should have header + node 2 + node 1 (ancestor)
    assert len(lines) == 3  # header + 2 nodes


def test_ignore_edge_features_at_export(get_tracks, tmp_path):
    """Test that edge features are ignored when exporting to csv"""

    tracks = get_tracks(ndim=3, with_seg=True, prefill_track_ids=True)
    temp_file = tmp_path / "test_export_node_features_only.csv"

    # enable node and edge features
    tracks.enable_features(["iou", "area"])

    # export should not fail
    export_to_csv(tracks, temp_file, use_display_names=True)

    # Ensure that node feature 'Area' is present but edge feature 'IoU' is not
    with open(temp_file) as f:
        lines = f.readlines()

    assert "Area" in lines[0]
    assert "IoU" not in lines[0]


@pytest.mark.parametrize("seg_relabel", ["tracklet", "lineage", None])
@pytest.mark.parametrize("seg_file_format", ["zarr", "tiff"])
def test_export_solution_to_csv_with_seg_and_node_subset(
    get_tracks,
    tmp_path,
    seg_relabel,
    seg_file_format,
):
    """
    Test CSV export with segmentation + node subset filtering.

    node_ids passed to export_to_csv are expanded (including ancestors),
    and segmentation must only include the resulting graph nodes, and nothing else.
    """

    tracks = get_tracks(ndim=3, with_seg=True, prefill_track_ids=True)

    csv_file = tmp_path / "export.csv"
    seg_path = tmp_path / "seg"

    export_to_csv(
        tracks,
        csv_file,
        node_ids={4, 6},
        export_seg=True,
        seg_path=seg_path,
        seg_file_format=seg_file_format,
        seg_relabel=seg_relabel,
    )

    # 1. Validate CSV node export
    with open(csv_file) as f:
        lines = f.readlines()

    assert len(lines) >= 2  # header + at least 1 row

    header = lines[0].strip().split(",")
    assert "id" in header

    exported_ids = {int(line.split(",")[header.index("id")]) for line in lines[1:]}

    # CSV should include expanded graph nodes (NOT just requested subset)
    expected_graph_nodes = {1, 3, 4, 6}
    assert exported_ids == expected_graph_nodes

    # 2. Validate segmentation
    if seg_file_format == "zarr":
        seg = zarr.open(str(seg_path), mode="r")
        seg_arr = np.asarray(seg[:])
    else:
        seg_arr = tifffile.imread(str(seg_path))

    assert seg_arr.shape == tracks.segmentation.shape

    unique_vals = set(seg_arr.flatten()) - {0}

    original = np.asarray(tracks.segmentation[:])

    graph_nodes = expected_graph_nodes

    if seg_relabel is None:
        expected = np.where(np.isin(original, list(graph_nodes)), original, 0)

        np.testing.assert_array_equal(seg_arr, expected)
        assert unique_vals == graph_nodes

    else:
        if seg_relabel == "lineage":
            label_key = tracks.features.lineage_key
        else:
            label_key = tracks.features.tracklet_key

        labels = tracks.graph_solution.node_attrs(attr_keys=[label_key])[
            label_key
        ].to_list()
        node_to_label = dict(zip(tracks.graph_solution.node_ids(), labels, strict=True))

        expected = np.zeros_like(original)

        for n in graph_nodes:
            expected[original == n] = node_to_label[n]

        np.testing.assert_array_equal(seg_arr, expected)

        expected_vals = {node_to_label[n] for n in graph_nodes}
        assert unique_vals == expected_vals


def test_export_full_vs_solution(get_tracks, tmp_path):
    """export_full=True includes soft-deleted (solution=False) nodes and surfaces the
    'Solution' column; the default exports only the solution view."""
    import pandas as pd

    from funtracks.user_actions import UserDeleteNode

    tracks = get_tracks(ndim=3, with_seg=False, prefill_track_ids=True)
    # Soft-delete a leaf node: dropped from the solution view, kept in graph_full.
    UserDeleteNode(tracks, 5)
    sol_n = tracks.graph_solution.num_nodes()
    full_n = tracks.graph_full.num_nodes()
    assert full_n == sol_n + 1

    # Default: solution view only — node 5 absent.
    sol_file = tmp_path / "sol.csv"
    export_to_csv(tracks, sol_file)
    sol = pd.read_csv(sol_file)
    assert len(sol) == sol_n
    assert 5 not in set(sol["id"])

    # export_full=True: full graph — node 5 present.
    full_file = tmp_path / "full.csv"
    export_to_csv(tracks, full_file, export_full=True)
    full = pd.read_csv(full_file)
    assert len(full) == full_n
    assert 5 in set(full["id"])

    # With display names, the full export carries a 'Solution' column distinguishing
    # the soft-deleted node (False) from the rest (True).
    disp_file = tmp_path / "full_disp.csv"
    export_to_csv(tracks, disp_file, export_full=True, use_display_names=True)
    disp = pd.read_csv(disp_file)
    assert "Solution" in disp.columns
    assert not bool(disp.loc[disp["ID"] == 5, "Solution"].iloc[0])
    assert bool(disp.loc[disp["ID"] == 1, "Solution"].iloc[0])
