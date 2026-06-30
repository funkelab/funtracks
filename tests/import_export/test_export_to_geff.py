import numpy as np
import polars as pl
import pytest
import tifffile
import zarr

from funtracks.data_model import Tracks
from funtracks.import_export import export_to_geff, import_from_geff, write_to_geff


def _assert_valid_geff_export(export_dir, expected_num_nodes=None):
    """Assert basic export correctness and return the opened zarr Group."""
    z = zarr.open((export_dir / "tracks.geff").as_posix(), mode="r")
    assert isinstance(z, zarr.Group)

    attrs = dict(z.attrs)
    assert "geff" in attrs
    assert "axes" in attrs["geff"]
    for ax in attrs["geff"]["axes"]:
        assert ax["scale"] is not None

    if expected_num_nodes is not None:
        assert len(z["nodes/ids"][:]) == expected_num_nodes

    return z


# --- Segmentation export ---


@pytest.mark.parametrize("ndim", [3, 4])
@pytest.mark.parametrize("seg_relabel", ["tracklet", "lineage", None])
def test_export_segmentation_relabel(get_tracks, ndim, seg_relabel, tmp_path):
    """Test segmentation export with each relabel strategy."""
    tracks = get_tracks(ndim=ndim, with_seg=True, is_solution=True)

    export_dir = tmp_path / "export"
    export_dir.mkdir()
    export_to_geff(tracks, export_dir, seg_relabel=seg_relabel)

    z = _assert_valid_geff_export(export_dir, tracks.graph_solution.num_nodes())

    # Segmentation file must exist
    seg_path = export_dir / "segmentation"
    seg_zarr = zarr.open(str(seg_path), mode="r")
    assert isinstance(seg_zarr, zarr.Array)
    assert seg_zarr.shape == tracks.segmentation.shape

    # Verify pixel values match relabel strategy
    unique_vals = set(seg_zarr[:].flatten()) - {0}
    if seg_relabel is not None:
        if seg_relabel == "lineage":
            label_key = tracks.features.lineage_key
        else:
            label_key = tracks.features.tracklet_key
        label_vals = set(
            tracks.graph_solution.node_attrs(attr_keys=[label_key])[label_key].to_list()
        )
        assert unique_vals == label_vals
    else:
        # values should be original node_ids
        node_ids_set = set(tracks.graph_solution.node_ids())
        assert unique_vals == node_ids_set

    # segmentation_shape must be in metadata
    attrs = dict(z.attrs)
    assert "segmentation_shape" in attrs
    assert tuple(attrs["segmentation_shape"]) == tracks.segmentation.shape


def test_export_no_segmentation_saved(get_tracks, tmp_path):
    """Test that save_segmentation=False suppresses segmentation file."""
    tracks = get_tracks(ndim=3, with_seg=True, is_solution=True)

    export_dir = tmp_path / "export"
    export_dir.mkdir()
    export_to_geff(tracks, export_dir, save_segmentation=False)

    z = _assert_valid_geff_export(export_dir, tracks.graph_solution.num_nodes())

    assert not (export_dir / "segmentation").exists()

    # segmentation_shape is still written (from graph metadata, not from the file)
    attrs = dict(z.attrs)
    assert "segmentation_shape" in attrs


def test_export_without_seg_on_tracks(get_tracks, tmp_path):
    """Test export when tracks have no segmentation at all."""
    tracks = get_tracks(ndim=3, with_seg=False, is_solution=True)

    export_dir = tmp_path / "export"
    export_dir.mkdir()
    export_to_geff(tracks, export_dir)

    z = _assert_valid_geff_export(export_dir, tracks.graph_solution.num_nodes())

    assert not (export_dir / "segmentation").exists()

    attrs = dict(z.attrs)
    assert "segmentation_shape" not in attrs


def test_export_segmentation_non_solution(get_tracks, tmp_path):
    """Non-solution tracks export segmentation fine with seg_relabel=None."""
    tracks = get_tracks(ndim=3, with_seg=True, is_solution=False)

    export_dir = tmp_path / "export"
    export_dir.mkdir()
    export_to_geff(tracks, export_dir, seg_relabel=None)

    z = _assert_valid_geff_export(export_dir, tracks.graph_solution.num_nodes())

    # No relabel: segmentation pixels keep original node_ids
    seg_zarr = zarr.open(str(export_dir / "segmentation"), mode="r")
    assert isinstance(seg_zarr, zarr.Array)
    assert seg_zarr.shape == tracks.segmentation.shape
    assert set(seg_zarr[:].flatten()) - {0} == set(tracks.graph_solution.node_ids())

    attrs = dict(z.attrs)
    assert "segmentation_shape" in attrs
    assert tuple(attrs["segmentation_shape"]) == tracks.segmentation.shape


# --- Position attribute splitting ---


@pytest.mark.parametrize("ndim", [3, 4])
@pytest.mark.parametrize("is_solution", [True, False])
def test_export_split_position_attrs(get_graph, ndim, is_solution, tmp_path):
    """Test export with split (list) position attributes."""
    graph = get_graph(ndim, is_solution=is_solution, with_seg=False)

    pos_keys = ["y", "x"] if ndim == 3 else ["z", "y", "x"]
    for key in pos_keys:
        graph.add_node_attr_key(key, default_value=0.0, dtype=pl.Float64)
    for node in graph.node_ids():
        pos = graph.nodes[node]["pos"]
        for i, key in enumerate(pos_keys):
            graph.nodes[node][key] = pos[i]
    graph.remove_node_attr_key("pos")

    tracks = Tracks(
        graph,
        time_attr="t",
        pos_attr=pos_keys,
        tracklet_attr="track_id" if is_solution else None,
        ndim=ndim,
    )

    export_dir = tmp_path / "export"
    export_dir.mkdir()
    export_to_geff(tracks, export_dir, save_segmentation=False)

    z = _assert_valid_geff_export(export_dir, tracks.graph_solution.num_nodes())

    # Verify axis names include the split position keys
    axes = dict(z.attrs)["geff"]["axes"]
    axis_names = [ax["name"] for ax in axes]
    for key in pos_keys:
        assert key in axis_names


# --- Node subset export ---


@pytest.mark.parametrize("ndim", [3, 4])
def test_export_node_subset(get_tracks, ndim, tmp_path):
    """Test exporting a subset of nodes includes ancestors."""
    tracks = get_tracks(ndim=ndim, with_seg=True, is_solution=True)

    export_dir = tmp_path / "export"
    export_dir.mkdir()

    # Nodes 4 and 6 requested; ancestors 1 and 3 should be included automatically
    node_ids = [4, 6]
    export_to_geff(tracks, export_dir, node_ids=node_ids, seg_relabel=None)

    expected_graph_nodes = np.array([1, 3, 4, 6])
    z = _assert_valid_geff_export(export_dir, len(expected_graph_nodes))

    node_ids_array = z["nodes/ids"][:]
    assert np.array_equal(np.sort(node_ids_array), expected_graph_nodes)

    # Segmentation filtered to subset nodes (seg_relabel=None for direct ID check)
    seg_path = export_dir / "segmentation"
    seg_zarr = zarr.open(str(seg_path), mode="r")
    assert seg_zarr.shape == tracks.segmentation.shape

    exported = np.asarray(seg_zarr[:])
    original = np.asarray(tracks.segmentation[:])
    graph_nodes = set(expected_graph_nodes)

    expected = np.where(np.isin(original, list(graph_nodes)), original, 0)
    np.testing.assert_array_equal(exported, expected)
    assert set(exported.flatten()) - {0} == graph_nodes


def test_export_node_subset_seg_relabel(get_tracks, tmp_path):
    """Test subset export with relabeled segmentation."""
    tracks = get_tracks(ndim=3, with_seg=True, is_solution=True)

    export_dir = tmp_path / "export"
    export_dir.mkdir()

    node_ids = [4, 6]
    export_to_geff(tracks, export_dir, node_ids=node_ids, seg_relabel="tracklet")

    expected_graph_nodes = np.array([1, 3, 4, 6])
    _assert_valid_geff_export(export_dir, len(expected_graph_nodes))

    seg_zarr = zarr.open(str(export_dir / "segmentation"), mode="r")
    exported = np.asarray(seg_zarr[:])
    original = np.asarray(tracks.segmentation[:])

    label_key = tracks.features.tracklet_key
    labels = tracks.graph_solution.node_attrs(attr_keys=[label_key])[label_key]
    node_to_label = dict(
        zip(tracks.graph_solution.node_ids(), labels.to_list(), strict=True)
    )

    graph_nodes = set(expected_graph_nodes)
    expected = np.zeros_like(original)
    for n in graph_nodes:
        expected[original == n] = node_to_label[n]
    np.testing.assert_array_equal(exported, expected)

    unique_vals = set(exported.flatten()) - {0}
    expected_vals = {node_to_label[n] for n in graph_nodes}
    assert unique_vals == expected_vals


def test_export_node_subset_without_seg(get_tracks, tmp_path):
    """Test subset export when tracks have no segmentation."""
    tracks = get_tracks(ndim=3, with_seg=False, is_solution=True)

    export_dir = tmp_path / "export"
    export_dir.mkdir()

    export_to_geff(tracks, export_dir, node_ids=[4, 6])

    expected_graph_nodes = np.array([1, 3, 4, 6])
    _assert_valid_geff_export(export_dir, len(expected_graph_nodes))

    assert not (export_dir / "segmentation").exists()


# --- Overwrite and error handling ---


def test_export_overwrite(get_tracks, tmp_path):
    """Test export with overwrite=True into non-empty directory."""
    tracks = get_tracks(ndim=3, with_seg=True, is_solution=True)

    export_dir = tmp_path / "export"
    export_dir.mkdir()
    (export_dir / "existing_file.txt").write_text("already here")

    export_to_geff(tracks, export_dir, overwrite=True)

    _assert_valid_geff_export(export_dir, tracks.graph_solution.num_nodes())

    # Segmentation is still written correctly alongside the overwritten dir
    seg_zarr = zarr.open(str(export_dir / "segmentation"), mode="r")
    assert isinstance(seg_zarr, zarr.Array)
    assert seg_zarr.shape == tracks.segmentation.shape


def test_export_non_directory_raises(get_tracks, tmp_path):
    """Test that exporting to a file path (not a directory) raises an error."""
    tracks = get_tracks(ndim=3, with_seg=False, is_solution=True)

    file_path = tmp_path / "not_a_dir"
    file_path.write_text("test")

    with pytest.raises(Exception):  # noqa: B017
        export_to_geff(tracks, file_path)


# --- Metadata ---


@pytest.mark.parametrize("ndim", [3, 4])
@pytest.mark.parametrize("with_seg", [True, False])
def test_export_metadata(get_tracks, ndim, with_seg, tmp_path):
    """Test axes structure, segmentation_shape, and FeatureDict in metadata."""
    tracks = get_tracks(ndim=ndim, with_seg=with_seg, is_solution=True)

    export_dir = tmp_path / "export"
    export_dir.mkdir()
    export_to_geff(tracks, export_dir, save_segmentation=with_seg)

    z = _assert_valid_geff_export(export_dir, tracks.graph_solution.num_nodes())
    attrs = dict(z.attrs)

    # Correct number of axes
    axes = attrs["geff"]["axes"]
    assert len(axes) == ndim
    expected_types = (
        ["time", "space", "space"] if ndim == 3 else ["time", "space", "space", "space"]
    )
    assert [ax["type"] for ax in axes] == expected_types

    # segmentation_shape present iff with_seg
    if with_seg:
        assert "segmentation_shape" in attrs
        assert tuple(attrs["segmentation_shape"]) == tracks.segmentation.shape
    else:
        assert "segmentation_shape" not in attrs

    # FeatureDict stored in extra.funtracks
    assert "funtracks" in attrs["geff"].get("extra", {})
    assert "features" in attrs["geff"]["extra"]["funtracks"]


# --- Tiff segmentation export (unchanged) ---


@pytest.mark.parametrize("ndim", [3, 4], ids=["2d", "3d"])
def test_export_to_geff_seg_tiff(get_tracks, ndim, tmp_path):
    """Test that segmentation can be exported as tiff alongside the geff graph."""
    tracks = get_tracks(ndim=ndim, with_seg=True, is_solution=True)
    export_dir = tmp_path / "export"
    export_dir.mkdir()

    export_to_geff(tracks, export_dir, seg_file_format="tiff")

    # tiff is saved next to (sibling of) the geff directory, not inside it
    assert (tmp_path / "segmentation.tif").exists()
    assert not (export_dir / "segmentation.tif").exists()
    assert not (export_dir / "segmentation").exists()

    seg_arr = tifffile.imread(str(tmp_path / "segmentation.tif"))
    assert seg_arr.shape == tracks.segmentation.shape

    # values should be tracklet_ids (default seg_relabel="tracklet")
    unique_vals = set(seg_arr.flatten()) - {0}
    label_key = tracks.features.tracklet_key
    track_ids = set(
        tracks.graph_solution.node_attrs(attr_keys=[label_key])[label_key].to_list()
    )
    assert unique_vals == track_ids

    # Check metadata references the tiff path with ../../ prefix (sibling of geff dir)
    z = zarr.open((export_dir / "tracks.geff").as_posix(), mode="r")
    related = dict(z.attrs)["geff"].get("related_objects", [])
    assert any(obj["path"] == "../../segmentation.tif" for obj in related)


# --- write_to_geff ---


@pytest.mark.parametrize("ndim", [3, 4])
@pytest.mark.parametrize("with_seg", [False, True])
def test_write_to_geff_roundtrip(get_tracks, ndim, with_seg, tmp_path):
    """write_to_geff then import_from_geff recovers the tracks."""
    tracks = get_tracks(ndim=ndim, with_seg=with_seg, is_solution=True)

    geff_path = tmp_path / "my_tracks.geff"
    write_to_geff(tracks, geff_path)

    loaded = import_from_geff(geff_path)

    assert loaded.graph_solution.num_nodes() == tracks.graph_solution.num_nodes()
    assert loaded.graph_solution.num_edges() == tracks.graph_solution.num_edges()
    assert set(loaded.graph_solution.node_ids()) == set(tracks.graph_solution.node_ids())
    assert loaded.features.dump_json() == tracks.features.dump_json()

    if with_seg:
        assert loaded.segmentation is not None
        assert loaded.segmentation.shape == tracks.segmentation.shape
        np.testing.assert_array_equal(
            np.asarray(loaded.segmentation[:]), np.asarray(tracks.segmentation[:])
        )


def test_write_to_geff_no_parent_container(get_tracks, tmp_path):
    """write_to_geff writes directly to the path — no parent .zgroup or
    tracks.geff subfolder."""
    tracks = get_tracks(ndim=3, with_seg=False, is_solution=True)

    geff_path = tmp_path / "my_tracks.geff"
    write_to_geff(tracks, geff_path)

    # The store is at the given path, not nested inside it
    z = zarr.open(str(geff_path), mode="r")
    assert "geff" in dict(z.attrs)
    assert "nodes" in z

    # No .zgroup/.zattrs at the parent level (tmp_path is not a zarr group)
    assert not (tmp_path / ".zgroup").exists()
    assert not (tmp_path / ".zattrs").exists()


def test_write_to_geff_overwrite(get_tracks, tmp_path):
    """write_to_geff with overwrite=True replaces existing store."""
    tracks = get_tracks(ndim=3, with_seg=False, is_solution=True)

    geff_path = tmp_path / "my_tracks.geff"
    write_to_geff(tracks, geff_path)
    write_to_geff(tracks, geff_path, overwrite=True)

    loaded = import_from_geff(geff_path)
    assert loaded.graph_solution.num_nodes() == tracks.graph_solution.num_nodes()


def test_write_to_geff_no_overwrite_raises(get_tracks, tmp_path):
    """write_to_geff with overwrite=False raises on existing store."""
    tracks = get_tracks(ndim=3, with_seg=False, is_solution=True)

    geff_path = tmp_path / "my_tracks.geff"
    write_to_geff(tracks, geff_path)

    with pytest.raises(FileExistsError):
        write_to_geff(tracks, geff_path, overwrite=False)


def test_write_to_geff_metadata(get_tracks, tmp_path):
    """write_to_geff stores axes and FeatureDict metadata."""
    tracks = get_tracks(ndim=3, with_seg=False, is_solution=True)

    geff_path = tmp_path / "my_tracks.geff"
    write_to_geff(tracks, geff_path)

    z = zarr.open(str(geff_path), mode="r")
    attrs = dict(z.attrs)

    axes = attrs["geff"]["axes"]
    assert len(axes) == 3
    assert [ax["type"] for ax in axes] == ["time", "space", "space"]

    assert "funtracks" in attrs["geff"].get("extra", {})
    assert "features" in attrs["geff"]["extra"]["funtracks"]


def test_write_to_geff_segmentation_shape(get_tracks, tmp_path):
    """write_to_geff writes segmentation_shape when masks are present."""
    tracks = get_tracks(ndim=3, with_seg=True, is_solution=True)

    geff_path = tmp_path / "my_tracks.geff"
    write_to_geff(tracks, geff_path)

    z = zarr.open(str(geff_path), mode="r")
    attrs = dict(z.attrs)
    assert "segmentation_shape" in attrs
    assert tuple(attrs["segmentation_shape"]) == tracks.segmentation.shape
