import dask.array as da
import numpy as np
import pytest
import tifffile
from geff.testing.data import create_memory_mock_geff

from funtracks.import_export.import_from_geff import import_from_geff


@pytest.fixture
def valid_store_and_attrs():
    store, graphattrs = create_memory_mock_geff(
        node_id_dtype="int",
        node_axis_dtypes={"position": "float64", "time": "int64"},
        directed=True,
        num_nodes=5,
        num_edges=2,
        include_t=True,
        include_z=False,
        include_y=True,
        include_x=True,
        extra_node_props={
            "track_id": np.arange(5),
            "seg_id": np.array([10, 20, 30, 40, 50]),
            "lineage_id": np.arange(5),
            "area": np.array([20, 41, 42, 776, 21]),
            "random_feature": np.array(["a", "b", "c", "d", "e"]),
            "random_feature2": np.array(["a", "b", "c", "d", "e"]),
        },
    )
    return store, graphattrs


@pytest.fixture
def invalid_store_and_attrs():
    store, graphattrs = create_memory_mock_geff(
        node_id_dtype="int",
        node_axis_dtypes={"position": "float64", "time": "int64"},
        directed=True,
        num_nodes=5,
        num_edges=2,
        include_t=True,
        include_z=False,
        include_y=True,
        include_x=True,
        extra_node_props={
            "track_id": np.arange(5),
            "seg_id": np.array([10.453, 20.23, 30.56, 40.78, 50.92]),
            "lineage_id": np.arange(5),
            "area": np.array([20, 41, 42, 776, 21]),
        },
    )
    return store, graphattrs


@pytest.fixture
def valid_segmentation():
    shape = (6, 600, 200)
    seg = np.zeros(shape, dtype=int)

    times = [1, 2, 3, 4, 5]
    x = [1.0, 0.775, 0.55, 0.325, 0.1]
    y = [100, 200, 300, 400, 500]
    scale = [1, 1, 100]
    seg_ids = np.array([10, 20, 30, 40, 50])

    for t, y_val, x_f, seg_id in zip(times, y, x, seg_ids, strict=False):
        x = int(x_f * scale[2])
        seg[t, y_val, x] = seg_id
    return seg


def test_duplicate_or_none_in_name_map(valid_store_and_attrs):
    """Test that duplicate values or missing t/y/x attributes are caught"""

    store, _ = valid_store_and_attrs
    # Duplicate value
    name_map = {"time": "time", "y": "y", "x": "y"}
    with pytest.raises(ValueError, match="duplicate values"):
        import_from_geff(store, name_map)
    # None value
    name_map = {"time": None, "y": "y", "x": "x"}
    with pytest.raises(ValueError, match="None values"):
        import_from_geff(store, name_map)


def test_segmentation_axes_mismatch(valid_store_and_attrs, tmp_path):
    """Test checking if number of dimensions match and if coordinates are within
    bounds."""

    store, _ = valid_store_and_attrs
    name_map = {"time": "t", "y": "y", "x": "x", "seg_id": "seg_id"}

    # Provide a segmentation with wrong shape
    wrong_seg = np.zeros((2, 20, 200), dtype=np.uint16)
    seg_path = tmp_path / "wrong_seg.npy"
    tifffile.imwrite(seg_path, wrong_seg)
    with pytest.raises(ValueError, match="out of bounds"):
        import_from_geff(store, name_map, segmentation_path=seg_path)

    # Provide a segmentation with a different number of dimensions than the graph.
    wrong_seg = np.zeros((2, 20, 200, 200), dtype=np.uint16)
    seg_path = tmp_path / "wrong_seg2.npy"
    tifffile.imwrite(seg_path, wrong_seg)
    with pytest.raises(ValueError, match="Axes in the geff do not match"):
        import_from_geff(store, name_map, segmentation_path=seg_path)


def test_tracks_with_segmentation(
    valid_store_and_attrs, invalid_store_and_attrs, valid_segmentation, tmp_path
):
    """Test relabeling of the segmentation from seg_id to node_id."""

    store, _ = valid_store_and_attrs
    name_map = {"time": "t", "y": "y", "x": "x", "seg_id": "seg_id"}
    valid_segmentation_path = tmp_path / "segmentation.tif"
    tifffile.imwrite(valid_segmentation_path, valid_segmentation)

    # Test that a tracks object is produced and that the seg_id has been relabeled.
    scale = [1, 1, (1 / 100)]
    extra_features = {"area": True, "random_feature": False}
    tracks = import_from_geff(
        store,
        name_map,
        segmentation_path=valid_segmentation_path,
        scale=scale,
        extra_features=extra_features,
    )
    assert hasattr(tracks, "segmentation")
    assert tracks.segmentation.shape == valid_segmentation.shape
    last_node = list(tracks.graph.nodes)[-1]
    coords = [
        tracks.graph.nodes[last_node]["t"],
        tracks.graph.nodes[last_node]["y"],
        tracks.graph.nodes[last_node]["x"],
    ]
    coords = tuple(int(c * 1 / s) for c, s in zip(coords, scale, strict=True))
    assert (
        valid_segmentation[tuple(coords)] == 50
    )  # in original segmentation, the pixel value is equal to seg_id
    assert (
        tracks.segmentation[tuple(coords)] == last_node
    )  # test that the seg id has been relabeled

    # Check that only required/requested features are present, and that area is recomputed
    _, data = list(tracks.graph.nodes(data=True))[-1]
    assert "random_feature" in data
    assert "random_feature2" not in data
    assert "area" in data
    assert (
        data["area"] == 0.01
    )  # recomputed area values should be 1 pixel, so 0.01 after applying the scaling.

    # Check that area is not recomputed but taken directly from the graph
    extra_features = {"area": False, "random_feature": False}  # set Recompute to False
    tracks = import_from_geff(
        store,
        name_map,
        segmentation_path=valid_segmentation_path,
        scale=scale,
        extra_features=extra_features,
    )
    _, data = list(tracks.graph.nodes(data=True))[-1]
    assert "area" in data
    assert data["area"] == 21

    # Test that import fails with ValueError when scaling information is missing or
    # incorrect
    with pytest.raises(ValueError):
        tracks = import_from_geff(
            store, name_map, segmentation_path=valid_segmentation_path, scale=None
        )

    # Test that import fails with ValueError when invalid seg_ids are provided.
    store, _ = invalid_store_and_attrs
    with pytest.raises(ValueError):
        tracks = import_from_geff(
            store, name_map, segmentation_path=valid_segmentation_path, scale=scale
        )


@pytest.mark.parametrize("segmentation_format", ["single_tif", "tif_folder", "zarr"])
def test_segmentation_loading_formats(
    segmentation_format, valid_store_and_attrs, valid_segmentation, tmp_path
):
    """Test loading segmentation from different formats using magic_imread."""
    store, _ = valid_store_and_attrs
    name_map = {"time": "t", "y": "y", "x": "x", "seg_id": "seg_id"}
    scale = [1, 1, 1 / 100]
    seg = valid_segmentation

    if segmentation_format == "single_tif":
        path = tmp_path / "segmentation.tif"
        tifffile.imwrite(path, seg)

    elif segmentation_format == "tif_folder":
        path = tmp_path / "tif_series"
        path.mkdir()
        for i, frame in enumerate(seg):
            tifffile.imwrite(path / f"seg_{i:03}.tif", frame)

    elif segmentation_format == "zarr":
        path = tmp_path / "segmentation.zarr"
        da.from_array(seg, chunks=(1, *seg.shape[1:])).to_zarr(path)

    else:
        raise ValueError(f"Unknown format: {segmentation_format}")

    tracks = import_from_geff(
        store,
        name_map,
        segmentation_path=path,
        scale=scale,
        extra_features={"area": False, "random_feature": False},
    )

    assert hasattr(tracks, "segmentation")
    assert np.array(tracks.segmentation).shape == seg.shape
