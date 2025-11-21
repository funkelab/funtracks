import dask.array as da
import numpy as np
import pytest
import tifffile
from geff.testing.data import create_mock_geff

from funtracks.import_export.import_from_geff import import_from_geff


@pytest.fixture
def valid_geff():
    store, memory_geff = create_mock_geff(
        node_id_dtype="uint",
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
            "circ": np.array([0.2, 0.1, 0.5, 0.3, 0.45]),
            "random_feature": np.array(["a", "b", "c", "d", "e"]),
            "random_feature2": np.array(["a", "b", "c", "d", "e"]),
        },
    )
    return store, memory_geff


@pytest.fixture
def invalid_geff():
    invalid_store, invalid_memory_geff = create_mock_geff(
        node_id_dtype="uint",
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
    return invalid_store, invalid_memory_geff


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


def test_duplicate_or_none_in_name_map(valid_geff):
    """Test that duplicate values or missing t/y/x attributes are caught"""

    store, _ = valid_geff
    # Duplicate value
    name_map = {"time": "t", "y": "y", "x": "y"}
    with pytest.raises(ValueError, match="duplicate values"):
        import_from_geff(store, name_map)
    # None value
    name_map = {"time": None, "y": "y", "x": "x"}
    with pytest.raises(ValueError, match="None values"):
        import_from_geff(store, name_map)


def test_segmentation_axes_mismatch(valid_geff, tmp_path):
    """Test checking if number of dimensions match and if coordinates are within
    bounds."""

    store, _ = valid_geff
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


def test_tracks_with_segmentation(valid_geff, invalid_geff, valid_segmentation, tmp_path):
    """Test relabeling of the segmentation from seg_id to node_id."""

    store, _ = valid_geff
    name_map = {"time": "t", "y": "y", "x": "x", "seg_id": "seg_id"}
    valid_segmentation_path = tmp_path / "segmentation.tif"
    tifffile.imwrite(valid_segmentation_path, valid_segmentation)

    # Test that a tracks object is produced and that the seg_id has been relabeled.
    scale = [1, 1, (1 / 100)]
    node_features = [
        {
            "prop_name": "area",
            "feature": "Area",
            "recompute": True,  # In geff, but should be recomputed
            "dtype": "float",
        },
        {
            "prop_name": "random_feature",
            "feature": None,
            "recompute": False,
            "dtype": "float",
        },
    ]

    tracks = import_from_geff(
        store,
        name_map,
        segmentation_path=valid_segmentation_path,
        scale=scale,
        node_features=node_features,
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
    node_features = [
        {"prop_name": "area", "feature": "Area", "recompute": False, "dtype": "float"},
        {
            "prop_name": "random_feature",
            "feature": None,
            "recompute": False,
            "dtype": "float",
        },
    ]

    tracks = import_from_geff(
        store,
        name_map,
        segmentation_path=valid_segmentation_path,
        scale=scale,
        node_features=node_features,
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
    store, _ = invalid_geff
    with pytest.raises(ValueError):
        tracks = import_from_geff(
            store, name_map, segmentation_path=valid_segmentation_path, scale=scale
        )


@pytest.mark.parametrize("segmentation_format", ["single_tif", "tif_folder", "zarr"])
def test_segmentation_loading_formats(
    segmentation_format, valid_geff, valid_segmentation, tmp_path
):
    """Test loading segmentation from different formats using magic_imread."""
    store, _ = valid_geff
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

    node_features = [
        {"prop_name": "area", "feature": "Area", "recompute": False, "dtype": "float"},
        {
            "prop_name": "random_feature",
            "feature": None,
            "recompute": False,
            "dtype": "float",
        },
    ]

    tracks = import_from_geff(
        store,
        name_map,
        segmentation_path=path,
        scale=scale,
        node_features=node_features,
    )

    assert hasattr(tracks, "segmentation")
    assert np.array(tracks.segmentation).shape == seg.shape


def test_node_features_compute_vs_load(valid_geff, valid_segmentation, tmp_path):
    """Test that node_features controls whether features are computed or loaded.

    Features marked True in node_features are computed using annotators.
    Features marked False are loaded directly from the geff file.
    Features not in the geff can still be computed if marked True.
    """
    store, _ = valid_geff
    name_map = {"time": "t", "y": "y", "x": "x", "seg_id": "seg_id"}
    scale = [1, 1, 1 / 100]
    valid_segmentation_path = tmp_path / "segmentation.tif"
    tifffile.imwrite(valid_segmentation_path, valid_segmentation)

    # Test 1: Mix of computed (True) and loaded (False) features
    node_features = [
        {
            "prop_name": "area",
            "feature": "Area",
            "recompute": True,  # In geff, but should be recomputed
            "dtype": "float",
        },
        {
            "prop_name": "random_feature",
            "feature": None,
            "recompute": False,
            "dtype": "float",
        },
        {
            "prop_name": "circ",
            "feature": "Circularity",
            "recompute": False,  # In geff, load without recomputing
            "dtype": "float",
        },
        {
            "prop_name": "ellipse",
            "feature": "Ellipse axis radii",
            "recompute": True,  # Not in geff, should be computed
            "dtype": "float",
        },
    ]

    tracks = import_from_geff(
        store,
        name_map,
        segmentation_path=valid_segmentation_path,
        scale=scale,
        node_features=node_features,
    )

    feature_keys = ["area", "random_feature", "ellipse", "circ"]
    for key in feature_keys:
        assert key in tracks.features

    _, data = list(tracks.graph.nodes(data=True))[-1]

    # All requested features should be present
    for key in feature_keys:
        assert key in data

    # Verify computed values (1 pixel = 0.01 after scaling)
    # Original geff had area=21 for last node
    assert data["area"] == 0.01
    assert data["ellipse"] is not None
    assert data["circ"] == 0.45  # the value should not be recomputed

    # Verify loaded value from geff
    assert data["random_feature"] == "e"


def test_node_features_unknown(valid_geff, valid_segmentation, tmp_path):
    """Test that providing an unknown feature raises a ValueError."""
    store, _ = valid_geff
    name_map = {"time": "t", "y": "y", "x": "x", "seg_id": "seg_id"}
    scale = [1, 1, 1 / 100]
    valid_segmentation_path = tmp_path / "segmentation.tif"
    tifffile.imwrite(valid_segmentation_path, valid_segmentation)

    # Test unknown regionprops feature
    node_features = [
        {
            "prop_name": "area",
            "feature": "AREA",  # Unknown feature
            "recompute": True,
            "dtype": "float",
        }
    ]

    with pytest.raises(
        ValueError,
        match="Requested activation of feature .* but no such feature found "
        "in computed features. Perhaps you need to provide a segmentation?",
    ):
        import_from_geff(
            store,
            name_map,
            segmentation_path=valid_segmentation_path,
            scale=scale,
            node_features=node_features,
        )


def test_compute_features_without_segmentation(valid_geff):
    """Test that computing regionprops features without segmentation raises an error."""
    store, _ = valid_geff
    name_map = {"time": "t", "y": "y", "x": "x"}
    scale = [1, 1, 1 / 100]

    # Try to compute area feature without providing segmentation
    node_features = [
        {
            "prop_name": "area",
            "feature": "Area",
            "recompute": True,  # Request computation
            "dtype": "float",
        }
    ]

    with pytest.raises(
        ValueError,
        match=(
            "Requested activation of feature .* but no such feature found "
            "in computed features. Perhaps you need to provide a segmentation?"
        ),
    ):
        import_from_geff(
            store,
            name_map,
            segmentation_path=None,  # No segmentation
            scale=scale,
            node_features=node_features,
        )


def test_load_features_without_segmentation_not_computed(valid_geff):
    """Test that loading regionprops features without segmentation raises error."""
    store, _ = valid_geff
    name_map = {"time": "t", "y": "y", "x": "x"}
    scale = [1, 1, 1 / 100]

    # Try to load circularity feature without providing segmentation
    node_features = [
        {
            "prop_name": "circ",
            "feature": "Circularity",
            "recompute": False,  # Just load, don't compute
            "dtype": "float",
        }
    ]

    with pytest.raises(
        ValueError,
        match=(
            "Requested activation of feature .* but no such feature found "
            "in computed features. Perhaps you need to provide a segmentation?"
        ),
    ):
        import_from_geff(
            store,
            name_map,
            segmentation_path=None,  # No segmentation
            scale=scale,
            node_features=node_features,
        )


def test_deprecated_dict_format(valid_geff, valid_segmentation, tmp_path):
    """Test backward compatibility with dict[str, bool] node_features format."""
    store, _ = valid_geff
    name_map = {"time": "t", "y": "y", "x": "x", "seg_id": "seg_id"}
    scale = [1, 1, 1 / 100]
    valid_segmentation_path = tmp_path / "segmentation.tif"
    tifffile.imwrite(valid_segmentation_path, valid_segmentation)

    # Use deprecated dict[str, bool] format
    node_features_dict = {
        "area": True,  # Computed feature - should recompute
        "circ": False,  # Computed feature - should load from geff
        "random_feature": False,  # Static feature - should load from geff
    }

    # Should issue DeprecationWarning but still work
    with pytest.warns(
        DeprecationWarning,
        match="Passing node_features as dict\\[str, bool\\] is deprecated",
    ):
        tracks = import_from_geff(
            store,
            name_map,
            segmentation_path=valid_segmentation_path,
            scale=scale,
            node_features=node_features_dict,
        )

    # All requested features should be present
    assert "area" in tracks.features
    assert "circ" in tracks.features
    assert "random_feature" in tracks.features

    # Check values in graph
    _, data = list(tracks.graph.nodes(data=True))[-1]

    # Area should be recomputed (1 pixel = 0.01 after scaling)
    assert data["area"] == 0.01

    # Circularity should be loaded from geff (not recomputed)
    assert data["circ"] == 0.45  # Original value from geff

    # Random feature should be loaded from geff
    assert data["random_feature"] == "e"


def test_deprecated_dict_format_unknown_feature(valid_geff):
    """Test that dict format with unknown feature raises clear error."""
    store, _ = valid_geff
    name_map = {"time": "t", "y": "y", "x": "x"}

    # Use deprecated dict format with unknown feature
    node_features_dict = {"unknown_feature": True}

    with (
        pytest.warns(
            DeprecationWarning,
            match="Passing node_features as dict\\[str, bool\\] is deprecated",
        ),
        pytest.raises(
            ValueError,
            match="Unknown feature 'unknown_feature' - not found in geff data",
        ),
    ):
        import_from_geff(store, name_map, node_features=node_features_dict)
