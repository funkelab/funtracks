import dask.array as da
import numpy as np
import pytest
import tifffile
from geff.testing.data import create_mock_geff

from funtracks.import_export import import_from_geff
from funtracks.import_export.geff._import import import_graph_from_geff


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


def test_import_graph_from_geff_renames_keys_to_standard(valid_geff):
    """Test that import_graph_from_geff renames custom GEFF keys to standard keys.

    This is a key architectural requirement: import_graph_from_geff should return
    an InMemoryGeff where all node_props keys have been renamed from custom GEFF
    property names to standard funtracks keys, using the provided node_name_map.
    """
    store, original_geff = valid_geff

    # Define node_name_map: standard_key -> custom_geff_key
    node_name_map = {
        "time": "t",  # standard key "time" maps to GEFF key "t"
        "y": "y",  # standard key "y" maps to GEFF key "y"
        "x": "x",  # standard key "x" maps to GEFF key "x"
        "track_id": "track_id",  # standard key "track_id" maps to GEFF key "track_id"
        "seg_id": "seg_id",  # standard key "seg_id" maps to GEFF key "seg_id"
        "circularity": "circ",  # standard key "circularity" maps to GEFF key "circ"
    }

    # Call import_graph_from_geff
    in_memory_geff, position_attr, ndims = import_graph_from_geff(store, node_name_map)

    # Assert the InMemoryGeff has standard keys, NOT custom GEFF keys
    node_props = in_memory_geff["node_props"]

    # Standard keys should be present
    assert "time" in node_props, "Standard key 'time' should be present"
    assert "y" in node_props, "Standard key 'y' should be present"
    assert "x" in node_props, "Standard key 'x' should be present"
    assert "track_id" in node_props, "Standard key 'track_id' should be present"
    assert "seg_id" in node_props, "Standard key 'seg_id' should be present"
    assert "circularity" in node_props, "Standard key 'circularity' should be present"

    # Custom GEFF keys should NOT be present
    assert "t" not in node_props, "Custom GEFF key 't' should have been renamed to 'time'"
    assert "circ" not in node_props, (
        "Custom GEFF key 'circ' should have been renamed to 'circularity'"
    )

    # Verify data integrity - values should be preserved
    assert len(node_props["time"]["values"]) == 5, "Should have 5 time values"
    assert len(node_props["track_id"]["values"]) == 5, "Should have 5 track_id values"
    np.testing.assert_array_equal(
        node_props["track_id"]["values"][:],
        np.arange(5),
        err_msg="track_id values should be preserved after renaming",
    )
    np.testing.assert_array_almost_equal(
        node_props["circularity"]["values"][:],
        np.array([0.2, 0.1, 0.5, 0.3, 0.45]),
        err_msg="circularity values should be preserved after renaming from 'circ'",
    )

    # Verify return values
    assert position_attr == ["y", "x"], "Should return standard position keys"
    assert ndims == 3, "Should be 3D (time + 2 spatial dims)"


def test_import_graph_from_geff_loads_custom_features(valid_geff):
    """Test that custom features can be loaded by including them in node_name_map.

    Custom features (not in the standard set) should be loaded when included
    in the node_name_map. The key should remain as the standard key (which equals
    the GEFF key in this case).
    """
    store, original_geff = valid_geff

    # Include custom features in node_name_map
    node_name_map = {
        "time": "t",
        "y": "y",
        "x": "x",
        "random_feature": "random_feature",  # Custom feature: maps to itself
        "random_feature2": "random_feature2",  # Another custom feature
    }

    # Call import_graph_from_geff
    in_memory_geff, position_attr, ndims = import_graph_from_geff(store, node_name_map)

    node_props = in_memory_geff["node_props"]

    # Custom features should be present with standard keys
    assert "random_feature" in node_props, (
        "Custom feature 'random_feature' should be loaded"
    )
    assert "random_feature2" in node_props, (
        "Custom feature 'random_feature2' should be loaded"
    )

    # Verify data integrity
    np.testing.assert_array_equal(
        node_props["random_feature"]["values"][:],
        np.array(["a", "b", "c", "d", "e"]),
        err_msg="random_feature values should be preserved",
    )
    np.testing.assert_array_equal(
        node_props["random_feature2"]["values"][:],
        np.array(["a", "b", "c", "d", "e"]),
        err_msg="random_feature2 values should be preserved",
    )


def test_import_graph_from_geff_custom_feature_with_different_name(valid_geff):
    """Test that custom features can be renamed using node_name_map.

    A custom feature with a GEFF name can be renamed to a different standard key
    using the node_name_map.
    """
    store, original_geff = valid_geff

    # Rename "circ" to "my_custom_circularity"
    node_name_map = {
        "time": "t",
        "y": "y",
        "x": "x",
        "my_custom_circularity": "circ",  # Rename circ to custom name
    }

    # Call import_graph_from_geff
    in_memory_geff, position_attr, ndims = import_graph_from_geff(store, node_name_map)

    node_props = in_memory_geff["node_props"]

    # The custom key should be present
    assert "my_custom_circularity" in node_props, (
        "Custom renamed feature should be present"
    )

    # The original GEFF key should NOT be present
    assert "circ" not in node_props, "Original GEFF key should be renamed"

    # Verify data integrity
    np.testing.assert_array_almost_equal(
        node_props["my_custom_circularity"]["values"][:],
        np.array([0.2, 0.1, 0.5, 0.3, 0.45]),
        err_msg="Values should be preserved after custom renaming",
    )


def test_import_graph_from_geff_edge_name_map_none(valid_geff):
    """Test that edge_name_map=None loads all edge properties.

    When edge_name_map is None, all edge properties should be loaded with their
    original GEFF names (no renaming).
    """
    store, original_geff = valid_geff

    # Define node name map
    node_name_map = {
        "time": "t",
        "y": "y",
        "x": "x",
    }

    # Call import_graph_from_geff without edge_name_map (defaults to None)
    in_memory_geff, position_attr, ndims = import_graph_from_geff(
        store, node_name_map, edge_name_map=None
    )

    # Should load successfully
    assert "node_props" in in_memory_geff
    assert "edge_props" in in_memory_geff
    # The fixture has no edge properties, so edge_props should be empty
    assert in_memory_geff["edge_props"] == {}


def test_none_in_name_map(valid_geff):
    """Test that None values in required t/y/x attributes are caught"""

    store, _ = valid_geff
    # None value for required field should raise error
    name_map = {"time": None, "pos": ["y", "x"]}
    with pytest.raises(ValueError, match="None values"):
        import_from_geff(store, name_map)


def test_duplicate_values_in_name_map(valid_geff):
    """Test that duplicate values in name_map are allowed."""
    store, _ = valid_geff

    # Duplicate values should be allowed - each standard key gets a copy of the data
    name_map = {"time": "t", "pos": ["y", "x"], "seg_id": "t"}

    # Should not raise - seg_id maps to same source as time
    tracks = import_from_geff(store, name_map)

    # Both time and seg_id should be present with same values
    for node_id in tracks.graph.nodes():
        assert tracks.get_node_attr(node_id, "seg_id") == tracks.get_node_attr(
            node_id, "time"
        )


def test_segmentation_axes_mismatch(valid_geff, tmp_path):
    """Test checking if number of dimensions match and if coordinates are within
    bounds."""

    store, _ = valid_geff
    name_map = {"time": "t", "pos": ["y", "x"], "seg_id": "seg_id"}

    # Provide a segmentation with wrong shape
    wrong_seg = np.zeros((2, 20, 200), dtype=np.uint16)
    seg_path = tmp_path / "wrong_seg.npy"
    tifffile.imwrite(seg_path, wrong_seg)
    with pytest.raises(ValueError, match="out of bounds"):
        import_from_geff(store, name_map, segmentation_path=seg_path)

    # Provide a segmentation with a different number of dimensions than the graph.
    # The error is caught during validation because pos has spatial_dims=True
    wrong_seg = np.zeros((2, 20, 200, 200), dtype=np.uint16)
    seg_path = tmp_path / "wrong_seg2.npy"
    tifffile.imwrite(seg_path, wrong_seg)
    with pytest.raises(ValueError, match="pos.*has 2 values.*3 spatial dimensions"):
        import_from_geff(store, name_map, segmentation_path=seg_path)


def test_tracks_with_segmentation(valid_geff, invalid_geff, valid_segmentation, tmp_path):
    """Test relabeling of the segmentation from seg_id to node_id."""

    store, _ = valid_geff
    name_map = {"time": "t", "pos": ["y", "x"], "seg_id": "seg_id"}
    valid_segmentation_path = tmp_path / "segmentation.tif"
    tifffile.imwrite(valid_segmentation_path, valid_segmentation)

    # Test that a tracks object is produced and that the seg_id has been relabeled.
    scale = [1, 1, (1 / 100)]
    node_features = {
        "area": True,  # In geff, but should be recomputed
        "random_feature": False,  # Static feature - load from geff
    }

    tracks = import_from_geff(
        store,
        name_map,
        segmentation_path=valid_segmentation_path,
        scale=scale,
        node_features=node_features,
    )
    assert hasattr(tracks, "segmentation")
    assert tracks.segmentation.shape == valid_segmentation.shape
    # Get last node by ID (don't rely on iteration order)
    last_node = max(tracks.graph.nodes())
    # With composite pos, position is stored as an array
    pos = tracks.graph.nodes[last_node]["pos"]
    coords = [
        tracks.graph.nodes[last_node]["time"],
        pos[0],  # y
        pos[1],  # x
    ]
    coords = tuple(int(c * 1 / s) for c, s in zip(coords, scale, strict=True))
    assert (
        valid_segmentation[tuple(coords)] == 50
    )  # in original segmentation, the pixel value is equal to seg_id
    assert (
        tracks.segmentation[tuple(coords)] == last_node
    )  # test that the seg id has been relabeled

    # Check that only required/requested features are present, and that area is recomputed
    data = tracks.graph.nodes[last_node]
    assert "random_feature" in data
    assert "random_feature2" not in data
    assert "area" in data
    assert (
        data["area"] == 0.01
    )  # recomputed area values should be 1 pixel, so 0.01 after applying the scaling.

    # Check that area is not recomputed but taken directly from the graph
    node_features = {
        "area": False,  # Load from geff, don't recompute
        "random_feature": False,  # Static feature - load from geff
    }

    tracks = import_from_geff(
        store,
        name_map,
        segmentation_path=valid_segmentation_path,
        scale=scale,
        node_features=node_features,
    )
    # Get last node by ID (don't rely on iteration order)
    last_node = max(tracks.graph.nodes())
    data = tracks.graph.nodes[last_node]
    assert "area" in data
    assert data["area"] == 21

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
    name_map = {"time": "t", "pos": ["y", "x"], "seg_id": "seg_id"}
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

    node_features = {
        "area": False,  # Load from geff, don't recompute
        "random_feature": False,  # Static feature - load from geff
    }

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
    name_map = {
        "time": "t",
        "pos": ["y", "x"],  # Composite position mapping
        "seg_id": "seg_id",
        "circularity": "circ",  # Map standard key to GEFF property name
    }
    scale = [1, 1, 1 / 100]
    valid_segmentation_path = tmp_path / "segmentation.tif"
    tifffile.imwrite(valid_segmentation_path, valid_segmentation)

    # Test 1: Mix of computed (True) and loaded (False) features
    node_features = {
        "area": True,  # In geff, but should be recomputed
        "random_feature": False,  # Static feature - load from geff
        "circularity": False,  # In geff, load without recomputing
        "ellipse_axis_radii": True,  # Not in geff, should be computed
    }

    tracks = import_from_geff(
        store,
        name_map,
        segmentation_path=valid_segmentation_path,
        scale=scale,
        node_features=node_features,
    )

    feature_keys = ["area", "random_feature", "ellipse_axis_radii", "circularity"]
    for key in feature_keys:
        assert key in tracks.features

    # Get last node by ID (don't rely on iteration order)
    max_node_id = max(tracks.graph.nodes())
    data = tracks.graph.nodes[max_node_id]

    # All requested features should be present
    for key in feature_keys:
        assert key in data

    # Verify computed values (1 pixel = 0.01 after scaling)
    # Original geff had area=21 for last node
    assert data["area"] == 0.01
    assert data["ellipse_axis_radii"] is not None
    assert data["circularity"] == 0.45  # the value should not be recomputed

    # Verify loaded value from geff
    assert data["random_feature"] == "e"


def test_node_features_unknown(valid_geff, valid_segmentation, tmp_path):
    """Test that providing an unknown feature raises a KeyError."""
    store, _ = valid_geff
    name_map = {"time": "t", "pos": ["y", "x"], "seg_id": "seg_id"}
    scale = [1, 1, 1 / 100]
    valid_segmentation_path = tmp_path / "segmentation.tif"
    tifffile.imwrite(valid_segmentation_path, valid_segmentation)

    # Test unknown feature that doesn't exist in annotators or GEFF
    node_features = {
        "unknown_computed_feature": True,  # Unknown feature, request computation
    }

    with pytest.raises(
        KeyError,
        match="Node features not available",
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
    name_map = {"time": "t", "pos": ["y", "x"]}
    scale = [1, 1, 1 / 100]

    # Try to compute area feature without providing segmentation
    node_features = {
        "area": True,  # Request computation without segmentation
    }

    with pytest.raises(
        KeyError,
        match="Node features not available",
    ):
        import_from_geff(
            store,
            name_map,
            segmentation_path=None,  # No segmentation
            scale=scale,
            node_features=node_features,
        )
