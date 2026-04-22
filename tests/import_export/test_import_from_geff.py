import dask.array as da
import numpy as np
import pytest
import tifffile
from geff.testing.data import create_mock_geff

from funtracks.data_model import SolutionTracks
from funtracks.import_export import export_to_geff, import_from_geff
from funtracks.import_export.geff._import import GeffTracksBuilder, import_graph_from_geff
from funtracks.utils.tracksdata_utils import create_empty_graphview_graph


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
    node_name_map = {"time": "t", "pos": ["y", "x"], "seg_id": "t"}

    # Should not raise - seg_id maps to same source as time
    tracks = import_from_geff(store, node_name_map)

    # Both time and seg_id should be present with same values
    for node_id in tracks.graph.node_ids():
        assert tracks.get_node_attr(node_id, "seg_id") == tracks.get_node_attr(
            node_id, "t"
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
    name_map_with_features = {
        **name_map,
        "area": "area",
        "random_feature": "random_feature",
    }

    tracks = import_from_geff(
        store,
        name_map_with_features,
        segmentation_path=valid_segmentation_path,
        scale=scale,
    )
    assert hasattr(tracks, "segmentation")
    assert tracks.segmentation.shape == valid_segmentation.shape
    # Get last node by ID (don't rely on iteration order)
    last_node = max(tracks.graph.node_ids())
    # With composite pos, position is stored as an array
    pos = tracks.graph.nodes[last_node]["pos"]
    coords = [
        tracks.graph.nodes[last_node]["t"],
        pos[0],  # y
        pos[1],  # x
    ]
    coords = tuple(int(c * 1 / s) for c, s in zip(coords, scale, strict=True))
    assert (
        valid_segmentation[tuple(coords)] == 50
    )  # in original segmentation, the pixel value is equal to seg_id
    assert (
        np.asarray(tracks.segmentation)[tuple(coords)] == last_node
    )  # test that the seg id has been relabeled

    # Check that only requested features are present and area is loaded from geff
    data = tracks.graph.nodes[last_node]
    assert "random_feature" in tracks.graph.node_attr_keys()
    assert "random_feature2" not in tracks.graph.node_attr_keys()
    assert "area" in tracks.graph.node_attr_keys()
    assert data["area"] == 21  # loaded directly from geff, not recomputed

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

    name_map_with_features = {
        **name_map,
        "area": "area",
        "random_feature": "random_feature",
    }

    tracks = import_from_geff(
        store,
        name_map_with_features,
        segmentation_path=path,
        scale=scale,
    )

    assert hasattr(tracks, "segmentation")
    assert np.array(tracks.segmentation).shape == seg.shape


def test_features_loaded_from_name_map(valid_geff, valid_segmentation, tmp_path):
    """Test that features included in node_name_map are loaded and registered."""
    store, _ = valid_geff
    node_name_map = {
        "time": "t",
        "pos": ["y", "x"],
        "seg_id": "seg_id",
        "circularity": "circ",  # Map standard key to GEFF property name
        "area": "area",  # Load area from geff
        "random_feature": "random_feature",  # Static feature - load from geff
    }
    scale = [1, 1, 1 / 100]
    valid_segmentation_path = tmp_path / "segmentation.tif"
    tifffile.imwrite(valid_segmentation_path, valid_segmentation)

    tracks = import_from_geff(
        store,
        node_name_map,
        segmentation_path=valid_segmentation_path,
        scale=scale,
    )

    feature_keys = ["area", "random_feature", "circularity"]
    for key in feature_keys:
        assert key in tracks.features

    # Get last node by ID (don't rely on iteration order)
    max_node_id = max(tracks.graph.node_ids())
    data = tracks.graph.nodes[max_node_id]

    # All requested features should be present and loaded from geff
    for key in feature_keys:
        assert data[key] is not None

    assert data["area"] == 21  # loaded from geff
    assert data["circularity"] == 0.45  # loaded from geff (renamed from "circ")
    assert data["random_feature"] == "e"  # static feature loaded from geff


def test_nonexistent_property_in_name_map(valid_geff):
    """Test that mapping to a non-existent GEFF property raises an error."""
    store, _ = valid_geff
    node_name_map = {
        "time": "t",
        "pos": ["y", "x"],
        "area": "nonexistent_column",  # This property doesn't exist in the GEFF
    }

    with pytest.raises(ValueError):
        import_from_geff(store, node_name_map)


def _make_mask(bbox):
    """Create a Mask from a bbox list [y_min, x_min, y_max, x_max]."""
    from tracksdata.nodes import Mask

    ndim = len(bbox) // 2
    shape = tuple(bbox[i + ndim] - bbox[i] for i in range(ndim))
    return Mask(np.ones(shape, dtype=bool), bbox=np.array(bbox, dtype=np.int64))


def test_import_from_geff_roundtrip_auto_axes(tmp_path):
    """Round-trip export_to_geff / import_from_geff for a graph with mask/bbox node
    attributes but no accompanying segmentation array.

    This is the typical shape of a motile-tracker candidate graph: each node carries
    a per-node Mask (local boolean array + bounding box) without a full dense
    segmentation volume being stored.

    Checks:
    - The time and spatial axes are inferred correctly from the geff axes metadata,
      not from fuzzy string matching (which would misassign 'pos' -> ['bbox']).
    - import_from_geff succeeds and sets segmentation=None when the geff file was
      saved without a segmentation array (i.e. no segmentation_shape in metadata).
    - Positions and scalar attributes are preserved through the round-trip.
    - The mask attribute comes back as a Mask object, not a raw numpy array.
    - The bbox values are preserved correctly through the round-trip.
    """
    import tracksdata as td
    from tracksdata.nodes import Mask

    graph = create_empty_graphview_graph(
        node_attributes=[
            "pos",
            "area",
            "track_id",
            "lineage_id",
            td.DEFAULT_ATTR_KEYS.MASK,
            td.DEFAULT_ATTR_KEYS.BBOX,
        ],
        edge_attributes=["iou"],
        ndim=3,
    )
    bbox = [30, 30, 71, 71]
    graph.bulk_add_nodes(
        nodes=[
            {
                "t": 0,
                "pos": np.array([50.0, 50.0]),
                "area": 1681.0,
                "track_id": 1,
                "lineage_id": 1,
                "solution": 1,
                td.DEFAULT_ATTR_KEYS.MASK: _make_mask(bbox),
                td.DEFAULT_ATTR_KEYS.BBOX: np.array(bbox, dtype=np.int64),
            }
        ],
        indices=[1],
    )
    # The graph carries segmentation_shape in its metadata (set by motile-tracker),
    # but no dense segmentation array is attached to the SolutionTracks object.
    graph._update_metadata(segmentation_shape=(5, 100, 100))

    run_dir = tmp_path / "run"
    run_dir.mkdir()

    st = SolutionTracks(graph, ndim=3, time_attr="t")
    export_to_geff(st, run_dir)

    tracks_path = run_dir / "tracks.geff"

    # The geff file contains typed axis metadata (type="time" / type="space").
    # The builder should use that directly instead of fuzzy string matching,
    # which would assign pos -> ['bbox'] and leave 'y'/'x' unmapped.
    builder = GeffTracksBuilder()
    builder.prepare(tracks_path)

    assert "time" in builder.node_name_map
    assert builder.node_name_map["time"] == "t"
    assert "pos" in builder.node_name_map
    pos_mapping = builder.node_name_map["pos"]
    assert isinstance(pos_mapping, list), (
        f"pos should map to a list of axis names, got {pos_mapping!r}"
    )
    assert pos_mapping == ["y", "x"], f"pos should map to ['y', 'x'], got {pos_mapping}"

    # export_to_geff writes segmentation_shape as an extra zarr attribute when the
    # graph carries mask/bbox node attributes. Verify it is present in the zarr.
    import zarr as _zarr

    z = _zarr.open(str(tracks_path), mode="r")
    zarr_attrs = dict(z.attrs)
    assert "segmentation_shape" in zarr_attrs, (
        "export_to_geff should write segmentation_shape to zarr attrs when masks present"
    )
    assert tuple(zarr_attrs["segmentation_shape"]) == (5, 100, 100)

    # import_from_geff must read segmentation_shape back from zarr attrs and
    # reconstruct a segmentation (GraphArrayView) — not return segmentation=None.
    tracks = import_from_geff(tracks_path)
    assert tracks.graph.num_nodes() == 1
    assert tracks.segmentation is not None, (
        "segmentation should be reconstructed from masks after round-trip"
    )
    assert tracks.segmentation.shape == (5, 100, 100)

    node1 = tracks.graph.nodes[1]

    assert node1["pos"] is not None
    np.testing.assert_array_almost_equal(node1["pos"], [50.0, 50.0])
    assert node1["area"] == pytest.approx(1681.0)

    # Regression test: RegionpropsAnnotator must be present and active after
    # round-trip import of a GEFF with embedded segmentation. Without the fix,
    # segmentation=None during Tracks.__init__ so the annotator was never
    # created, causing newly painted cells to get position (0, 0) and area 0.
    from funtracks.annotators import RegionpropsAnnotator

    assert any(isinstance(a, RegionpropsAnnotator) for a in tracks.annotators), (
        "RegionpropsAnnotator should be in the annotator registry after importing "
        "a GEFF with embedded segmentation (mask + bbox + segmentation_shape)"
    )
    regionprops = next(
        a for a in tracks.annotators if isinstance(a, RegionpropsAnnotator)
    )
    assert "pos" in regionprops.features, (
        "'pos' should be activated in RegionpropsAnnotator so that newly "
        "painted cells get their position computed correctly"
    )

    # The geff format stores mask data as a plain numeric array; funtracks must
    # reconstruct the Mask wrapper from the raw array + bbox after loading.
    loaded_mask = node1[td.DEFAULT_ATTR_KEYS.MASK]
    loaded_bbox = node1[td.DEFAULT_ATTR_KEYS.BBOX]

    assert isinstance(loaded_mask, Mask), (
        f"mask should be a Mask object after round-trip, got {type(loaded_mask)}"
    )
    # bbox is stored as an array column in the SQL-backed graph so it comes back
    # as a polars Series rather than a numpy array; check values only.
    np.testing.assert_array_equal(
        np.array(loaded_bbox, dtype=np.int64), np.array(bbox, dtype=np.int64)
    )


def test_import_from_geff_warns_missing_segmentation_shape(tmp_path):
    """import_from_geff should warn when masks/bboxes are present but
    segmentation_shape is absent from the zarr attributes.

    This simulates a GEFF written by an older version of funtracks (before the
    export fix) or by an external tool that stores per-node masks without writing
    segmentation_shape. The import must still succeed, return segmentation=None,
    and emit a UserWarning so the user knows the segmentation cannot be shown.
    """
    import warnings

    import tracksdata as td
    import zarr as _zarr

    graph = create_empty_graphview_graph(
        node_attributes=[
            "pos",
            td.DEFAULT_ATTR_KEYS.MASK,
            td.DEFAULT_ATTR_KEYS.BBOX,
        ],
        ndim=3,
    )
    bbox = [30, 30, 71, 71]
    graph.bulk_add_nodes(
        nodes=[
            {
                "t": 0,
                "pos": np.array([50.0, 50.0]),
                "solution": 1,
                td.DEFAULT_ATTR_KEYS.MASK: _make_mask(bbox),
                td.DEFAULT_ATTR_KEYS.BBOX: np.array(bbox, dtype=np.int64),
            }
        ],
        indices=[1],
    )
    graph._update_metadata(segmentation_shape=(5, 100, 100))

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    st = SolutionTracks(graph, ndim=3, time_attr="t")
    export_to_geff(st, run_dir)

    tracks_path = run_dir / "tracks.geff"

    # Simulate old funtracks / external tool: remove segmentation_shape from zarr attrs.
    # Use put() (full replacement) rather than update() (merge) so the key is truly gone.
    z = _zarr.open(str(tracks_path), mode="a")
    attrs = dict(z.attrs)
    attrs.pop("segmentation_shape", None)
    z.attrs.put(attrs)

    # import_from_geff must warn and still succeed (segmentation=None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tracks = import_from_geff(tracks_path)

    warning_messages = [
        str(w.message) for w in caught if issubclass(w.category, UserWarning)
    ]
    assert any("segmentation_shape" in msg for msg in warning_messages), (
        f"Expected a UserWarning mentioning segmentation_shape, got: {warning_messages}"
    )
    assert tracks.segmentation is None


def test_get_time_works_after_import(valid_geff):
    """Regression test: tracks.get_time() must work on a SolutionTracks returned by
    import_from_geff().

    Previously, TracksBuilder.build() stored time as "t" in the graph (tracksdata
    convention) but created SolutionTracks(time_attr=TIME_ATTR) where TIME_ATTR="time".
    This caused features.time_key="time" while the graph only had attribute "t",
    making get_time() raise KeyError: 'time'.
    """
    store, _ = valid_geff
    name_map = {"time": "t", "pos": ["y", "x"]}
    tracks = import_from_geff(store, name_map)

    for node_id in tracks.graph.node_ids():
        # This must not raise KeyError: 'time'
        t = tracks.get_time(node_id)
        assert isinstance(t, int), f"get_time() should return int, got {type(t)}"

    # get_times() on all nodes must also work
    all_node_ids = list(tracks.graph.node_ids())
    times = tracks.get_times(all_node_ids)
    assert len(times) == len(all_node_ids)


@pytest.fixture
def geff_with_bool_prop():
    """A minimal GEFF store that contains a np.bool_ node property."""
    store, _ = create_mock_geff(
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
            "is_dividing": np.array([True, False, True, False, True], dtype=np.bool_),
        },
    )
    return store


def test_bool_node_property_schema(geff_with_bool_prop):
    """Fix 1: np.bool_ columns must produce a pl.Boolean schema, not pl.Int64.

    In numpy 2.x, np.bool_ is no longer a subtype of np.integer.  Before the
    fix, construct_graph() fell through to ``else: default_value = 0`` (Python
    int), making polars infer a pl.Int64 schema.  Building a pl.Series of type
    Int64 from np.bool_ values then raised:
        TypeError: unexpected value while building Series of type Int64;
                   found value of type Float64: 1.0
    """
    import polars as pl

    name_map = {"time": "t", "pos": ["y", "x"], "is_dividing": "is_dividing"}
    tracks = import_from_geff(geff_with_bool_prop, name_map)

    df = tracks.graph.node_attrs(attr_keys=["is_dividing"])
    assert df["is_dividing"].dtype == pl.Boolean, (
        f"Expected pl.Boolean schema for 'is_dividing', got {df['is_dividing'].dtype}. "
        "Likely cause: np.bool_ default_value fell through to int in construct_graph()."
    )


def test_bool_node_property_values(geff_with_bool_prop):
    """Fix 2: np.bool_ values must be converted to Python bool in construct_graph.

    Even when the polars schema is correctly pl.Boolean (Fix 1), individual
    np.bool_ values in the node-attrs dict must be explicitly cast to Python
    bool.  Without the cast, the value stored in the graph node dict is an
    np.bool_ object, which is not an instance of Python bool and can cause
    type errors in downstream code that calls isinstance(val, bool).
    """
    name_map = {"time": "t", "pos": ["y", "x"], "is_dividing": "is_dividing"}
    tracks = import_from_geff(geff_with_bool_prop, name_map)

    node_ids = sorted(tracks.graph.node_ids())
    expected = [True, False, True, False, True]
    for node_id, exp in zip(node_ids, expected, strict=True):
        val = tracks.graph.nodes[node_id]["is_dividing"]
        assert type(val) is bool, (
            f"Expected Python bool for 'is_dividing', got {type(val)} for node {node_id}"
            "Likely cause: np.bool_ value not cast to bool in construct_graph()."
        )
        assert val == exp, f"Wrong value for node {node_id}: expected {exp}, got {val}"


def test_3d_pos_survives_sql_roundtrip(tmp_path):
    """Regression test: 3D pos (z, y, x) must keep Array dtype through SQL roundtrip.

    The construct_graph() method must pass the correct ndim to
    create_empty_graphview_graph() so the pos schema is Array(Float64, 3) not
    Array(Float64, 2). A schema mismatch causes SQLGraph.from_other() to
    downgrade the column to List(Float64), which breaks downstream callers that
    rely on to_numpy() returning a 2D float64 array.
    """
    import polars as pl
    import tracksdata as td

    store, _ = create_mock_geff(
        node_id_dtype="uint",
        node_axis_dtypes={"position": "float64", "time": "int64"},
        directed=True,
        num_nodes=5,
        num_edges=2,
        include_t=True,
        include_z=True,
        include_y=True,
        include_x=True,
        extra_node_props={"track_id": np.arange(5)},
    )
    name_map = {"time": "t", "pos": ["z", "y", "x"]}
    tracks = import_from_geff(store, name_map)

    # Verify the RX graph has correct Array dtype for 3D pos
    df_rx = tracks.graph.node_attrs(attr_keys=["pos"])
    assert df_rx["pos"].dtype == pl.Array(pl.Float64, 3), (
        f"RX graph pos should be Array(Float64, 3), got {df_rx['pos'].dtype}"
    )

    # Convert to SQL and reload — this is where the schema mismatch used to surface
    db_path = str(tmp_path / "test.db")
    td.graph.SQLGraph.from_other(tracks.graph, drivername="sqlite", database=db_path)
    sql_graph2 = td.graph.SQLGraph("sqlite", db_path)

    df_sql = sql_graph2.node_attrs(attr_keys=["pos"])
    assert df_sql["pos"].dtype == pl.Array(pl.Float64, 3), (
        f"Reloaded SQL pos should be Array(Float64, 3), got {df_sql['pos'].dtype}. "
        "This likely means construct_graph() used the wrong ndim when registering "
        "the pos schema, causing a mismatch with the actual 3-element data."
    )
