import numpy as np
import polars as pl
import pytest
import tifffile
import zarr

from funtracks.data_model import SolutionTracks, Tracks
from funtracks.import_export import export_to_geff


@pytest.mark.parametrize("ndim", [3, 4])
@pytest.mark.parametrize("with_seg", [True, False])
@pytest.mark.parametrize("save_segmentation", [True, False])
@pytest.mark.parametrize("seg_label_attr", ["track_id", None])
@pytest.mark.parametrize("is_solution", [True, False])
@pytest.mark.parametrize("pos_attr_type", (str, list))
def test_export_to_geff(
    get_tracks,
    get_graph,
    ndim,
    with_seg,
    save_segmentation,
    seg_label_attr,
    is_solution,
    pos_attr_type,
    tmp_path,
):
    # Skip split pos with segmentation - centroid will replace the list automatically
    # TODO: allow centroid with split attribute storage
    if pos_attr_type is list and with_seg:
        pytest.skip(
            "Split pos attributes with segmentation not currently supported "
            "by export_to_geff"
        )

    # in the case the pos_attr_type is a list, split the position values over multiple
    # attributes to create a list type pos_attr.
    if pos_attr_type is list:
        # For split pos, we need to manually create tracks since get_tracks
        # doesn't support this
        graph = get_graph(ndim, is_solution=is_solution, with_seg=with_seg)

        # Determine position attribute keys based on dimensions
        pos_keys = ["y", "x"] if ndim == 3 else ["z", "y", "x"]
        # Split the composite position attribute into separate attributes
        for key in pos_keys:
            graph.add_node_attr_key(key, default_value=0.0, dtype=pl.Float64)
        for node in graph.node_ids():
            pos = graph.nodes[node]["pos"]
            for i, key in enumerate(pos_keys):
                graph.nodes[node][key] = pos[i]
        graph.remove_node_attr_key("pos")
        # Create Tracks with split position attributes
        tracks_cls = SolutionTracks if is_solution else Tracks
        tracks = tracks_cls(
            graph,
            time_attr="t",
            pos_attr=pos_keys,
            tracklet_attr="track_id" if is_solution else None,
            ndim=ndim,
        )
    else:
        # Use get_tracks fixture for the simple case
        tracks = get_tracks(ndim=ndim, with_seg=with_seg, is_solution=is_solution)

    # Exporting segmentation with a label_attr not present in the graph raises ValueError.
    # Non-solution tracks don't have track_id as a node attribute.
    if (
        with_seg
        and save_segmentation
        and seg_label_attr == "track_id"
        and not is_solution
    ):
        export_dir = tmp_path / "export"
        export_dir.mkdir()
        with pytest.raises(ValueError):
            export_to_geff(
                tracks,
                export_dir,
                save_segmentation=save_segmentation,
                seg_label_attr=seg_label_attr,
            )
        return

    # Export to subdirectory to avoid conflicts with database files in tmp_path
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    export_to_geff(
        tracks,
        export_dir,
        save_segmentation=save_segmentation,
        seg_label_attr=seg_label_attr,
    )
    z = zarr.open((export_dir / "tracks").as_posix(), mode="r")
    assert isinstance(z, zarr.Group)

    # Segmentation is saved only when with_seg=True and save_segmentation=True
    seg_path = export_dir / "segmentation"
    if with_seg and save_segmentation:
        seg_zarr = zarr.open(str(seg_path), mode="r")
        assert isinstance(seg_zarr, zarr.Array)
        assert seg_zarr.shape == tracks.segmentation.shape
        unique_vals = set(seg_zarr[:].flatten()) - {0}
        if seg_label_attr is not None:
            # values should be the label_attr values (e.g. track_ids)
            label_vals = set(
                tracks.graph.node_attrs(attr_keys=[seg_label_attr])[
                    seg_label_attr
                ].to_list()
            )
            assert unique_vals == label_vals
        else:
            # values should be original node_ids
            node_ids_set = set(tracks.graph.node_ids())
            assert unique_vals == node_ids_set
    else:
        assert not seg_path.exists()

    # Check that scaling info is present in metadata
    attrs = dict(z.attrs)
    assert "geff" in attrs
    assert "axes" in attrs["geff"]
    for ax in attrs["geff"]["axes"]:
        assert ax["scale"] is not None

    # test that providing a nondirectory path raises error
    file_path = tmp_path / "not_a_dir"
    file_path.write_text("test")

    with pytest.raises(Exception):  # noqa B017 (any error is fine)
        export_to_geff(tracks, file_path)

    # Test that saving to a non empty dir with overwrite=True works fine
    export_dir = tmp_path / "export2"
    export_dir.mkdir()
    (export_dir / "existing_file.txt").write_text("already here")

    export_to_geff(
        tracks,
        export_dir,
        overwrite=True,
        save_segmentation=save_segmentation,
        seg_label_attr=seg_label_attr,
    )
    z = zarr.open((export_dir / "tracks").as_posix(), mode="r")
    assert isinstance(z, zarr.Group)

    seg_path = export_dir / "segmentation"
    if with_seg and save_segmentation:
        seg_zarr = zarr.open(str(seg_path), mode="r")
        assert isinstance(seg_zarr, zarr.Array)
        assert seg_zarr.shape == tracks.segmentation.shape
    else:
        assert not seg_path.exists()

    # Test exporting a subset of nodes
    export_dir = tmp_path / "export3"
    export_dir.mkdir()

    # We expect that nodes 1 and 3 are also included because they are ancestors of node 4
    node_ids = [4, 6]
    export_to_geff(
        tracks,
        export_dir,
        node_ids=node_ids,
        save_segmentation=save_segmentation,
        seg_label_attr=seg_label_attr,
    )
    z = zarr.open((export_dir / "tracks").as_posix(), mode="r")
    assert isinstance(z, zarr.Group)

    node_ids_array = z["nodes/ids"][:]
    assert np.array_equal(np.sort(node_ids_array), np.array([1, 3, 4, 6])), (
        f"Unexpected node IDs: found {node_ids_array}, expected {[1, 3, 4, 6]}"
    )

    seg_path = export_dir / "segmentation"
    if with_seg and save_segmentation:
        seg_zarr = zarr.open(str(seg_path), mode="r")
        assert isinstance(seg_zarr, zarr.Array)
        assert seg_zarr.shape == tracks.segmentation.shape
        if seg_label_attr is not None:
            kept_vals = set(
                tracks.graph.filter(node_ids=[1, 3, 4, 6])
                .node_attrs(attr_keys=[seg_label_attr])[seg_label_attr]
                .to_list()
            )
            unique_vals = set(seg_zarr[:].flatten()) - {0}
            assert unique_vals == kept_vals
    else:
        assert not seg_path.exists()


@pytest.mark.parametrize("ndim", [3, 4], ids=["2d", "3d"])
def test_export_to_geff_seg_tiff(get_tracks, ndim, tmp_path):
    """Test that segmentation can be exported as tiff alongside the geff graph."""
    tracks = get_tracks(ndim=ndim, with_seg=True, is_solution=True)
    export_dir = tmp_path / "export"
    export_dir.mkdir()

    export_to_geff(tracks, export_dir, seg_file_format="tiff")

    # tiff file should exist, zarr segmentation directory should not
    assert (export_dir / "segmentation.tif").exists()
    assert not (export_dir / "segmentation").exists()

    seg_arr = tifffile.imread(str(export_dir / "segmentation.tif"))
    assert seg_arr.shape == tracks.segmentation.shape

    # values should be track_ids (default seg_label_attr="track_id")
    unique_vals = set(seg_arr.flatten()) - {0}
    track_ids = set(tracks.graph.node_attrs(attr_keys=["track_id"])["track_id"].to_list())
    assert unique_vals == track_ids

    # Check metadata references the tiff path
    z = zarr.open((export_dir / "tracks").as_posix(), mode="r")
    related = dict(z.attrs)["geff"].get("related_objects", [])
    assert any("segmentation.tif" in obj["path"] for obj in related)
