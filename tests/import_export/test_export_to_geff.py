import numpy as np
import pytest
import zarr

from funtracks.data_model import SolutionTracks, Tracks
from funtracks.import_export.export_to_geff import export_to_geff


@pytest.mark.parametrize("ndim", [3, 4])
@pytest.mark.parametrize("with_seg", [True, False])
@pytest.mark.parametrize("is_solution", [True, False])
@pytest.mark.parametrize("pos_attr_type", (str, list))
def test_export_to_geff(
    get_tracks,
    get_graph,
    get_segmentation,
    ndim,
    with_seg,
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
        graph = get_graph(ndim, with_features="computed")
        segmentation = get_segmentation(ndim) if with_seg else None

        # Determine position attribute keys based on dimensions
        pos_keys = ["y", "x"] if ndim == 3 else ["z", "y", "x"]
        # Split the composite position attribute into separate attributes
        for node in graph.nodes():
            pos = graph.nodes[node]["pos"]
            for i, key in enumerate(pos_keys):
                graph.nodes[node][key] = pos[i]
            del graph.nodes[node]["pos"]
        # Create Tracks with split position attributes
        # Features like area, track_id will be auto-detected from the graph
        tracks_cls = SolutionTracks if is_solution else Tracks
        tracks = tracks_cls(
            graph,
            segmentation=segmentation,
            time_attr="t",
            pos_attr=pos_keys,
            tracklet_attr="track_id",
            ndim=ndim,
        )
    else:
        # Use get_tracks fixture for the simple case
        tracks = get_tracks(ndim=ndim, with_seg=with_seg, is_solution=is_solution)
    export_to_geff(tracks, tmp_path)
    z = zarr.open((tmp_path / "tracks").as_posix(), mode="r")
    assert isinstance(z, zarr.Group)

    # Check that segmentation was saved (only when using segmentation)
    if with_seg:
        seg_path = tmp_path / "segmentation"
        seg_zarr = zarr.open(str(seg_path), mode="r")
        assert isinstance(seg_zarr, zarr.Array)
        np.testing.assert_array_equal(seg_zarr[:], tracks.segmentation)

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

    export_to_geff(tracks, export_dir, overwrite=True)
    z = zarr.open((export_dir / "tracks").as_posix(), mode="r")
    assert isinstance(z, zarr.Group)

    # Check segmentation only when it was used
    if with_seg:
        seg_path = export_dir / "segmentation"
        seg_zarr = zarr.open(str(seg_path), mode="r")
        assert isinstance(seg_zarr, zarr.Array)
        np.testing.assert_array_equal(seg_zarr[:], tracks.segmentation)

    # Test exporting a subset of nodes
    export_dir = tmp_path / "export3"
    export_dir.mkdir()

    # We expect that nodes 1 and 3 are also included because they are ancestors of node 4
    node_ids = [4, 6]
    export_to_geff(tracks, export_dir, node_ids=node_ids)
    z = zarr.open((export_dir / "tracks").as_posix(), mode="r")
    assert isinstance(z, zarr.Group)

    # Navigate to the dataset containing node IDs and check that we only have the
    # expected nodes
    node_ids_array = z["nodes/ids"][:]
    assert np.array_equal(np.sort(node_ids_array), np.array([1, 3, 4, 6])), (
        f"Unexpected node IDs: found {node_ids_array}, expected {[1, 3, 4, 6]}"
    )

    # Check segmentation only when it was used
    if with_seg:
        seg_path = export_dir / "segmentation"
        seg_zarr = zarr.open(str(seg_path), mode="r")
        assert isinstance(seg_zarr, zarr.Array)

        filtered_seg = tracks.segmentation.copy()
        mask = np.isin(filtered_seg, [1, 3, 4, 6])
        filtered_seg[~mask] = 0
        np.testing.assert_array_equal(seg_zarr[:], filtered_seg)
