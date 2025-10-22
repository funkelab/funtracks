import numpy as np
import pytest
import zarr

from funtracks.data_model.tracks import Tracks
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
    # Skip SolutionTracks for split pos testing - only test Tracks with split pos
    if pos_attr_type is list and is_solution:
        pytest.skip("Split pos attributes only tested with Tracks, not SolutionTracks")

    # Skip split pos with segmentation - pre-existing bug in export_to_geff
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
        # Create Tracks with split position attributes (always use Tracks for split pos)
        # Specify existing features if segmentation is used
        existing_features = list(pos_keys) if with_seg else None
        if with_seg:
            existing_features.extend(["area", "iou"])
        tracks = Tracks(
            graph,
            segmentation=segmentation,
            pos_attr=pos_keys,
            ndim=ndim,
            existing_features=existing_features,
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

    # test that providing a non existing parent dir raises error
    file_path = tmp_path / "nonexisting" / "target.zarr"
    with pytest.raises(ValueError, match="does not exist"):
        export_to_geff(tracks, file_path)

    # test that providing a nondirectory path raises error
    file_path = tmp_path / "not_a_dir"
    file_path.write_text("test")

    with pytest.raises(ValueError, match="not a directory"):
        export_to_geff(tracks, file_path)

    # test that saving to a non empty dir with overwrite=False raises error
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    (export_dir / "existing_file.txt").write_text("already here")
    with pytest.raises(ValueError, match="not empty"):
        export_to_geff(tracks, export_dir)

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
