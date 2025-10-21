import numpy as np
import pytest
import zarr

from funtracks.data_model.solution_tracks import SolutionTracks
from funtracks.data_model.tracks import Tracks
from funtracks.import_export.export_to_geff import export_to_geff


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("track_type", (Tracks, SolutionTracks))
@pytest.mark.parametrize("pos_attr_type", (str, list))
def test_export_to_geff(
    ndim,
    track_type,
    pos_attr_type,
    tmp_path,
    request,
):
    if ndim == 2:
        graph = request.getfixturevalue("graph_2d")
        segmentation = request.getfixturevalue("segmentation_2d")
    else:
        graph = request.getfixturevalue("graph_3d")
        segmentation = request.getfixturevalue("segmentation_3d")

    # in the case the pos_attr_type is a list, split the position values over multiple
    # attributes to create a list type pos_attr.
    if pos_attr_type is list:
        # Determine position attribute keys based on dimensions
        pos_keys = ["y", "x"] if ndim == 2 else ["z", "y", "x"]
        # Split the composite position attribute into separate attributes
        for node in graph.nodes():
            pos = graph.nodes[node]["pos"]
            for i, key in enumerate(pos_keys):
                graph.nodes[node][key] = pos[i]
            del graph.nodes[node]["pos"]
        # Don't use segmentation when testing split pos, as it would compute centroids
        tracks = track_type(graph, segmentation=None, pos_attr=pos_keys, ndim=ndim + 1)
    else:
        tracks = track_type(graph, segmentation=segmentation, ndim=ndim + 1)
    export_to_geff(tracks, tmp_path)
    z = zarr.open((tmp_path / "tracks").as_posix(), mode="r")
    assert isinstance(z, zarr.Group)

    # Check that segmentation was saved (only when using segmentation)
    if pos_attr_type is not list:
        seg_path = tmp_path / "segmentation"
        seg_zarr = zarr.open(str(seg_path), mode="r")
        assert isinstance(seg_zarr, zarr.Array)
        np.testing.assert_array_equal(seg_zarr[:], segmentation)

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
    if pos_attr_type is not list:
        seg_path = export_dir / "segmentation"
        seg_zarr = zarr.open(str(seg_path), mode="r")
        assert isinstance(seg_zarr, zarr.Array)
        np.testing.assert_array_equal(seg_zarr[:], segmentation)
