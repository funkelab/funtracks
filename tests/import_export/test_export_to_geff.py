import pytest
import zarr

from funtracks.data_model.tracks import Tracks
from funtracks.import_export.export_to_geff import export_to_geff


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("track_type", (Tracks,))
def test_export_to_geff(
    ndim,
    track_type,
    tmp_path,
    request,
):
    if ndim == 2:
        graph = request.getfixturevalue("graph_2d")
    else:
        graph = request.getfixturevalue("graph_3d")
    tracks = track_type(graph, ndim=ndim + 1)
    export_to_geff(tracks, tmp_path)
    z = zarr.open(tmp_path.as_posix(), mode="r")
    assert isinstance(z, zarr.Group)

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
    z = zarr.open(export_dir.as_posix(), mode="r")
    assert isinstance(z, zarr.Group)
