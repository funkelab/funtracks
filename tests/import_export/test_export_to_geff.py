import zarr
import pytest
from funtracks.import_export.export_to_geff import export_to_geff
from funtracks.data_model.tracks import Tracks

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