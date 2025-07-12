import json

import pytest
from networkx.utils import graphs_equal
from numpy.testing import assert_array_almost_equal

from funtracks.data_model import SolutionTracks, Tracks
from funtracks.import_export.internal_format import (
    delete_tracks,
    load_tracks,
    save_tracks,
)


@pytest.mark.parametrize("use_seg", [True, False])
@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("track_type", (Tracks, SolutionTracks))
def test_save_load(
    use_seg,
    ndim,
    track_type,
    tmp_path,
    request,
):
    if ndim == 2:
        graph = request.getfixturevalue("graph_2d")
        seg = request.getfixturevalue("segmentation_2d")
    else:
        graph = request.getfixturevalue("graph_3d")
        seg = request.getfixturevalue("segmentation_3d")
    if not use_seg:
        seg = None
    tracks = track_type(graph, seg, ndim=ndim + 1)
    save_tracks(tracks, tmp_path)

    solution = bool(issubclass(track_type, SolutionTracks))
    loaded = load_tracks(tmp_path, solution=solution)
    assert loaded.ndim == tracks.ndim
    assert loaded.pos_attr == tracks.pos_attr
    assert loaded.time_attr == tracks.time_attr
    assert loaded.scale == tracks.scale
    assert loaded.ndim == tracks.ndim

    if issubclass(track_type, SolutionTracks):
        assert loaded.track_id_to_node == tracks.track_id_to_node

    if use_seg:
        assert_array_almost_equal(loaded.segmentation, tracks.segmentation)
    else:
        assert loaded.segmentation is None

    assert graphs_equal(loaded.graph, tracks.graph)


@pytest.mark.parametrize("use_seg", [True, False])
@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("track_type", (Tracks, SolutionTracks))
def test_delete(
    use_seg,
    ndim,
    track_type,
    tmp_path,
    request,
):
    tracks_path = tmp_path / "test_tracks"
    if ndim == 2:
        graph = request.getfixturevalue("graph_2d")
        seg = request.getfixturevalue("segmentation_2d")
    else:
        graph = request.getfixturevalue("graph_3d")
        seg = request.getfixturevalue("segmentation_3d")
    if not use_seg:
        seg = None
    tracks = track_type(graph, seg, ndim=ndim + 1)
    save_tracks(tracks, tracks_path)
    delete_tracks(tracks_path)
    with pytest.raises(StopIteration):
        next(tmp_path.iterdir())


# for backward compatibility
def test_load_without_features(tmp_path, graph_2d):
    tracks = Tracks(graph_2d, ndim=3)
    tracks_path = tmp_path / "test_tracks"
    save_tracks(tracks, tracks_path)
    attrs_path = tracks_path / "attrs.json"
    with open(attrs_path) as f:
        attrs = json.load(f)

    del attrs["features"]
    attrs["time_attr"] = "time"
    attrs["pos_attr"] = "pos"
    with open(attrs_path, "w") as f:
        json.dump(attrs, f)

    imported_tracks = load_tracks(tracks_path)
    assert imported_tracks.features.time.key == "time"
    assert imported_tracks.features.position.key == "pos"
