import json
from collections.abc import Sequence

import pytest
from networkx.utils import graphs_equal
from numpy.testing import assert_array_almost_equal

from funtracks.data_model import Tracks
from funtracks.import_export.internal_format import (
    delete_tracks,
    load_tracks,
    save_tracks,
)


@pytest.mark.parametrize("with_seg", [True, False])
@pytest.mark.parametrize("ndim", [3, 4])
@pytest.mark.parametrize("is_solution", [True, False])
def test_save_load(
    get_tracks,
    with_seg,
    ndim,
    is_solution,
    tmp_path,
):
    tracks = get_tracks(ndim=ndim, with_seg=with_seg, is_solution=is_solution)
    save_tracks(tracks, tmp_path)

    loaded = load_tracks(tmp_path, solution=is_solution)
    assert loaded.ndim == tracks.ndim
    # Check feature keys and important properties match (allow tuple vs list diff)
    assert loaded.features.time_key == tracks.features.time_key
    assert loaded.features.position_key == tracks.features.position_key

    # Check that features dictionaries have same keys
    assert set(loaded.features.keys()) == set(tracks.features.keys())

    # Check that each feature has matching values
    for key in tracks.features:
        loaded_feature = loaded.features[key]
        tracks_feature = tracks.features[key]

        for attr_name, attr_value in tracks_feature.items():
            loaded_attr_value = loaded_feature[attr_name]

            # For sequence attributes, cast to list to compare (handles tuple vs list)
            if isinstance(attr_value, Sequence) and not isinstance(attr_value, str):
                assert list(loaded_attr_value) == list(attr_value), (
                    f"Feature '{key}' attribute '{attr_name}' mismatch: "
                    f"{loaded_attr_value} != {attr_value}"
                )
            # For non-sequence attributes, direct equality
            else:
                assert loaded_attr_value == attr_value, (
                    f"Feature '{key}' attribute '{attr_name}' mismatch: "
                    f"{loaded_attr_value} != {attr_value}"
                )

    assert loaded.scale == tracks.scale
    assert loaded.ndim == tracks.ndim

    if is_solution:
        loaded_annotator = loaded.track_annotator
        tracks_annotator = tracks.track_annotator
        assert (
            loaded_annotator.tracklet_id_to_nodes == tracks_annotator.tracklet_id_to_nodes
        )

    if with_seg:
        assert_array_almost_equal(loaded.segmentation, tracks.segmentation)
    else:
        assert loaded.segmentation is None

    assert graphs_equal(loaded.graph, tracks.graph)


@pytest.mark.parametrize("with_seg", [True, False])
@pytest.mark.parametrize("ndim", [3, 4])
@pytest.mark.parametrize("is_solution", [True, False])
def test_delete(
    get_tracks,
    with_seg,
    ndim,
    is_solution,
    tmp_path,
):
    tracks_path = tmp_path / "test_tracks"
    tracks = get_tracks(ndim=ndim, with_seg=with_seg, is_solution=is_solution)
    save_tracks(tracks, tracks_path)
    delete_tracks(tracks_path)
    with pytest.raises(StopIteration):
        next(tmp_path.iterdir())


# for backward compatibility
def test_load_without_features(tmp_path, graph_2d_with_computed_features):
    tracks = Tracks(graph_2d_with_computed_features, ndim=3)
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
    assert imported_tracks.features.time_key == "time"
    assert imported_tracks.features.position_key == "pos"
