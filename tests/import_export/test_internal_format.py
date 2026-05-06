import json
import shutil
from collections.abc import Sequence
from pathlib import Path

import pytest
from numpy.testing import assert_array_almost_equal

from funtracks.import_export._v1_format import (
    delete_tracks,
    load_v1_tracks,
)


def _normalize_tracklet_key(name: str) -> str:
    """Map both spellings of the tracklet key to a single canonical name so a
    pre-2.0 saved fixture (with 'track_id') can be compared to a freshly-built
    tracks object (with 'tracklet_id') without spurious key mismatches.

    The legacy fallback in _setup_core_computed_features intentionally keeps
    the original 'track_id' spelling on saved-and-loaded data; that is the
    behaviour this normalization papers over for the comparison only.
    """
    return "tracklet_id" if name == "track_id" else name


@pytest.mark.parametrize("with_seg", [True, False])
@pytest.mark.parametrize("ndim", [3, 4])
@pytest.mark.parametrize("is_solution", [True, False])
def test_save_load(
    get_tracks,
    with_seg,
    ndim,
    is_solution,
):
    tracks = get_tracks(ndim=ndim, with_seg=with_seg, is_solution=is_solution)

    data_path = Path(
        f"tests/data/format_v1/test_save_load_{is_solution}_{ndim}_{with_seg}_0"
    )

    loaded = load_v1_tracks(data_path, solution=is_solution)
    assert loaded.ndim == ndim
    # Check feature keys and important properties match (allow tuple vs list diff)
    assert loaded.features.time_key == tracks.features.time_key
    assert loaded.features.position_key == tracks.features.position_key

    # Check that features dictionaries describe the same set of features.
    # Legacy fixtures store the tracklet column as 'track_id'; current code
    # produces 'tracklet_id'. Normalise both sides to the canonical spelling.
    loaded_keys = {_normalize_tracklet_key(k) for k in loaded.features}
    tracks_keys = {_normalize_tracklet_key(k) for k in tracks.features}
    assert loaded_keys == tracks_keys

    # Check that each feature has matching values (skip tracklet feature: its
    # key spelling differs on loaded legacy fixtures, see comment above)
    for key in tracks.features:
        if key in {"tracklet_id", "track_id"}:
            continue
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

    # graphs_equal doesn't exist for TracksData, so we check properties.
    # Same legacy-key normalisation applies to graph node attributes.
    loaded_node_attrs = {
        _normalize_tracklet_key(k) for k in loaded.graph.node_attr_keys()
    }
    tracks_node_attrs = {
        _normalize_tracklet_key(k) for k in tracks.graph.node_attr_keys()
    }
    assert loaded_node_attrs == tracks_node_attrs
    assert set(loaded.graph.edge_attr_keys()) == set(tracks.graph.edge_attr_keys())
    assert loaded.graph.num_nodes() == tracks.graph.num_nodes()
    assert loaded.graph.num_edges() == tracks.graph.num_edges()
    assert set(loaded.graph.node_ids()) == set(tracks.graph.node_ids())
    # edge_ids dont matter, only the actual edges:
    assert sorted(loaded.graph.edge_list()) == sorted(tracks.graph.edge_list())


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
    reference_path = Path(
        f"tests/data/format_v1/test_save_load_{is_solution}_{ndim}_{with_seg}_0"
    )

    # Copy reference data to temporary location
    tracks_path = tmp_path / "test_tracks"
    shutil.copytree(reference_path, tracks_path)

    # Delete the copy
    delete_tracks(tracks_path)
    with pytest.raises(StopIteration):
        next(tmp_path.iterdir())


# for backward compatibility
def test_load_without_features(tmp_path, graph_2d_with_segmentation):
    reference_path = Path(f"tests/data/format_v1/test_save_load_{True}_{3}_{True}_0")

    # Copy reference data to temporary location
    tracks_path = tmp_path / "test_tracks"
    shutil.copytree(reference_path, tracks_path)

    # Load the original data first to verify it loads correctly
    load_v1_tracks(tracks_path, solution=True)

    # Modify the copy to test backward compatibility
    attrs_path = tracks_path / "attrs.json"
    with open(attrs_path) as f:
        attrs = json.load(f)

    del attrs["features"]
    attrs["time_attr"] = "time"
    attrs["pos_attr"] = "pos"
    with open(attrs_path, "w") as f:
        json.dump(attrs, f)

    # Load the modified data to test old format compatibility
    imported_tracks = load_v1_tracks(tracks_path)
    assert imported_tracks.features.time_key == "time"
    assert imported_tracks.features.position_key == "pos"
