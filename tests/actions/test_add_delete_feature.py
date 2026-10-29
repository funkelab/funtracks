import numpy as np
import pytest

from funtracks.actions import AddFeature, DeleteFeature, UpdateNodeAttrs
from funtracks.features import Feature
from funtracks.user_actions import UserDeleteNode


def _group_feature() -> Feature:
    return Feature(
        feature_type="node",
        value_type="bool",
        num_values=1,
        display_name="group",
        default_value=False,
    )


@pytest.mark.parametrize("ndim", [3, 4])
@pytest.mark.parametrize("with_seg", [True, False])
def test_add_feature(get_tracks, ndim, with_seg):
    tracks = get_tracks(ndim=ndim, with_seg=with_seg, prefill_track_ids=True)
    action = AddFeature(tracks, "group", _group_feature())
    assert "group" in tracks.features
    assert "group" in tracks.graph_full.node_attr_keys()
    assert tracks.get_node_attr(1, "group") is False

    inverse = action.inverse()
    assert "group" not in tracks.features
    assert "group" not in tracks.graph_full.node_attr_keys()

    inverse.inverse()
    assert "group" in tracks.features
    assert tracks.get_node_attr(1, "group") is False


@pytest.mark.parametrize("ndim", [3, 4])
@pytest.mark.parametrize("with_seg", [True, False])
def test_delete_feature_restores_values(get_tracks, ndim, with_seg):
    tracks = get_tracks(ndim=ndim, with_seg=with_seg, prefill_track_ids=True)
    feature = _group_feature()
    AddFeature(tracks, "group", feature)
    UpdateNodeAttrs(tracks, 1, {"group": True})
    UpdateNodeAttrs(tracks, 2, {"group": True})
    # A soft-deleted node keeps its value in graph_full, and must get it back
    UserDeleteNode(tracks, 2)
    assert not tracks.graph_solution.has_node(2)

    action = DeleteFeature(tracks, "group")
    assert "group" not in tracks.features
    assert "group" not in tracks.graph_full.node_attr_keys()
    np.testing.assert_array_equal(np.sort(action.values[0]), [1, 2])

    inverse = action.inverse()
    assert tracks.features["group"] == feature
    assert tracks.get_node_attr(1, "group") is True
    assert tracks.get_node_attr(2, "group") is True
    assert tracks.get_node_attr(3, "group") is False

    inverse.inverse()
    assert "group" not in tracks.features
    assert "group" not in tracks.graph_full.node_attr_keys()


def test_delete_feature_object_values(get_tracks):
    """Columns that are not numeric take the slow path, which must also be sparse."""
    tracks = get_tracks(ndim=3, with_seg=False, prefill_track_ids=True)
    feature = Feature(
        feature_type="node",
        value_type="str",
        num_values=1,
        display_name="label",
        default_value="",
    )
    AddFeature(tracks, "label", feature)
    UpdateNodeAttrs(tracks, 3, {"label": "a"})

    action = DeleteFeature(tracks, "label")
    assert action.values[0].tolist() == [3]

    action.inverse()
    assert tracks.get_node_attr(3, "label") == "a"
    assert tracks.get_node_attr(1, "label") == ""


def test_add_existing_feature_raises(get_tracks):
    tracks = get_tracks(ndim=3, with_seg=True, prefill_track_ids=True)
    AddFeature(tracks, "group", _group_feature())
    with pytest.raises(ValueError, match="already exists"):
        AddFeature(tracks, "group", _group_feature())


def test_add_annotator_feature_raises(get_tracks):
    tracks = get_tracks(ndim=3, with_seg=True, prefill_track_ids=True)
    tracks.disable_features(["area"])
    with pytest.raises(ValueError, match="computed by an annotator"):
        AddFeature(tracks, "area", _group_feature())


@pytest.mark.parametrize("key", ["t", "pos", "mask", "bbox", "solution", "track_id"])
def test_delete_core_feature_raises(get_tracks, key):
    tracks = get_tracks(ndim=3, with_seg=True, prefill_track_ids=True)
    with pytest.raises(ValueError, match="core feature"):
        DeleteFeature(tracks, key)


def test_delete_annotator_feature_raises(get_tracks):
    tracks = get_tracks(ndim=3, with_seg=True, prefill_track_ids=True)
    with pytest.raises(ValueError, match="computed by an annotator"):
        DeleteFeature(tracks, "area")


def test_delete_missing_feature_raises(get_tracks):
    tracks = get_tracks(ndim=3, with_seg=True, prefill_track_ids=True)
    with pytest.raises(ValueError, match="does not exist"):
        DeleteFeature(tracks, "group")


def test_edge_feature_raises(get_tracks):
    tracks = get_tracks(ndim=3, with_seg=True, prefill_track_ids=True)
    feature = _group_feature()
    feature["feature_type"] = "edge"
    with pytest.raises(NotImplementedError):
        AddFeature(tracks, "edge_group", feature)

    tracks.add_feature("edge_group", feature)
    with pytest.raises(NotImplementedError):
        DeleteFeature(tracks, "edge_group")
