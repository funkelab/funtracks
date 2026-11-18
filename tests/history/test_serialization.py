import json

import numpy as np
import polars as pl
import pytest
from tracksdata.nodes import Mask

from funtracks.actions import (
    AddEdge,
    AddFeature,
    AddNode,
    BasicAction,
    DeleteEdge,
    DeleteFeature,
    DeleteNode,
    UpdateNodeAttrs,
    UpdateNodeSeg,
    UpdateTrackIDs,
)
from funtracks.actions.serialization import (
    action_from_dict,
    action_registry,
    decode_value,
    encode_value,
)
from funtracks.cli import compare_tracks
from funtracks.history import recompute_track_ids

from ._helpers import full_graph_state, group_feature


def json_round_trip(value):
    return decode_value(json.loads(json.dumps(encode_value(value), allow_nan=True)))


@pytest.mark.parametrize(
    "array",
    [
        np.array([1.5, 2.25]),
        np.array([1, 2, 3], dtype=np.int32),
        np.array([True, False]),
        np.arange(40, dtype=np.uint16),
        np.arange(12, dtype=np.float32).reshape(3, 4),
    ],
)
def test_arrays_keep_dtype_and_values(array):
    decoded = json_round_trip(array)
    assert isinstance(decoded, np.ndarray)
    assert decoded.dtype == array.dtype
    np.testing.assert_array_equal(decoded, array)


def test_other_values_round_trip():
    mask = Mask(np.array([[True, False, True], [False, True, True]]), bbox=[3, 4, 5, 7])
    decoded = json_round_trip({"mask": mask, "list": [1, 2], "x": np.int64(5)})
    np.testing.assert_array_equal(decoded["mask"].mask, mask.mask)
    np.testing.assert_array_equal(decoded["mask"].bbox, mask.bbox)
    assert decoded["list"] == [1, 2]
    assert decoded["x"] == 5 and type(decoded["x"]) is int

    floats = json_round_trip([0.1, float("nan"), float("inf")])
    assert floats[0] == 0.1 and np.isnan(floats[1]) and floats[2] == float("inf")

    objects = np.empty(2, dtype=object)
    objects[:] = ["a", None]
    assert list(json_round_trip(objects)) == ["a", None]


def test_registry_has_every_basic_action_but_track_ids():
    def subclasses(cls):
        for sub in cls.__subclasses__():
            yield sub
            yield from subclasses(sub)

    expected = {cls.__name__ for cls in subclasses(BasicAction)} - {"UpdateTrackIDs"}
    assert set(action_registry()) == expected
    with pytest.raises(ValueError, match="Unknown action"):
        action_from_dict(None, {"a": "UpdateTrackIDs"})


def add_label_column(tracks):
    tracks.graph_full.add_node_attr_key("label", default_value="", dtype=pl.String)


# (with_seg only, setup, apply) for each basic action
ACTIONS = {
    "AddNode": (
        False,
        None,
        lambda t: AddNode(t, 7, {"t": 2, "pos": [5.0] * (t.ndim - 1)}),
    ),
    "AddNode_mask": (
        True,
        None,
        lambda t: AddNode(
            t,
            7,
            {
                "t": 2,
                "mask": Mask(np.ones((3, 3), bool), bbox=[1, 1, 4, 4]),
                "bbox": np.array([1, 1, 4, 4]),
            },
        ),
    ),
    "DeleteNode": (False, None, lambda t: DeleteNode(t, 6)),
    "AddEdge": (False, None, lambda t: AddEdge(t, (2, 6))),
    "DeleteEdge": (False, None, lambda t: DeleteEdge(t, (4, 5))),
    "UpdateNodeAttrs": (
        False,
        add_label_column,
        lambda t: UpdateNodeAttrs(t, 2, {"label": "x"}),
    ),
    "UpdateNodeSeg": (
        True,
        None,
        lambda t: UpdateNodeSeg(t, 5, Mask(np.ones((4, 4), bool), bbox=[0, 0, 4, 4])),
    ),
    "AddFeature": (
        False,
        None,
        lambda t: AddFeature(
            t, "group", group_feature(), values=(np.array([1, 4]), np.array([True, True]))
        ),
    ),
    "DeleteFeature": (
        False,
        lambda t: AddFeature(t, "group", group_feature()),
        lambda t: DeleteFeature(t, "group"),
    ),
}


@pytest.mark.parametrize("ndim", [3, 4])
@pytest.mark.parametrize("name", ACTIONS)
def test_action_round_trip(get_tracks, name, ndim):
    """Applying an action from its JSON dict gives the same result as the action."""
    seg_only, setup, apply = ACTIONS[name]
    if seg_only and ndim == 4:
        pytest.skip("the 2D masks above do not fit 3D tracks")
    trackses = []
    for _ in range(2):
        tracks = get_tracks(ndim=ndim, with_seg=seg_only, prefill_track_ids=True)
        if setup is not None:
            setup(tracks)
        trackses.append(tracks)
    original, replayed = trackses

    dicts = []
    original.action_applied.connect(lambda a: dicts.append(json.dumps(a.to_dict())))
    apply(original)
    assert len(dicts) == 1
    action_from_dict(replayed, json.loads(dicts[0]))

    for tracks in trackses:
        recompute_track_ids(tracks)
    assert compare_tracks(replayed, original) == []
    assert full_graph_state(replayed) == full_graph_state(original)


def test_track_ids_are_not_logged(get_tracks):
    tracks = get_tracks(ndim=3, with_seg=True, prefill_track_ids=True)
    key = tracks.features.tracklet_key
    action = DeleteNode(tracks, 6)
    assert key in action.attributes
    assert key not in action.inverse().to_dict()["attributes"]
    with pytest.raises(NotImplementedError):
        UpdateTrackIDs(tracks, 5, 99).to_dict()
