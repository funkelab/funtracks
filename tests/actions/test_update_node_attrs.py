import pytest

from funtracks.actions import (
    UpdateNodeAttrs,
)


@pytest.mark.parametrize("ndim", [3, 4])
def test_update_node_attrs(get_tracks, ndim):
    tracks = get_tracks(ndim=ndim, with_seg=True, is_solution=True)
    node = 1
    new_attr = {"score": 1.0}

    action = UpdateNodeAttrs(tracks, node, new_attr)
    assert tracks.get_node_attr(node, "score") == 1.0

    inverse = action.inverse()
    assert tracks.get_node_attr(node, "score") is None

    inverse.inverse()
    assert tracks.get_node_attr(node, "score") == 1.0


@pytest.mark.parametrize("attr", ["t", "area", "track_id"])
def test_update_protected_attr(get_tracks, attr):
    tracks = get_tracks(ndim=3, with_seg=True, is_solution=True)
    with pytest.raises(ValueError, match="Cannot update attribute .* manually"):
        UpdateNodeAttrs(tracks, 1, {attr: 2})
