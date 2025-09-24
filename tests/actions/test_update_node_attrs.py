import pytest

from funtracks.actions import (
    UpdateNodeAttrs,
)
from funtracks.data_model import SolutionTracks


def test_update_node_attrs(graph_2d):
    tracks = SolutionTracks(graph_2d, ndim=3)
    node = 1
    new_attr = {"score": 1.0}

    action = UpdateNodeAttrs(tracks, node, new_attr)
    assert tracks.get_node_attr(node, "score") == 1.0

    inverse = action.inverse()
    assert tracks.get_node_attr(node, "score") is None

    inverse.inverse()
    assert tracks.get_node_attr(node, "score") == 1.0


@pytest.mark.parametrize("attr", ["time", "area", "track_id"])
def test_update_protected_attr(attr, graph_2d):
    tracks = SolutionTracks(graph_2d, ndim=3)
    with pytest.raises(ValueError, match="Cannot update attribute .* manually"):
        UpdateNodeAttrs(tracks, 1, {attr: 2})
