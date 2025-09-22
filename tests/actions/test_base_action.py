import pytest

from funtracks.actions import (
    TracksAction,
)
from funtracks.data_model import SolutionTracks


def test_initialize_base_class(graph_2d):
    tracks = SolutionTracks(graph_2d, ndim=3)
    action = TracksAction(tracks)
    with pytest.raises(NotImplementedError):
        action.inverse()
