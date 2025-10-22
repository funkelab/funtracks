import pytest

from funtracks.actions import (
    TracksAction,
)


def test_initialize_base_class(get_tracks):
    tracks = get_tracks(ndim=3, with_seg=True, is_solution=True)
    action = TracksAction(tracks)
    with pytest.raises(NotImplementedError):
        action.inverse()
