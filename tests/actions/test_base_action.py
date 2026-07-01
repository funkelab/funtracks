import pytest

from funtracks.actions import (
    Action,
)


def test_initialize_base_class(get_tracks):
    tracks = get_tracks(ndim=3, with_seg=True, prefill_track_ids=True)
    action = Action(tracks)
    with pytest.raises(NotImplementedError):
        action.inverse()
