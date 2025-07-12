import pytest

from funtracks.features import Position, Time, ValueType


@pytest.mark.parametrize("key", [None, "t"])
def test_time(key):
    feat = Time(key=key)
    assert feat.key == key if key is not None else "time"
    assert feat.required
    assert not feat.recompute
    assert feat.display_name == "Time"
    assert feat.value_type == ValueType.int


@pytest.mark.parametrize("key", [None, "location"])
@pytest.mark.parametrize("recompute", [False, True])
@pytest.mark.parametrize(("axes", "num_values"), [(["x", "y"], 2), (["z", "y", "x"], 3)])
def test_position(key, recompute, axes, num_values):
    feat = Position(key=key, recompute=recompute, axes=axes)
    assert feat.key == key if key is not None else "pos"
    assert feat.required
    assert feat.recompute == recompute
    assert feat.display_name == axes
    assert feat.value_type == ValueType.float
    assert feat.num_values == num_values
