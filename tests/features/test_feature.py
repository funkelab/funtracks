import numpy as np
import pytest
from pydantic import ValidationError

from funtracks.features import Feature, FeatureType


def test_base_feature():
    with pytest.raises(ValidationError):
        feat = Feature()
    with pytest.raises(ValidationError):
        feat = Feature(key="test")
    with pytest.raises(ValidationError):
        feat = Feature(key="test", feature_type=FeatureType.NODE)
    feat = Feature(key="test", feature_type=FeatureType.NODE, value_type=int)
    assert hash(feat) == hash(("node", "test"))
    assert str(feat) == "node_test"

    feat = Feature(
        key="computed",
        feature_type=FeatureType.EDGE,
        value_type=np.float32,
        num_values=3,
        display_name=["1", "2", "3"],
        recompute=True,
        required=False,
        default_value=0.0,
    )

    assert hash(feat) == hash(("edge", "computed"))
    assert feat.key == "computed"
    assert feat.feature_type == FeatureType.EDGE
    assert feat.value_type == np.float32
    assert feat.display_name == ["1", "2", "3"]
    assert feat.recompute
    assert not feat.required
    assert feat.default_value == 0.0
