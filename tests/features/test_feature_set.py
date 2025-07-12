import pytest

from funtracks.features import Feature, FeatureSet, FeatureType, Position, Time


class TestFeatureSet:
    def test_init(self):
        fs = FeatureSet(Time(), Position(("y", "x")))
        assert len(fs._features) == 2
        assert fs.time == Time()
        assert fs.position == Position(("y", "x"))

        with pytest.raises(KeyError, match="Key pos already in feature set"):
            FeatureSet(Time("pos"), Position(("y", "x")))

        extra_features = [
            Feature(key="1", feature_type=FeatureType.NODE, value_type=int),
            Feature(key="1", feature_type=FeatureType.EDGE, value_type=float),
        ]
        fs = FeatureSet(Time(), Position(("y", "x")), extra_features=extra_features)
        assert len(fs._features) == 4

        extra_features[0].feature_type = FeatureType.EDGE
        with pytest.raises(KeyError, match="Key 1 already in feature set"):
            fs = FeatureSet(Time(), Position(("y", "x")), extra_features=extra_features)

    @pytest.mark.parametrize(
        "extra_features",
        [
            [],
            [
                Feature(
                    key="1",
                    feature_type=FeatureType.NODE,
                    value_type=str,
                    display_name="TEST",
                )
            ],
        ],
    )
    def test_json(self, extra_features):
        fs = FeatureSet(Time(), Position(("z", "y", "x")), extra_features=extra_features)
        json_dict = fs.dump_json()
        assert "FeatureSet" in json_dict
        assert len(json_dict["FeatureSet"]) == 2 + len(extra_features)

        imported_fs = FeatureSet.from_json(json_dict)
        # can't assert equality because the Time class is now a generic Feature
        for feat, loaded_feat in zip(fs._features, imported_fs._features, strict=True):
            assert feat.model_dump() == loaded_feat.model_dump()
