import pytest

from funtracks.features import FeatureDict, Position, Time


class TestFeatureDict:
    def test_init(self):
        """Test basic initialization of FeatureDict"""
        features = {"time": Time(), "pos": Position(("y", "x"))}
        fd = FeatureDict(
            features=features, time_key="time", position_key="pos", tracklet_key=None
        )

        assert len(fd) == 2
        assert fd.time_key == "time"
        assert fd.position_key == "pos"
        assert fd["time"] == Time()
        assert fd["pos"] == Position(("y", "x"))

    def test_init_with_list_position(self):
        """Test initialization with list of position keys"""
        features = {
            "time": Time(),
            "y": {
                "feature_type": "node",
                "value_type": "float",
                "num_values": 1,
                "recompute": False,
                "required": True,
                "default_value": None,
            },
            "x": {
                "feature_type": "node",
                "value_type": "float",
                "num_values": 1,
                "recompute": False,
                "required": True,
                "default_value": None,
            },
        }
        fd = FeatureDict(
            features=features, time_key="time", position_key=["y", "x"], tracklet_key=None
        )

        assert len(fd) == 3
        assert fd.time_key == "time"
        assert fd.position_key == ["y", "x"]
        # Check that both position features exist
        assert "y" in fd
        assert "x" in fd

    def test_init_validation(self):
        """Test that init validates time and position keys exist"""
        features = {"time": Time(), "pos": Position(("y", "x"))}

        # Missing time key
        with pytest.raises(KeyError, match="time_key 'invalid' not found"):
            FeatureDict(
                features, time_key="invalid", position_key="pos", tracklet_key=None
            )

        # Missing position key
        with pytest.raises(KeyError, match="position_key 'invalid' not found"):
            FeatureDict(
                features, time_key="time", position_key="invalid", tracklet_key=None
            )

        # Missing one of multiple position keys
        with pytest.raises(KeyError, match="position_key 'z' not found"):
            FeatureDict(
                features, time_key="time", position_key=["z", "y"], tracklet_key=None
            )

    def test_node_features(self):
        """Test node_features property filters correctly"""
        features = {
            "time": Time(),
            "pos": Position(("y", "x")),
            "iou": {
                "feature_type": "edge",
                "value_type": "float",
                "num_values": 1,
                "display_name": "IoU",
                "recompute": True,
                "required": False,
                "default_value": None,
            },
        }
        fd = FeatureDict(features, time_key="time", position_key="pos", tracklet_key=None)

        node_feats = fd.node_features
        assert len(node_feats) == 2
        assert "time" in node_feats
        assert "pos" in node_feats
        assert "iou" not in node_feats

    def test_edge_features(self):
        """Test edge_features property filters correctly"""
        features = {
            "time": Time(),
            "pos": Position(("y", "x")),
            "iou": {
                "feature_type": "edge",
                "value_type": "float",
                "num_values": 1,
                "display_name": "IoU",
                "recompute": True,
                "required": False,
                "default_value": None,
            },
        }
        fd = FeatureDict(features, time_key="time", position_key="pos", tracklet_key=None)

        edge_feats = fd.edge_features
        assert len(edge_feats) == 1
        assert "iou" in edge_feats
        assert "time" not in edge_feats

    @pytest.mark.parametrize("composite_position", [True, False])
    def test_json_dump_and_load(self, composite_position):
        """Test JSON serialization and deserialization"""
        if composite_position:
            pos_key = "pos"
            features = {"time": Time(), pos_key: Position(("y", "x"))}
        else:
            pos_key = ["y", "x"]
            features = {
                "time": Time(),
                "y": {
                    "feature_type": "node",
                    "value_type": "float",
                    "num_values": 1,
                    "recompute": False,
                    "required": True,
                    "default_value": None,
                },
                "x": {
                    "feature_type": "node",
                    "value_type": "float",
                    "num_values": 1,
                    "recompute": False,
                    "required": True,
                    "default_value": None,
                },
            }

        fd = FeatureDict(
            features, time_key="time", position_key=pos_key, tracklet_key=None
        )
        json_dict = fd.dump_json()

        assert "FeatureDict" in json_dict
        assert "features" in json_dict["FeatureDict"]
        assert "time_key" in json_dict["FeatureDict"]
        assert "position_key" in json_dict["FeatureDict"]
        assert "tracklet_key" in json_dict["FeatureDict"]

        # Load back from JSON
        loaded_fd = FeatureDict.from_json(json_dict)
        assert loaded_fd.time_key == fd.time_key
        assert loaded_fd.position_key == fd.position_key
        assert len(loaded_fd) == len(fd)

        # Check features match
        for key in fd:
            assert key in loaded_fd
            assert loaded_fd[key] == fd[key]

    def test_dict_behavior(self):
        """Test that FeatureDict behaves like a dict"""
        features = {"time": Time(), "pos": Position(("y", "x"))}
        fd = FeatureDict(features, time_key="time", position_key="pos", tracklet_key=None)

        # Can iterate over keys
        keys = list(fd.keys())
        assert "time" in keys
        assert "pos" in keys

        # Can access like a dict
        assert fd["time"] == Time()
        assert fd["pos"] == Position(("y", "x"))

        # Can check membership
        assert "time" in fd
        assert "nonexistent" not in fd
