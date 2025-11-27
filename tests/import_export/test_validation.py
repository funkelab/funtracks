"""Tests for validation functions in funtracks.import_export._validation."""

import pytest

from funtracks.import_export._validation import (
    _validate_edge_name_map,
    _validate_feature_key_collisions,
    _validate_node_name_map,
    validate_name_map,
)


class TestValidateNodeNameMap:
    """Test _validate_node_name_map helper function."""

    def test_valid_node_name_map(self):
        """Test that a valid node name_map passes validation."""
        name_map = {"time": "t", "x": "x_coord", "y": "y_coord"}
        importable_props = ["t", "x_coord", "y_coord", "area"]
        required_features = ["time"]
        position_attr = ["z", "y", "x"]
        ndim = 3  # 2D + time

        # Should not raise
        _validate_node_name_map(
            name_map, importable_props, required_features, position_attr, ndim
        )

    def test_missing_required_feature(self):
        """Test that missing required features raise ValueError."""
        name_map = {"x": "x_coord", "y": "y_coord"}  # Missing "time"
        importable_props = ["t", "x_coord", "y_coord"]
        required_features = ["time"]
        position_attr = ["z", "y", "x"]
        ndim = 3

        with pytest.raises(ValueError, match="None values"):
            _validate_node_name_map(
                name_map, importable_props, required_features, position_attr, ndim
            )

    def test_missing_position_attr(self):
        """Test that missing position attributes raise ValueError."""
        name_map = {"time": "t", "x": "x_coord"}  # Missing "y"
        importable_props = ["t", "x_coord", "y_coord"]
        required_features = ["time"]
        position_attr = ["z", "y", "x"]
        ndim = 3

        with pytest.raises(ValueError, match="None values"):
            _validate_node_name_map(
                name_map, importable_props, required_features, position_attr, ndim
            )

    def test_none_value_in_required_field(self):
        """Test that None values in required fields raise ValueError."""
        name_map = {"time": "t", "x": None, "y": "y_coord"}
        importable_props = ["t", "y_coord"]
        required_features = ["time"]
        position_attr = ["z", "y", "x"]
        ndim = 3

        with pytest.raises(ValueError, match="cannot contain None values"):
            _validate_node_name_map(
                name_map, importable_props, required_features, position_attr, ndim
            )

    def test_duplicate_values_in_required_fields(self):
        """Test that duplicate values in required fields raise ValueError."""
        name_map = {"time": "t", "x": "coord", "y": "coord"}  # Duplicate "coord"
        importable_props = ["t", "coord"]
        required_features = ["time"]
        position_attr = ["z", "y", "x"]
        ndim = 3

        with pytest.raises(ValueError, match="duplicate values"):
            _validate_node_name_map(
                name_map, importable_props, required_features, position_attr, ndim
            )

    def test_nonexistent_property(self):
        """Test that mapping to non-existent properties raises ValueError."""
        name_map = {"time": "t", "x": "x_coord", "y": "y_coord"}
        importable_props = ["t", "x_coord"]  # "y_coord" doesn't exist
        required_features = ["time"]
        position_attr = ["z", "y", "x"]
        ndim = 3

        with pytest.raises(ValueError, match="non-existent properties"):
            _validate_node_name_map(
                name_map, importable_props, required_features, position_attr, ndim
            )


class TestValidateEdgeNameMap:
    """Test _validate_edge_name_map helper function."""

    def test_valid_edge_name_map(self):
        """Test that a valid edge name_map passes validation."""
        edge_name_map = {"iou": "overlap", "distance": "dist"}
        importable_props = ["overlap", "dist", "weight"]

        # Should not raise
        _validate_edge_name_map(edge_name_map, importable_props)

    def test_nonexistent_edge_property(self):
        """Test that mapping to non-existent edge properties raises ValueError."""
        edge_name_map = {"iou": "overlap", "distance": "dist"}
        importable_props = ["overlap"]  # "dist" doesn't exist

        with pytest.raises(ValueError, match="non-existent properties"):
            _validate_edge_name_map(edge_name_map, importable_props)

    def test_empty_importable_props(self):
        """Test that empty importable_props list doesn't raise."""
        edge_name_map = {"iou": "overlap"}
        importable_props = []

        # Should not raise when importable_props is empty
        _validate_edge_name_map(edge_name_map, importable_props)


class TestValidateFeatureKeyCollisions:
    """Test _validate_feature_key_collisions helper function."""

    def test_no_collision(self):
        """Test that non-overlapping keys pass validation."""
        name_map = {"time": "t", "x": "x", "y": "y", "area": "area"}
        edge_name_map = {"iou": "iou", "distance": "distance"}

        # Should not raise
        _validate_feature_key_collisions(name_map, edge_name_map)

    def test_single_collision(self):
        """Test that a single colliding key raises ValueError."""
        name_map = {"time": "t", "x": "x", "y": "y", "iou": "node_iou"}
        edge_name_map = {"iou": "edge_iou", "distance": "distance"}

        with pytest.raises(ValueError, match="Feature keys cannot be shared"):
            _validate_feature_key_collisions(name_map, edge_name_map)

        with pytest.raises(ValueError, match="iou"):
            _validate_feature_key_collisions(name_map, edge_name_map)

    def test_multiple_collisions(self):
        """Test that multiple colliding keys are all reported."""
        name_map = {
            "time": "t",
            "x": "x",
            "y": "y",
            "iou": "node_iou",
            "weight": "node_weight",
        }
        edge_name_map = {"iou": "edge_iou", "weight": "edge_weight", "distance": "dist"}

        with pytest.raises(ValueError, match="Feature keys cannot be shared"):
            _validate_feature_key_collisions(name_map, edge_name_map)

        with pytest.raises(ValueError, match="iou"):
            _validate_feature_key_collisions(name_map, edge_name_map)

        with pytest.raises(ValueError, match="weight"):
            _validate_feature_key_collisions(name_map, edge_name_map)

    def test_none_edge_name_map(self):
        """Test that None edge_name_map doesn't raise."""
        name_map = {"time": "t", "x": "x", "y": "y", "iou": "iou"}
        edge_name_map = None

        # Should not raise
        _validate_feature_key_collisions(name_map, edge_name_map)


class TestValidateNameMap:
    """Test the main validate_name_map orchestrator function."""

    def test_valid_node_only(self):
        """Test validation with only node name_map."""
        name_map = {"time": "t", "x": "x", "y": "y"}
        importable_props = ["t", "x", "y", "area"]
        required_features = ["time"]
        position_attr = ["z", "y", "x"]
        ndim = 3

        # Should not raise
        validate_name_map(
            name_map, importable_props, required_features, position_attr, ndim
        )

    def test_valid_node_and_edge(self):
        """Test validation with both node and edge name_maps."""
        name_map = {"time": "t", "x": "x", "y": "y", "area": "area"}
        edge_name_map = {"iou": "iou", "distance": "dist"}
        importable_node_props = ["t", "x", "y", "area"]
        importable_edge_props = ["iou", "dist"]
        required_features = ["time"]
        position_attr = ["z", "y", "x"]
        ndim = 3

        # Should not raise
        validate_name_map(
            name_map,
            importable_node_props,
            required_features,
            position_attr,
            ndim,
            edge_name_map=edge_name_map,
            importable_edge_props=importable_edge_props,
        )

    def test_invalid_node_name_map(self):
        """Test that invalid node name_map raises ValueError."""
        name_map = {"time": "t", "x": "x"}  # Missing "y"
        importable_props = ["t", "x", "y"]
        required_features = ["time"]
        position_attr = ["z", "y", "x"]
        ndim = 3

        with pytest.raises(ValueError, match="None values"):
            validate_name_map(
                name_map, importable_props, required_features, position_attr, ndim
            )

    def test_invalid_edge_name_map(self):
        """Test that invalid edge name_map raises ValueError."""
        name_map = {"time": "t", "x": "x", "y": "y"}
        edge_name_map = {"iou": "iou", "distance": "dist"}
        importable_node_props = ["t", "x", "y"]
        importable_edge_props = ["iou"]  # "dist" doesn't exist
        required_features = ["time"]
        position_attr = ["z", "y", "x"]
        ndim = 3

        with pytest.raises(ValueError, match="non-existent properties"):
            validate_name_map(
                name_map,
                importable_node_props,
                required_features,
                position_attr,
                ndim,
                edge_name_map=edge_name_map,
                importable_edge_props=importable_edge_props,
            )

    def test_collision_between_node_and_edge(self):
        """Test that feature key collisions are detected."""
        name_map = {"time": "t", "x": "x", "y": "y", "iou": "node_iou"}
        edge_name_map = {"iou": "edge_iou"}
        importable_node_props = ["t", "x", "y", "node_iou"]
        importable_edge_props = ["edge_iou"]
        required_features = ["time"]
        position_attr = ["z", "y", "x"]
        ndim = 3

        with pytest.raises(ValueError, match="Feature keys cannot be shared"):
            validate_name_map(
                name_map,
                importable_node_props,
                required_features,
                position_attr,
                ndim,
                edge_name_map=edge_name_map,
                importable_edge_props=importable_edge_props,
            )
