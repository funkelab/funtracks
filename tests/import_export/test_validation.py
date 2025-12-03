"""Tests for validation functions in funtracks.import_export._validation."""

import pytest

from funtracks.import_export._validation import (
    validate_edge_name_map,
    validate_feature_key_collisions,
    validate_node_name_map,
)


class TestValidateNodeNameMap:
    """Test validate_node_name_map helper function."""

    def test_valid_node_name_map(self):
        """Test that a valid node name_map passes validation."""
        name_map = {"time": "t", "x": "x_coord", "y": "y_coord"}
        importable_props = ["t", "x_coord", "y_coord", "area"]
        required_features = ["time"]
        position_attr = ["z", "y", "x"]
        ndim = 3  # 2D + time

        # Should not raise
        validate_node_name_map(
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
            validate_node_name_map(
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
            validate_node_name_map(
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
            validate_node_name_map(
                name_map, importable_props, required_features, position_attr, ndim
            )

    def test_duplicate_values_in_required_fields(self):
        """Test that duplicate values in required fields are allowed.

        Multiple standard keys can map to the same source property.
        This is useful for cases like seg_id mapping to the same column as id.
        """
        name_map = {"time": "t", "x": "coord", "y": "coord"}  # Duplicate "coord"
        importable_props = ["t", "coord"]
        required_features = ["time"]
        position_attr = ["z", "y", "x"]
        ndim = 3

        # Should not raise - duplicates are allowed
        validate_node_name_map(
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
            validate_node_name_map(
                name_map, importable_props, required_features, position_attr, ndim
            )


class TestValidateEdgeNameMap:
    """Test validate_edge_name_map helper function."""

    def test_valid_edge_name_map(self):
        """Test that a valid edge name_map passes validation."""
        edge_name_map = {"iou": "overlap", "distance": "dist"}
        importable_props = ["overlap", "dist", "weight"]

        # Should not raise
        validate_edge_name_map(edge_name_map, importable_props)

    def test_nonexistent_edge_property(self):
        """Test that mapping to non-existent edge properties raises ValueError."""
        edge_name_map = {"iou": "overlap", "distance": "dist"}
        importable_props = ["overlap"]  # "dist" doesn't exist

        with pytest.raises(ValueError, match="non-existent properties"):
            validate_edge_name_map(edge_name_map, importable_props)

    def test_empty_importable_props(self):
        """Test that empty importable_props list doesn't raise."""
        edge_name_map = {"iou": "overlap"}
        importable_props = []

        # Should not raise when importable_props is empty
        validate_edge_name_map(edge_name_map, importable_props)


class TestValidateFeatureKeyCollisions:
    """Test validate_feature_key_collisions helper function."""

    def test_no_collision(self):
        """Test that non-overlapping keys pass validation."""
        name_map = {"time": "t", "x": "x", "y": "y", "area": "area"}
        edge_name_map = {"iou": "iou", "distance": "distance"}

        # Should not raise
        validate_feature_key_collisions(name_map, edge_name_map)

    def test_single_collision(self):
        """Test that a single colliding key raises ValueError."""
        name_map = {"time": "t", "x": "x", "y": "y", "iou": "node_iou"}
        edge_name_map = {"iou": "edge_iou", "distance": "distance"}

        with pytest.raises(ValueError, match="Feature keys cannot be shared"):
            validate_feature_key_collisions(name_map, edge_name_map)

        with pytest.raises(ValueError, match="iou"):
            validate_feature_key_collisions(name_map, edge_name_map)

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
            validate_feature_key_collisions(name_map, edge_name_map)

        with pytest.raises(ValueError, match="iou"):
            validate_feature_key_collisions(name_map, edge_name_map)

        with pytest.raises(ValueError, match="weight"):
            validate_feature_key_collisions(name_map, edge_name_map)

    def test_none_edge_name_map(self):
        """Test that None edge_name_map doesn't raise."""
        name_map = {"time": "t", "x": "x", "y": "y", "iou": "iou"}
        edge_name_map = None

        # Should not raise
        validate_feature_key_collisions(name_map, edge_name_map)
