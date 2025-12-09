"""Unit tests for name mapping helper functions."""

from __future__ import annotations

from funtracks.import_export._name_mapping import (
    _map_remaining_to_self,
    _match_display_names_exact,
    _match_display_names_fuzzy,
    _match_exact,
    _match_fuzzy,
    build_display_name_mapping,
    build_standard_fields,
    infer_edge_name_map,
    infer_node_name_map,
)


class TestMatchExact:
    """Test exact matching between target fields and available properties."""

    def test_perfect_match(self):
        """Test when all target fields have exact matches."""
        target_fields = ["time", "x", "y"]
        available_props = ["time", "x", "y", "area"]
        mapping = {}

        remaining = _match_exact(target_fields, available_props, mapping)

        assert mapping == {"time": "time", "x": "x", "y": "y"}
        assert remaining == ["area"]

    def test_partial_match(self):
        """Test when only some target fields have exact matches."""
        target_fields = ["time", "x", "y", "z"]
        available_props = ["time", "x", "area"]
        mapping = {}

        remaining = _match_exact(target_fields, available_props, mapping)

        assert mapping == {"time": "time", "x": "x"}
        assert remaining == ["area"]

    def test_no_matches(self):
        """Test when no target fields have exact matches."""
        target_fields = ["time", "x", "y"]
        available_props = ["t", "X", "Y"]
        mapping = {}

        remaining = _match_exact(target_fields, available_props, mapping)

        assert mapping == {}
        assert remaining == ["t", "X", "Y"]

    def test_empty_inputs(self):
        """Test with empty inputs."""
        mapping = {}
        remaining = _match_exact([], [], mapping)
        assert mapping == {}
        assert remaining == []

    def test_skip_existing_mapping(self):
        """Test that fields already in existing_mapping are skipped."""
        target_fields = ["time", "x", "y"]
        available_props = ["time", "x", "y"]
        mapping = {"time": "t"}  # time already mapped

        remaining = _match_exact(target_fields, available_props, mapping)

        assert mapping == {"time": "t", "x": "x", "y": "y"}
        assert "time" in remaining  # time should not be consumed


class TestMatchFuzzy:
    """Test fuzzy matching between target fields and available properties."""

    def test_case_insensitive_match(self):
        """Test case-insensitive fuzzy matching."""
        target_fields = ["time", "x", "y"]
        available_props = ["Time", "X", "Y"]
        mapping = {}

        remaining = _match_fuzzy(target_fields, available_props, mapping)

        assert mapping == {"time": "Time", "x": "X", "y": "Y"}
        assert remaining == []

    def test_abbreviation_match(self):
        """Test matching abbreviations (e.g., 't' matches 'time')."""
        target_fields = ["time"]
        available_props = ["t"]
        mapping = {}

        _ = _match_fuzzy(target_fields, available_props, mapping)

        # 't' should match 'time' (above 40% similarity)
        assert "time" in mapping
        assert mapping["time"] == "t"

    def test_cutoff_threshold(self):
        """Test that matches below cutoff are not returned."""
        target_fields = ["time"]
        available_props = ["abc"]  # Very dissimilar
        mapping = {}

        remaining = _match_fuzzy(target_fields, available_props, mapping, cutoff=0.4)

        assert mapping == {}
        assert remaining == ["abc"]

    def test_custom_cutoff(self):
        """Test with custom cutoff value."""
        target_fields = ["time"]
        available_props = ["ti"]

        # With low cutoff, should match
        mapping_low = {}
        _ = _match_fuzzy(target_fields, available_props, mapping_low, cutoff=0.2)
        assert "time" in mapping_low

        # With high cutoff, should not match
        mapping_high = {}
        _ = _match_fuzzy(target_fields, available_props, mapping_high, cutoff=0.9)
        assert mapping_high == {}

    def test_skip_existing_mapping(self):
        """Test that fields already mapped are skipped."""
        target_fields = ["time", "x"]
        available_props = ["t", "X"]
        mapping = {"time": "t"}

        _ = _match_fuzzy(target_fields, available_props, mapping)

        assert mapping["time"] == "t"  # Should remain unchanged
        assert "x" in mapping

    def test_empty_available_props(self):
        """Test with no available properties."""
        target_fields = ["time", "x", "y"]
        available_props = []
        mapping = {}

        remaining = _match_fuzzy(target_fields, available_props, mapping)

        assert mapping == {}
        assert remaining == []


class TestMatchDisplayNamesExact:
    """Test exact matching between properties and feature display names."""

    def test_exact_display_name_match(self):
        """Test exact matching with display names."""
        available_props = ["Area", "Circularity", "time"]
        display_name_to_key = {
            "Area": ("area", 0),
            "Circularity": ("circularity", 0),
        }
        mapping: dict = {}

        remaining = _match_display_names_exact(
            available_props, display_name_to_key, mapping
        )

        assert mapping == {"area": "Area", "circularity": "Circularity"}
        assert remaining == ["time"]

    def test_no_matches(self):
        """Test when no properties match display names."""
        available_props = ["t", "x", "y"]
        display_name_to_key = {
            "Area": ("area", 0),
            "Circularity": ("circularity", 0),
        }
        mapping: dict = {}

        remaining = _match_display_names_exact(
            available_props, display_name_to_key, mapping
        )

        assert mapping == {}
        assert remaining == ["t", "x", "y"]

    def test_empty_inputs(self):
        """Test with empty inputs."""
        mapping: dict = {}
        remaining = _match_display_names_exact([], {}, mapping)
        assert mapping == {}
        assert remaining == []

    def test_case_sensitive(self):
        """Test that exact matching is case-sensitive."""
        available_props = ["area", "AREA"]
        display_name_to_key = {"Area": ("area", 0)}
        mapping: dict = {}

        remaining = _match_display_names_exact(
            available_props, display_name_to_key, mapping
        )

        assert mapping == {}  # Neither "area" nor "AREA" matches "Area" exactly
        assert set(remaining) == {"area", "AREA"}

    def test_multi_value_feature(self):
        """Test matching multi-value features by value_names."""
        available_props = ["major_axis", "minor_axis", "Area"]
        display_name_to_key = {
            "Area": ("area", 0),
            "major_axis": ("ellipsoid_axes", 0),
            "semi_minor_axis": ("ellipsoid_axes", 1),
            "minor_axis": ("ellipsoid_axes", 2),
        }
        mapping: dict = {}

        remaining = _match_display_names_exact(
            available_props, display_name_to_key, mapping
        )

        assert mapping == {
            "area": "Area",
            "ellipsoid_axes": ["major_axis", "minor_axis"],  # sorted by index
        }
        assert remaining == []


class TestMatchDisplayNamesFuzzy:
    """Test fuzzy matching between properties and feature display names."""

    def test_case_insensitive_match(self):
        """Test case-insensitive fuzzy matching."""
        available_props = ["area", "CIRC"]
        display_name_to_key = {
            "Area": ("area", 0),
            "Circularity": ("circularity", 0),
        }
        mapping: dict = {}

        _ = _match_display_names_fuzzy(available_props, display_name_to_key, mapping)

        assert "area" in mapping
        assert "circularity" in mapping

    def test_abbreviation_match(self):
        """Test matching abbreviations to display names."""
        available_props = ["Circ", "Ecc"]
        display_name_to_key = {
            "Circularity": ("circularity", 0),
            "Eccentricity": ("eccentricity", 0),
        }
        mapping: dict = {}

        _ = _match_display_names_fuzzy(available_props, display_name_to_key, mapping)

        assert "circularity" in mapping
        assert "eccentricity" in mapping

    def test_no_matches(self):
        """Test when no fuzzy matches found."""
        available_props = ["xyz", "abc"]
        display_name_to_key = {"Area": ("area", 0)}
        mapping: dict = {}

        remaining = _match_display_names_fuzzy(
            available_props, display_name_to_key, mapping
        )

        assert mapping == {}
        assert set(remaining) == {"xyz", "abc"}

    def test_empty_available_props(self):
        """Test with empty available properties."""
        mapping: dict = {}
        remaining = _match_display_names_fuzzy([], {"Area": ("area", 0)}, mapping)

        assert mapping == {}
        assert remaining == []

    def test_custom_cutoff(self):
        """Test with custom cutoff value."""
        available_props = ["Ar"]
        display_name_to_key = {"Area": ("area", 0)}

        # With low cutoff, should match
        mapping_low: dict = {}
        _ = _match_display_names_fuzzy(
            available_props, display_name_to_key, mapping_low, cutoff=0.2
        )
        assert "area" in mapping_low

        # With high cutoff, should not match
        mapping_high: dict = {}
        _ = _match_display_names_fuzzy(
            available_props, display_name_to_key, mapping_high, cutoff=0.9
        )
        assert mapping_high == {}

    def test_multi_value_feature(self):
        """Test fuzzy matching multi-value features by value_names."""
        available_props = ["Major_Axis", "Minor_Axis", "area"]
        display_name_to_key = {
            "Area": ("area", 0),
            "major_axis": ("ellipsoid_axes", 0),
            "semi_minor_axis": ("ellipsoid_axes", 1),
            "minor_axis": ("ellipsoid_axes", 2),
        }
        mapping: dict = {}

        remaining = _match_display_names_fuzzy(
            available_props, display_name_to_key, mapping
        )

        assert mapping == {
            "area": "area",
            "ellipsoid_axes": ["Major_Axis", "Minor_Axis"],  # sorted by index
        }
        assert remaining == []


class TestMapRemainingToSelf:
    """Test identity mapping for remaining properties."""

    def test_basic_mapping(self):
        """Test basic identity mapping."""
        remaining_props = ["custom_col1", "custom_col2", "feature_x"]

        mapping = _map_remaining_to_self(remaining_props)

        assert mapping == {
            "custom_col1": "custom_col1",
            "custom_col2": "custom_col2",
            "feature_x": "feature_x",
        }

    def test_empty_input(self):
        """Test with empty input."""
        mapping = _map_remaining_to_self([])
        assert mapping == {}

    def test_single_property(self):
        """Test with single property."""
        mapping = _map_remaining_to_self(["prop"])
        assert mapping == {"prop": "prop"}


class TestBuildStandardFields:
    """Test building list of standard fields to match.

    Position attributes (z, y, x) are NOT included - they are matched via
    Position feature's value_names to create composite "pos" mapping.
    """

    def test_basic_required_features(self):
        """Test standard fields include required features and seg_id."""
        required_features = ["time"]

        standard_fields = build_standard_fields(required_features)

        assert "time" in standard_fields
        # Optional fields
        assert "seg_id" in standard_fields
        # Position attrs should NOT be in standard fields
        assert "z" not in standard_fields
        assert "y" not in standard_fields
        assert "x" not in standard_fields

    def test_multiple_required_features(self):
        """Test with multiple required features (e.g., CSV format)."""
        required_features = ["time", "id", "parent_id"]

        standard_fields = build_standard_fields(required_features)

        assert "time" in standard_fields
        assert "id" in standard_fields
        assert "parent_id" in standard_fields
        assert "seg_id" in standard_fields


class TestBuildDisplayNameMapping:
    """Test building display name to feature key mapping."""

    def test_basic_mapping(self):
        """Test basic display name mapping for single-value features."""
        available_computed_features = {
            "area": {"display_name": "Area", "num_values": 1},
            "circularity": {"display_name": "Circularity", "num_values": 1},
            "eccentricity": {"display_name": "Eccentricity", "num_values": 1},
        }

        mapping = build_display_name_mapping(available_computed_features)

        assert mapping == {
            "Area": ("area", 0),
            "Circularity": ("circularity", 0),
            "Eccentricity": ("eccentricity", 0),
        }

    def test_multi_value_features_use_value_names(self):
        """Test that multi-value features map each value_name to (key, index)."""
        available_computed_features = {
            "area": {"display_name": "Area", "num_values": 1},
            "position": {
                "display_name": "Position",
                "num_values": 2,
                "value_names": ["y", "x"],
            },
            "ellipsoid_axes": {
                "display_name": "Ellipsoid axis radii",
                "num_values": 3,
                "value_names": ["major", "semi_minor", "minor"],
            },
        }

        mapping = build_display_name_mapping(available_computed_features)

        assert mapping == {
            "Area": ("area", 0),
            "y": ("position", 0),
            "x": ("position", 1),
            "major": ("ellipsoid_axes", 0),
            "semi_minor": ("ellipsoid_axes", 1),
            "minor": ("ellipsoid_axes", 2),
        }

    def test_missing_display_name(self):
        """Test single-value features without display_name are skipped."""
        available_computed_features = {
            "area": {"display_name": "Area", "num_values": 1},
            "other": {"num_values": 1},  # No display_name
        }

        mapping = build_display_name_mapping(available_computed_features)

        assert mapping == {"Area": ("area", 0)}

    def test_empty_input(self):
        """Test with empty features dict."""
        mapping = build_display_name_mapping({})
        assert mapping == {}


class TestInferNodeNameMapIntegration:
    """Integration tests for the full infer_node_name_map pipeline.

    Position columns (z, y, x) are matched via Position feature's value_names,
    producing composite mapping like {"pos": ["y", "x"]} or {"pos": ["z", "y", "x"]}.
    """

    def test_perfect_exact_matches_2d(self):
        """Test when all fields have exact matches (2D data)."""
        importable_props = ["time", "x", "y", "area", "circularity"]
        required_features = ["time"]
        available_computed_features = {
            "area": {"feature_type": "node", "display_name": "Area", "num_values": 1},
            "circularity": {
                "feature_type": "node",
                "display_name": "Circularity",
                "num_values": 1,
            },
            "pos": {
                "feature_type": "node",
                "display_name": "Position",
                "num_values": 2,
                "value_names": ["y", "x"],
            },
        }

        mapping = infer_node_name_map(
            importable_props,
            required_features,
            available_computed_features,
        )

        assert mapping["time"] == "time"
        # Position should be composite
        assert mapping["pos"] == ["y", "x"]
        assert mapping["area"] == "area"
        assert mapping["circularity"] == "circularity"

    def test_fuzzy_matching_abbreviations(self):
        """Test fuzzy matching with abbreviations."""
        importable_props = ["t", "X", "Y", "Circ"]
        required_features = ["time"]
        available_computed_features = {
            "circularity": {
                "feature_type": "node",
                "display_name": "Circularity",
                "num_values": 1,
            },
            "pos": {
                "feature_type": "node",
                "display_name": "Position",
                "num_values": 2,
                "value_names": ["y", "x"],
            },
        }

        mapping = infer_node_name_map(
            importable_props,
            required_features,
            available_computed_features,
        )

        # Should fuzzy match t->time
        assert mapping["time"] == "t"
        # Should fuzzy match X->x, Y->y via Position value_names
        assert mapping["pos"] == ["Y", "X"]
        # Should fuzzy match Circ->circularity via display name
        assert mapping["circularity"] == "Circ"

    def test_custom_properties(self):
        """Test that unmatched properties map to themselves."""
        importable_props = ["time", "x", "y", "custom_col1", "custom_col2"]
        required_features = ["time"]
        available_computed_features = {
            "pos": {
                "feature_type": "node",
                "display_name": "Position",
                "num_values": 2,
                "value_names": ["y", "x"],
            },
        }

        mapping = infer_node_name_map(
            importable_props,
            required_features,
            available_computed_features,
        )

        # Custom properties should map to themselves
        assert mapping["custom_col1"] == "custom_col1"
        assert mapping["custom_col2"] == "custom_col2"

    def test_priority_order(self):
        """Test that matching happens in correct priority order."""
        # Exact standard match should take priority over fuzzy feature match
        importable_props = ["time", "Time", "x", "y"]
        required_features = ["time"]
        available_computed_features = {
            "time_feature": {
                "feature_type": "node",
                "display_name": "Time",
                "num_values": 1,
            },
            "pos": {
                "feature_type": "node",
                "display_name": "Position",
                "num_values": 2,
                "value_names": ["y", "x"],
            },
        }

        mapping = infer_node_name_map(
            importable_props,
            required_features,
            available_computed_features,
        )

        # "time" should match exactly to standard field "time"
        assert mapping["time"] == "time"
        # "Time" should fuzzy match to feature "time_feature"
        assert mapping["time_feature"] == "Time"

    def test_3d_position(self):
        """Test inference for 3D position (z, y, x)."""
        importable_props = ["t", "z", "y", "x"]
        required_features = ["time"]
        available_computed_features = {
            "pos": {
                "feature_type": "node",
                "display_name": "Position",
                "num_values": 3,
                "value_names": ["z", "y", "x"],
            },
        }

        mapping = infer_node_name_map(
            importable_props,
            required_features,
            available_computed_features,
        )

        # Should fuzzy match t->time
        assert mapping["time"] == "t"
        # Should exact match z, y, x via Position value_names -> composite pos
        assert mapping["pos"] == ["z", "y", "x"]

    def test_optional_fields(self):
        """Test that optional fields (seg_id, track_id) are matched."""
        importable_props = ["time", "x", "y", "seg_id", "track_id"]
        required_features = ["time"]
        available_computed_features = {
            "pos": {
                "feature_type": "node",
                "display_name": "Position",
                "num_values": 2,
                "value_names": ["y", "x"],
            },
        }

        mapping = infer_node_name_map(
            importable_props,
            required_features,
            available_computed_features,
        )

        assert mapping["seg_id"] == "seg_id"
        assert mapping["track_id"] == "track_id"

    def test_csv_format_with_id_columns(self):
        """Test inference for CSV format with id and parent_id."""
        importable_props = ["t", "x", "y", "id", "parent_id", "Area"]
        required_features = ["time", "id", "parent_id"]
        available_computed_features = {
            "area": {"feature_type": "node", "display_name": "Area", "num_values": 1},
            "pos": {
                "feature_type": "node",
                "display_name": "Position",
                "num_values": 2,
                "value_names": ["y", "x"],
            },
        }

        mapping = infer_node_name_map(
            importable_props,
            required_features,
            available_computed_features,
        )

        # Should fuzzy match t->time
        assert mapping["time"] == "t"
        # Should exact match id, parent_id
        assert mapping["id"] == "id"
        assert mapping["parent_id"] == "parent_id"
        # Should exact match Area via display name
        assert mapping["area"] == "Area"
        # Position should be composite
        assert mapping["pos"] == ["y", "x"]


class TestInferEdgeNameMapIntegration:
    """Integration tests for the full infer_edge_name_map pipeline."""

    def test_perfect_exact_matches(self):
        """Test when all edge properties have exact matches."""
        importable_props = ["iou", "distance", "custom_edge_prop"]
        available_computed_features = {
            "iou": {"feature_type": "edge", "display_name": "IOU"},
            "distance": {"feature_type": "edge", "display_name": "Distance"},
            "area": {"feature_type": "node", "display_name": "Area"},  # Should be ignored
        }

        mapping = infer_edge_name_map(importable_props, available_computed_features)

        assert mapping["iou"] == "iou"
        assert mapping["distance"] == "distance"
        assert mapping["custom_edge_prop"] == "custom_edge_prop"

    def test_fuzzy_matching_abbreviations(self):
        """Test fuzzy matching with abbreviations."""
        importable_props = ["IOU", "dist"]
        available_computed_features = {
            "iou": {"feature_type": "edge", "display_name": "IOU"},
            "distance": {"feature_type": "edge", "display_name": "Distance"},
        }

        mapping = infer_edge_name_map(importable_props, available_computed_features)

        # Should fuzzy match IOU->iou, dist->distance
        assert mapping["iou"] == "IOU"
        assert mapping["distance"] == "dist"

    def test_custom_properties(self):
        """Test that unmatched edge properties map to themselves."""
        importable_props = ["custom_edge1", "custom_edge2", "iou"]
        available_computed_features = {
            "iou": {"feature_type": "edge", "display_name": "IOU"},
        }

        mapping = infer_edge_name_map(importable_props, available_computed_features)

        # Custom properties should map to themselves
        assert mapping["custom_edge1"] == "custom_edge1"
        assert mapping["custom_edge2"] == "custom_edge2"
        # Standard property should exact match
        assert mapping["iou"] == "iou"

    def test_empty_input(self):
        """Test with empty edge properties list."""
        mapping = infer_edge_name_map([])
        assert mapping == {}

    def test_with_edge_features_dict(self):
        """Test inference with edge feature display names."""
        importable_props = ["Overlap", "Dist", "custom"]
        available_computed_features = {
            "iou": {"feature_type": "edge", "display_name": "Overlap"},
            "distance": {"feature_type": "edge", "display_name": "Distance"},
            "area": {"feature_type": "node", "display_name": "Area"},  # Should be ignored
        }

        mapping = infer_edge_name_map(importable_props, available_computed_features)

        # Should match via display names
        assert mapping["iou"] == "Overlap"
        # Dist should fuzzy match to "distance" edge feature key
        assert mapping["distance"] == "Dist"
        # Custom should map to itself
        assert mapping["custom"] == "custom"

    def test_without_edge_features_dict(self):
        """Test inference without edge feature display names (None)."""
        importable_props = ["iou", "distance", "custom"]

        mapping = infer_edge_name_map(importable_props, available_computed_features=None)

        # Without feature dict, everything maps to itself
        assert mapping["iou"] == "iou"
        assert mapping["distance"] == "distance"
        assert mapping["custom"] == "custom"

    def test_case_insensitive_matching(self):
        """Test that matching is case-insensitive."""
        importable_props = ["IOU", "Distance"]
        available_computed_features = {
            "iou": {"feature_type": "edge", "display_name": "IOU"},
            "distance": {"feature_type": "edge", "display_name": "Distance"},
        }

        mapping = infer_edge_name_map(importable_props, available_computed_features)

        # Should fuzzy match despite case differences
        assert mapping["iou"] == "IOU"
        assert mapping["distance"] == "Distance"
