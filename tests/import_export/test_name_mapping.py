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
    infer_name_map,
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
        display_name_to_key = {"Area": "area", "Circularity": "circularity"}
        mapping = {}

        remaining = _match_display_names_exact(
            available_props, display_name_to_key, mapping
        )

        assert mapping == {"area": "Area", "circularity": "Circularity"}
        assert remaining == ["time"]

    def test_no_matches(self):
        """Test when no properties match display names."""
        available_props = ["t", "x", "y"]
        display_name_to_key = {"Area": "area", "Circularity": "circularity"}
        mapping = {}

        remaining = _match_display_names_exact(
            available_props, display_name_to_key, mapping
        )

        assert mapping == {}
        assert remaining == ["t", "x", "y"]

    def test_empty_inputs(self):
        """Test with empty inputs."""
        mapping = {}
        remaining = _match_display_names_exact([], {}, mapping)
        assert mapping == {}
        assert remaining == []

    def test_case_sensitive(self):
        """Test that exact matching is case-sensitive."""
        available_props = ["area", "AREA"]
        display_name_to_key = {"Area": "area"}
        mapping = {}

        remaining = _match_display_names_exact(
            available_props, display_name_to_key, mapping
        )

        assert mapping == {}  # Neither "area" nor "AREA" matches "Area" exactly
        assert set(remaining) == {"area", "AREA"}


class TestMatchDisplayNamesFuzzy:
    """Test fuzzy matching between properties and feature display names."""

    def test_case_insensitive_match(self):
        """Test case-insensitive fuzzy matching."""
        available_props = ["area", "CIRC"]
        display_name_to_key = {"Area": "area", "Circularity": "circularity"}
        mapping = {}

        _ = _match_display_names_fuzzy(available_props, display_name_to_key, mapping)

        assert "area" in mapping
        assert "circularity" in mapping

    def test_abbreviation_match(self):
        """Test matching abbreviations to display names."""
        available_props = ["Circ", "Ecc"]
        display_name_to_key = {
            "Circularity": "circularity",
            "Eccentricity": "eccentricity",
        }
        mapping = {}

        _ = _match_display_names_fuzzy(available_props, display_name_to_key, mapping)

        assert "circularity" in mapping
        assert "eccentricity" in mapping

    def test_no_matches(self):
        """Test when no fuzzy matches found."""
        available_props = ["xyz", "abc"]
        display_name_to_key = {"Area": "area"}
        mapping = {}

        remaining = _match_display_names_fuzzy(
            available_props, display_name_to_key, mapping
        )

        assert mapping == {}
        assert set(remaining) == {"xyz", "abc"}

    def test_empty_available_props(self):
        """Test with empty available properties."""
        mapping = {}
        remaining = _match_display_names_fuzzy([], {"Area": "area"}, mapping)

        assert mapping == {}
        assert remaining == []

    def test_custom_cutoff(self):
        """Test with custom cutoff value."""
        available_props = ["Ar"]
        display_name_to_key = {"Area": "area"}

        # With low cutoff, should match
        mapping_low = {}
        _ = _match_display_names_fuzzy(
            available_props, display_name_to_key, mapping_low, cutoff=0.2
        )
        assert "area" in mapping_low

        # With high cutoff, should not match
        mapping_high = {}
        _ = _match_display_names_fuzzy(
            available_props, display_name_to_key, mapping_high, cutoff=0.9
        )
        assert mapping_high == {}


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
    """Test building list of standard fields to match."""

    def test_3d_data(self):
        """Test standard fields for 3D data (2D + time)."""
        required_features = ["time"]
        position_attr = ["z", "y", "x"]
        ndim = 3

        standard_fields = build_standard_fields(required_features, position_attr, ndim)

        assert "time" in standard_fields
        assert "y" in standard_fields
        assert "x" in standard_fields
        assert "z" not in standard_fields  # 3D means 2 spatial dims
        # Optional fields
        assert "seg_id" in standard_fields
        assert "track_id" in standard_fields
        assert "lineage_id" in standard_fields

    def test_4d_data(self):
        """Test standard fields for 4D data (3D + time)."""
        required_features = ["time"]
        position_attr = ["z", "y", "x"]
        ndim = 4

        standard_fields = build_standard_fields(required_features, position_attr, ndim)

        assert "time" in standard_fields
        assert "z" in standard_fields
        assert "y" in standard_fields
        assert "x" in standard_fields

    def test_multiple_required_features(self):
        """Test with multiple required features (e.g., CSV format)."""
        required_features = ["time", "id", "parent_id"]
        position_attr = ["z", "y", "x"]
        ndim = 3

        standard_fields = build_standard_fields(required_features, position_attr, ndim)

        assert "time" in standard_fields
        assert "id" in standard_fields
        assert "parent_id" in standard_fields


class TestBuildDisplayNameMapping:
    """Test building display name to feature key mapping."""

    def test_basic_mapping(self):
        """Test basic display name mapping."""
        available_computed_features = {
            "area": {"display_name": "Area", "other": "data"},
            "circularity": {"display_name": "Circularity"},
            "eccentricity": {"display_name": "Eccentricity"},
        }

        mapping = build_display_name_mapping(available_computed_features)

        assert mapping == {
            "Area": "area",
            "Circularity": "circularity",
            "Eccentricity": "eccentricity",
        }

    def test_skip_non_string_display_names(self):
        """Test that non-string display names (tuples, lists) are skipped."""
        available_computed_features = {
            "area": {"display_name": "Area"},
            "position": {"display_name": ("y", "x")},  # Tuple, should be skipped
            "color": {"display_name": ["r", "g", "b"]},  # List, should be skipped
        }

        mapping = build_display_name_mapping(available_computed_features)

        assert mapping == {"Area": "area"}

    def test_missing_display_name(self):
        """Test features without display_name are skipped."""
        available_computed_features = {
            "area": {"display_name": "Area"},
            "other": {},  # No display_name
        }

        mapping = build_display_name_mapping(available_computed_features)

        assert mapping == {"Area": "area"}

    def test_empty_input(self):
        """Test with empty features dict."""
        mapping = build_display_name_mapping({})
        assert mapping == {}


class TestInferNameMapIntegration:
    """Integration tests for the full infer_name_map pipeline."""

    def test_perfect_exact_matches(self):
        """Test when all fields have exact matches."""
        importable_props = ["time", "x", "y", "area", "circularity"]
        required_features = ["time"]
        position_attr = ["z", "y", "x"]
        ndim = 3
        available_computed_features = {
            "area": {"display_name": "Area"},
            "circularity": {"display_name": "Circularity"},
        }

        mapping = infer_name_map(
            importable_props,
            required_features,
            position_attr,
            ndim,
            available_computed_features,
        )

        assert mapping["time"] == "time"
        assert mapping["x"] == "x"
        assert mapping["y"] == "y"
        assert mapping["area"] == "area"
        assert mapping["circularity"] == "circularity"

    def test_fuzzy_matching_abbreviations(self):
        """Test fuzzy matching with abbreviations."""
        importable_props = ["t", "X", "Y", "Circ"]
        required_features = ["time"]
        position_attr = ["z", "y", "x"]
        ndim = 3
        available_computed_features = {
            "circularity": {"display_name": "Circularity"},
        }

        mapping = infer_name_map(
            importable_props,
            required_features,
            position_attr,
            ndim,
            available_computed_features,
        )

        # Should fuzzy match t->time, X->x, Y->y
        assert mapping["time"] == "t"
        assert mapping["x"] == "X"
        assert mapping["y"] == "Y"
        # Should fuzzy match Circ->circularity via display name
        assert mapping["circularity"] == "Circ"

    def test_custom_properties(self):
        """Test that unmatched properties map to themselves."""
        importable_props = ["time", "x", "y", "custom_col1", "custom_col2"]
        required_features = ["time"]
        position_attr = ["z", "y", "x"]
        ndim = 3
        available_computed_features = {}

        mapping = infer_name_map(
            importable_props,
            required_features,
            position_attr,
            ndim,
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
        position_attr = ["z", "y", "x"]
        ndim = 3
        available_computed_features = {
            "time_feature": {"display_name": "Time"},
        }

        mapping = infer_name_map(
            importable_props,
            required_features,
            position_attr,
            ndim,
            available_computed_features,
        )

        # "time" should match exactly to standard field "time"
        assert mapping["time"] == "time"
        # "Time" should fuzzy match to feature "time_feature"
        assert mapping["time_feature"] == "Time"

    def test_4d_data(self):
        """Test inference for 4D data (3D + time)."""
        importable_props = ["t", "z", "y", "x"]
        required_features = ["time"]
        position_attr = ["z", "y", "x"]
        ndim = 4
        available_computed_features = {}

        mapping = infer_name_map(
            importable_props,
            required_features,
            position_attr,
            ndim,
            available_computed_features,
        )

        # Should fuzzy match t->time
        assert mapping["time"] == "t"
        # Should exact match z, y, x
        assert mapping["z"] == "z"
        assert mapping["y"] == "y"
        assert mapping["x"] == "x"

    def test_optional_fields(self):
        """Test that optional fields (seg_id, track_id, lineage_id) are matched."""
        importable_props = ["time", "x", "y", "seg_id", "track_id"]
        required_features = ["time"]
        position_attr = ["z", "y", "x"]
        ndim = 3
        available_computed_features = {}

        mapping = infer_name_map(
            importable_props,
            required_features,
            position_attr,
            ndim,
            available_computed_features,
        )

        assert mapping["seg_id"] == "seg_id"
        assert mapping["track_id"] == "track_id"

    def test_csv_format_with_id_columns(self):
        """Test inference for CSV format with id and parent_id."""
        importable_props = ["t", "x", "y", "id", "parent_id", "Area"]
        required_features = ["time", "id", "parent_id"]
        position_attr = ["z", "y", "x"]
        ndim = 3
        available_computed_features = {
            "area": {"display_name": "Area"},
        }

        mapping = infer_name_map(
            importable_props,
            required_features,
            position_attr,
            ndim,
            available_computed_features,
        )

        # Should fuzzy match t->time
        assert mapping["time"] == "t"
        # Should exact match id, parent_id
        assert mapping["id"] == "id"
        assert mapping["parent_id"] == "parent_id"
        # Should exact match Area via display name
        assert mapping["area"] == "Area"
