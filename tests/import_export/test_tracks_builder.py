"""Tests for TracksBuilder internals."""

import numpy as np

from funtracks.import_export.csv._import import CSVTracksBuilder


class TestCombineMultiValueProps:
    """Test _combine_props_from_name_map method."""

    def test_combine_missing_arrays_with_or(self):
        """Test that missing arrays are combined with OR logic.

        If any component column has a missing value for a row, the combined
        property should also be marked as missing for that row.
        """
        # Create a builder and set up minimal state
        builder = CSVTracksBuilder()
        builder.node_name_map = {"pos": ["y", "x"]}

        # Create in_memory_geff with missing arrays
        # y is missing for nodes 0, 2 (indices where missing_y is True)
        # x is missing for nodes 1, 2 (indices where missing_x is True)
        # Combined should be missing for nodes 0, 1, 2
        builder.in_memory_geff = {
            "metadata": None,
            "node_ids": np.array([0, 1, 2, 3]),
            "edge_ids": np.array([]).reshape(0, 2),
            "node_props": {
                "y": {
                    "values": np.array([1.0, 2.0, 3.0, 4.0]),
                    "missing": np.array([True, False, True, False]),
                },
                "x": {
                    "values": np.array([10.0, 20.0, 30.0, 40.0]),
                    "missing": np.array([False, True, True, False]),
                },
            },
            "edge_props": {},
        }

        # Call the method
        builder._combine_multi_value_props(
            builder.in_memory_geff["node_props"], builder.node_name_map
        )

        # Check that pos was created with combined values
        assert "pos" in builder.in_memory_geff["node_props"]
        pos_prop = builder.in_memory_geff["node_props"]["pos"]

        # Check values are stacked correctly
        expected_values = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])
        np.testing.assert_array_equal(pos_prop["values"], expected_values)

        # Check missing is OR of component missing arrays
        # Node 0: y missing (T) OR x missing (F) = T
        # Node 1: y missing (F) OR x missing (T) = T
        # Node 2: y missing (T) OR x missing (T) = T
        # Node 3: y missing (F) OR x missing (F) = F
        expected_missing = np.array([True, True, True, False])
        np.testing.assert_array_equal(pos_prop["missing"], expected_missing)

        # Check individual columns were removed
        assert "y" not in builder.in_memory_geff["node_props"]
        assert "x" not in builder.in_memory_geff["node_props"]

    def test_combine_no_missing_arrays(self):
        """Test combining when no component has missing arrays."""
        builder = CSVTracksBuilder()
        builder.node_name_map = {"pos": ["y", "x"]}

        builder.in_memory_geff = {
            "metadata": None,
            "node_ids": np.array([0, 1, 2]),
            "edge_ids": np.array([]).reshape(0, 2),
            "node_props": {
                "y": {"values": np.array([1.0, 2.0, 3.0]), "missing": None},
                "x": {"values": np.array([10.0, 20.0, 30.0]), "missing": None},
            },
            "edge_props": {},
        }

        builder._combine_multi_value_props(
            builder.in_memory_geff["node_props"], builder.node_name_map
        )

        pos_prop = builder.in_memory_geff["node_props"]["pos"]
        assert pos_prop["missing"] is None

    def test_combine_partial_missing_arrays(self):
        """Test combining when only some components have missing arrays."""
        builder = CSVTracksBuilder()
        builder.node_name_map = {"pos": ["y", "x"]}

        # Only y has a missing array, x does not
        builder.in_memory_geff = {
            "metadata": None,
            "node_ids": np.array([0, 1, 2]),
            "edge_ids": np.array([]).reshape(0, 2),
            "node_props": {
                "y": {
                    "values": np.array([1.0, 2.0, 3.0]),
                    "missing": np.array([True, False, False]),
                },
                "x": {"values": np.array([10.0, 20.0, 30.0]), "missing": None},
            },
            "edge_props": {},
        }

        builder._combine_multi_value_props(
            builder.in_memory_geff["node_props"], builder.node_name_map
        )

        pos_prop = builder.in_memory_geff["node_props"]["pos"]
        # Should use the missing array from y (the only one with missing)
        expected_missing = np.array([True, False, False])
        np.testing.assert_array_equal(pos_prop["missing"], expected_missing)

    def test_combine_edge_props_with_missing(self):
        """Test combining edge properties with missing arrays."""
        builder = CSVTracksBuilder()
        builder.edge_name_map = {"multi_edge_feat": ["a", "b"]}

        builder.in_memory_geff = {
            "metadata": None,
            "node_ids": np.array([0, 1, 2]),
            "edge_ids": np.array([[0, 1], [1, 2]]),
            "node_props": {},
            "edge_props": {
                "a": {
                    "values": np.array([1.0, 2.0]),
                    "missing": np.array([True, False]),
                },
                "b": {
                    "values": np.array([10.0, 20.0]),
                    "missing": np.array([False, True]),
                },
            },
        }

        builder._combine_multi_value_props(
            builder.in_memory_geff["edge_props"], builder.edge_name_map
        )

        edge_prop = builder.in_memory_geff["edge_props"]["multi_edge_feat"]
        expected_values = np.array([[1.0, 10.0], [2.0, 20.0]])
        np.testing.assert_array_equal(edge_prop["values"], expected_values)

        # Edge 0: a missing (T) OR b missing (F) = T
        # Edge 1: a missing (F) OR b missing (T) = T
        expected_missing = np.array([True, True])
        np.testing.assert_array_equal(edge_prop["missing"], expected_missing)


class TestPreprocessNameMap:
    """Test _preprocess_name_map method."""

    def test_removes_none_values(self):
        """Test that None values are removed from name maps."""
        builder = CSVTracksBuilder()
        builder.node_name_map = {
            "time": "t",
            "pos": ["y", "x"],
            "optional_feat": None,
        }
        builder.edge_name_map = {"iou": "overlap", "unused": None}

        builder._preprocess_name_map()

        assert "optional_feat" not in builder.node_name_map
        assert "unused" not in builder.edge_name_map
        assert builder.node_name_map == {"time": "t", "pos": ["y", "x"]}
        assert builder.edge_name_map == {"iou": "overlap"}

    def test_removes_empty_lists(self):
        """Test that empty lists are removed from name maps."""
        builder = CSVTracksBuilder()
        builder.node_name_map = {
            "time": "t",
            "pos": ["y", "x"],
            "empty_feat": [],
        }
        builder.edge_name_map = None

        builder._preprocess_name_map()

        assert "empty_feat" not in builder.node_name_map
