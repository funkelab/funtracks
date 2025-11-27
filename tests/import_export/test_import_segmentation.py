"""Tests for _import_segmentation module."""

import networkx as nx
import numpy as np
import tifffile

from funtracks.import_export._import_segmentation import (
    load_segmentation,
    relabel_segmentation,
)


class TestLoadSegmentation:
    """Tests for load_segmentation function."""

    def test_load_from_path(self, tmp_path):
        """Test loading segmentation from a tif file."""
        # Create test segmentation
        seg = np.array([[[1, 0], [0, 2]], [[3, 0], [0, 4]]], dtype=np.uint16)
        seg_path = tmp_path / "test_seg.tif"
        tifffile.imwrite(seg_path, seg)

        # Load and verify
        result = load_segmentation(seg_path)
        np.testing.assert_array_equal(result.compute(), seg)

    def test_load_from_array(self):
        """Test wrapping a numpy array in dask."""
        seg = np.array([[[1, 0], [0, 2]], [[3, 0], [0, 4]]], dtype=np.uint16)

        result = load_segmentation(seg)

        # Should be a dask array
        assert hasattr(result, "compute")
        np.testing.assert_array_equal(result.compute(), seg)


class TestRelabelSegmentation:
    """Tests for relabel_segmentation function."""

    def test_basic_relabeling(self):
        """Test basic seg_id to node_id relabeling."""
        # Create segmentation with seg_ids 10, 20
        seg = np.zeros((2, 5, 5), dtype=np.uint16)
        seg[0, 1, 1] = 10  # seg_id 10 at t=0
        seg[1, 2, 2] = 20  # seg_id 20 at t=1

        # Create graph with node_ids 1, 2
        graph = nx.DiGraph()
        graph.add_node(1)
        graph.add_node(2)

        node_ids = np.array([1, 2])
        seg_ids = np.array([10, 20])
        time_values = np.array([0, 1])

        result = relabel_segmentation(seg, graph, node_ids, seg_ids, time_values)

        # seg_id 10 -> node_id 1, seg_id 20 -> node_id 2
        assert result[0, 1, 1] == 1
        assert result[1, 2, 2] == 2
        # Background should remain 0
        assert result[0, 0, 0] == 0

    def test_relabeling_with_node_id_zero(self):
        """Test that node_id 0 is handled by offsetting all IDs."""
        # Create segmentation with seg_ids 10, 20
        seg = np.zeros((2, 5, 5), dtype=np.uint16)
        seg[0, 1, 1] = 10  # seg_id 10 at t=0
        seg[1, 2, 2] = 20  # seg_id 20 at t=1

        # Create graph with node_ids 0, 1 (includes 0!)
        graph = nx.DiGraph()
        graph.add_node(0)
        graph.add_node(1)

        node_ids = np.array([0, 1])
        seg_ids = np.array([10, 20])
        time_values = np.array([0, 1])

        result = relabel_segmentation(seg, graph, node_ids, seg_ids, time_values)

        # node_ids should be offset by 1: 0->1, 1->2
        # seg_id 10 -> node_id 1 (was 0), seg_id 20 -> node_id 2 (was 1)
        assert result[0, 1, 1] == 1
        assert result[1, 2, 2] == 2

        # Graph should also be relabeled
        assert 1 in graph.nodes()
        assert 2 in graph.nodes()
        assert 0 not in graph.nodes()

    def test_no_relabeling_needed_same_ids(self):
        """Test when seg_ids equal node_ids (relabeling still applies mapping)."""
        # Create segmentation with seg_ids 1, 2
        seg = np.zeros((2, 5, 5), dtype=np.uint16)
        seg[0, 1, 1] = 1
        seg[1, 2, 2] = 2

        graph = nx.DiGraph()
        graph.add_node(1)
        graph.add_node(2)

        node_ids = np.array([1, 2])
        seg_ids = np.array([1, 2])  # Same as node_ids
        time_values = np.array([0, 1])

        result = relabel_segmentation(seg, graph, node_ids, seg_ids, time_values)

        # Should still produce valid output (identity mapping)
        assert result[0, 1, 1] == 1
        assert result[1, 2, 2] == 2

    def test_multiple_nodes_same_timepoint(self):
        """Test relabeling with multiple nodes at the same timepoint."""
        # Create segmentation with seg_ids 10, 20, 30 at t=0
        seg = np.zeros((1, 5, 5), dtype=np.uint16)
        seg[0, 1, 1] = 10
        seg[0, 2, 2] = 20
        seg[0, 3, 3] = 30

        graph = nx.DiGraph()
        graph.add_nodes_from([1, 2, 3])

        node_ids = np.array([1, 2, 3])
        seg_ids = np.array([10, 20, 30])
        time_values = np.array([0, 0, 0])

        result = relabel_segmentation(seg, graph, node_ids, seg_ids, time_values)

        assert result[0, 1, 1] == 1
        assert result[0, 2, 2] == 2
        assert result[0, 3, 3] == 3
