import numpy as np
import pandas as pd
import pytest

from funtracks.data_model import SolutionTracks
from funtracks.import_export import tracks_from_df


@pytest.fixture
def simple_df_2d():
    """Simple 2D DataFrame."""
    return pd.DataFrame(
        {
            "time": [0, 1, 1, 2],
            "y": [10.0, 20.0, 30.0, 40.0],
            "x": [15.0, 25.0, 35.0, 45.0],
            "id": [1, 2, 3, 4],
            "parent_id": [-1, 1, 1, 2],
        }
    )


@pytest.fixture
def df_3d():
    """3D DataFrame."""
    return pd.DataFrame(
        {
            "time": [0, 1, 1],
            "z": [5.0, 10.0, 15.0],
            "y": [10.0, 20.0, 30.0],
            "x": [15.0, 25.0, 35.0],
            "id": [1, 2, 3],
            "parent_id": [-1, 1, 1],
        }
    )


class TestDataFrameImportBasic:
    """Test basic DataFrame import."""

    def test_import_2d(self, simple_df_2d):
        """Test importing 2D DataFrame."""
        tracks = tracks_from_df(simple_df_2d)

        assert isinstance(tracks, SolutionTracks)
        assert tracks.graph.number_of_nodes() == 4
        assert tracks.graph.number_of_edges() == 3
        assert tracks.ndim == 3

    def test_import_3d(self, df_3d):
        """Test importing 3D DataFrame."""
        tracks = tracks_from_df(df_3d)

        assert tracks.ndim == 4
        assert tracks.graph.number_of_nodes() == 3
        # Check z coordinate
        pos = tracks.get_position(1)
        assert len(pos) == 3  # z, y, x
        assert pos[0] == 5.0  # z

    def test_with_scale(self, simple_df_2d):
        """Test importing with scale."""
        scale = [1.0, 2.0, 1.5]
        tracks = tracks_from_df(simple_df_2d, scale=scale)

        assert tracks.scale == scale

    def test_node_positions(self, simple_df_2d):
        """Test that node positions are correctly imported."""
        tracks = tracks_from_df(simple_df_2d)

        pos_1 = tracks.get_position(1)
        assert pos_1 == [10.0, 15.0]  # y, x

        pos_2 = tracks.get_position(2)
        assert pos_2 == [20.0, 25.0]

    def test_edges_created(self, simple_df_2d):
        """Test that edges are created from parent_id."""
        tracks = tracks_from_df(simple_df_2d)

        # Check specific edges exist
        assert tracks.graph.has_edge(1, 2)
        assert tracks.graph.has_edge(1, 3)
        assert tracks.graph.has_edge(2, 4)

        # Check node 1 has two children (division)
        assert len(list(tracks.graph.successors(1))) == 2


class TestSegmentationHandling:
    """Test DataFrame import with segmentation."""

    def test_with_2d_segmentation(self, simple_df_2d):
        """Test importing with 2D segmentation."""
        # Add seg_id column (required when segmentation provided)
        df = simple_df_2d.copy()
        df["seg_id"] = df["id"]

        seg = np.zeros((3, 100, 100), dtype=np.uint16)
        seg[0, 10, 15] = 1
        seg[1, 20, 25] = 2
        seg[1, 30, 35] = 3
        seg[2, 40, 45] = 4

        tracks = tracks_from_df(df, seg)

        assert tracks.segmentation is not None
        assert tracks.segmentation.shape == (3, 100, 100)

    def test_seg_id_matches_id(self, simple_df_2d):
        """Test when seg_id matches id (no relabeling needed)."""
        # Add seg_id column matching id
        df = simple_df_2d.copy()
        df["seg_id"] = df["id"]

        seg = np.zeros((3, 100, 100), dtype=np.uint16)
        seg[0, 10, 15] = 1
        seg[1, 20, 25] = 2
        seg[1, 30, 35] = 3
        seg[2, 40, 45] = 4

        tracks = tracks_from_df(df, seg)
        assert tracks.segmentation is not None
        # Segmentation should not be relabeled
        assert tracks.segmentation[0, 10, 15] == 1


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_node(self):
        """Test DataFrame with single node."""
        df = pd.DataFrame(
            {
                "time": [0],
                "y": [10.0],
                "x": [15.0],
                "id": [1],
                "parent_id": [-1],
            }
        )

        tracks = tracks_from_df(df)

        assert tracks.graph.number_of_nodes() == 1
        assert tracks.graph.number_of_edges() == 0

    def test_multiple_roots(self):
        """Test multiple independent lineages."""
        df = pd.DataFrame(
            {
                "time": [0, 0, 1, 1],
                "y": [10.0, 20.0, 15.0, 25.0],
                "x": [15.0, 25.0, 20.0, 30.0],
                "id": [1, 2, 3, 4],
                "parent_id": [-1, -1, 1, 2],  # Two roots
            }
        )

        tracks = tracks_from_df(df)

        assert tracks.graph.number_of_nodes() == 4
        assert tracks.graph.number_of_edges() == 2

        # Should have two root nodes
        roots = [n for n in tracks.graph.nodes() if tracks.graph.in_degree(n) == 0]
        assert len(roots) == 2

    def test_division_event(self):
        """Test cell division (one parent, two children)."""
        df = pd.DataFrame(
            {
                "time": [0, 1, 1],
                "y": [10.0, 20.0, 30.0],
                "x": [15.0, 25.0, 35.0],
                "id": [1, 2, 3],
                "parent_id": [-1, 1, 1],  # 1 divides into 2 and 3
            }
        )

        tracks = tracks_from_df(df)

        assert tracks.graph.number_of_nodes() == 3
        assert tracks.graph.number_of_edges() == 2

        # Node 1 should have two children
        children = list(tracks.graph.successors(1))
        assert len(children) == 2
        assert set(children) == {2, 3}

    def test_long_track(self):
        """Test a long track without divisions."""
        df = pd.DataFrame(
            {
                "time": list(range(10)),
                "y": [float(i * 10) for i in range(10)],
                "x": [float(i * 10) for i in range(10)],
                "id": list(range(1, 11)),
                "parent_id": [-1] + list(range(1, 10)),
            }
        )

        tracks = tracks_from_df(df)

        assert tracks.graph.number_of_nodes() == 10
        assert tracks.graph.number_of_edges() == 9

        # Should form a single linear chain
        roots = [n for n in tracks.graph.nodes() if tracks.graph.in_degree(n) == 0]
        assert len(roots) == 1

        # Each non-leaf node should have exactly one child
        non_leaves = [n for n in tracks.graph.nodes() if tracks.graph.out_degree(n) > 0]
        for node in non_leaves:
            assert tracks.graph.out_degree(node) == 1

    def test_orphaned_node_raises_error(self):
        """Test that node with invalid parent_id raises error."""
        df = pd.DataFrame(
            {
                "time": [0, 1],
                "y": [10.0, 20.0],
                "x": [15.0, 25.0],
                "id": [1, 2],
                "parent_id": [-1, 999],  # Parent 999 doesn't exist
            }
        )

        # tracks_from_df validates that parent exists
        with pytest.raises(ValueError, match="missing nodes"):
            tracks_from_df(df)


class TestFeatureHandling:
    """Test feature computation and loading."""

    def test_load_area_from_df(self):
        """Test loading pre-computed area from DataFrame."""
        df = pd.DataFrame(
            {
                "time": [0, 1],
                "y": [10.0, 20.0],
                "x": [15.0, 25.0],
                "id": [1, 2],
                "parent_id": [-1, 1],
                "area": [100.0, 200.0],
            }
        )

        tracks = tracks_from_df(df, features={"Area": "area"})

        # Area should be loaded from DataFrame
        assert tracks.get_node_attr(1, "area") == 100.0
        assert tracks.get_node_attr(2, "area") == 200.0

    def test_recompute_area_from_seg(self):
        """Test recomputing area from segmentation."""
        df = pd.DataFrame(
            {
                "time": [0, 1],
                "y": [10.0, 20.0],
                "x": [15.0, 25.0],
                "id": [1, 2],
                "parent_id": [-1, 1],
                "seg_id": [1, 2],  # Required when segmentation provided
            }
        )

        # Create segmentation with known areas
        seg = np.zeros((2, 100, 100), dtype=np.uint16)
        seg[0, 8:13, 13:18] = 1  # 5x5 = 25 pixels
        seg[1, 18:23, 23:28] = 2  # 5x5 = 25 pixels

        tracks = tracks_from_df(df, seg, features={"Area": "Recompute"})

        # Area should be computed from segmentation
        assert tracks.get_node_attr(1, "area") == 25
        assert tracks.get_node_attr(2, "area") == 25

    def test_load_multi_value_feature_from_columns(self):
        """Test loading a multi-value feature from separate columns."""
        df = pd.DataFrame(
            {
                "time": [0, 1],
                "y": [10.0, 20.0],
                "x": [15.0, 25.0],
                "id": [1, 2],
                "parent_id": [-1, 1],
                "major_axis": [5.0, 6.0],
                "minor_axis": [2.0, 3.0],
            }
        )

        # Map ellipsoid_axes to a list of column names
        # Position uses composite "pos" mapping
        name_map = {
            "id": "id",
            "parent_id": "parent_id",
            "time": "time",
            "pos": ["y", "x"],  # Composite position mapping
            "ellipsoid_axes": ["major_axis", "minor_axis"],
        }

        tracks = tracks_from_df(df, node_name_map=name_map)

        # The multi-value feature should be loaded as a tuple/array
        axes_1 = tracks.get_node_attr(1, "ellipsoid_axes")
        axes_2 = tracks.get_node_attr(2, "ellipsoid_axes")

        # Values should be combined in order (as a list)
        assert list(axes_1) == [5.0, 2.0]
        assert list(axes_2) == [6.0, 3.0]


class TestDuplicateMappings:
    """Test duplicate value handling in name_map."""

    def test_seg_id_same_as_id(self, simple_df_2d):
        """Test that seg_id can map to same column as id."""
        # This is valid when segmentation labels already match node IDs
        name_map = {
            "id": "id",
            "parent_id": "parent_id",
            "time": "time",
            "pos": ["y", "x"],  # Composite position mapping
            "seg_id": "id",  # seg_id maps to same column as id
        }

        tracks = tracks_from_df(simple_df_2d, node_name_map=name_map)

        # Both id and seg_id should be present with same values
        assert tracks.graph.number_of_nodes() == 4
        for node_id in tracks.graph.nodes():
            assert tracks.get_node_attr(node_id, "seg_id") == node_id

    def test_duplicate_mapping_with_segmentation(self, simple_df_2d):
        """Test seg_id=id with actual segmentation (no relabeling needed)."""
        name_map = {
            "id": "id",
            "parent_id": "parent_id",
            "time": "time",
            "pos": ["y", "x"],  # Composite position mapping
            "seg_id": "id",  # seg_id = id
        }

        seg = np.zeros((3, 100, 100), dtype=np.uint16)
        seg[0, 10, 15] = 1
        seg[1, 20, 25] = 2
        seg[1, 30, 35] = 3
        seg[2, 40, 45] = 4

        tracks = tracks_from_df(simple_df_2d, segmentation=seg, node_name_map=name_map)

        assert tracks.segmentation is not None
        # Segmentation should not be relabeled since seg_id == id
        assert tracks.segmentation[0, 10, 15] == 1


class TestValidationErrors:
    """Test that invalid data raises appropriate errors."""

    def test_non_unique_ids(self):
        """Test that non-unique IDs raise error."""
        df = pd.DataFrame(
            {
                "time": [0, 1],
                "y": [10.0, 20.0],
                "x": [15.0, 25.0],
                "id": [1, 1],  # Duplicate!
                "parent_id": [-1, -1],
            }
        )

        with pytest.raises(ValueError, match="unique"):
            tracks_from_df(df)

    def test_missing_required_column(self):
        """Test that missing required columns raise error."""
        df = pd.DataFrame(
            {
                # Missing 'time' column
                "y": [10.0, 20.0],
                "x": [15.0, 25.0],
                "id": [1, 2],
                "parent_id": [-1, 1],
            }
        )

        # tracks_from_df validates required columns
        with pytest.raises(ValueError, match="None values"):
            tracks_from_df(df)

    def test_pos_mapping_dimension_mismatch(self):
        """Test that pos mapping dimension must match segmentation ndim.

        When the position mapping has more coordinates than the segmentation
        has spatial dimensions, an error is raised during validation.
        """
        df = pd.DataFrame(
            {
                "time": [0, 1],
                "y": [10.0, 20.0],
                "x": [15.0, 25.0],
                "id": [1, 2],
                "parent_id": [-1, 1],
                "seg_id": [1, 2],
            }
        )

        # 2D segmentation (ndim=3: time + 2 spatial)
        seg = np.zeros((2, 100, 100), dtype=np.uint16)
        seg[0, 10, 15] = 1
        seg[1, 20, 25] = 2

        # But provide 3D position mapping (3 coords but seg is 2D spatial)
        name_map = {
            "id": "id",
            "parent_id": "parent_id",
            "time": "time",
            "pos": ["y", "x", "y"],  # 3 coords but seg is 2D
            "seg_id": "seg_id",
        }

        # The mismatch is caught during validation (pos has spatial_dims=True)
        with pytest.raises(ValueError, match="pos.*has 3 values.*2 spatial dimensions"):
            tracks_from_df(df, segmentation=seg, node_name_map=name_map)


class TestSpatialDimsValidation:
    """Test validation of features with spatial_dims=True."""

    def test_ellipse_axis_radii_dimension_mismatch_with_segmentation(self):
        """Test that ellipse_axis_radii with wrong number of values raises error.

        When segmentation is provided, features with spatial_dims=True must
        have num_values matching the number of spatial dimensions.
        """
        df = pd.DataFrame(
            {
                "time": [0, 1],
                "y": [10.0, 20.0],
                "x": [15.0, 25.0],
                "id": [1, 2],
                "parent_id": [-1, 1],
                "seg_id": [1, 2],
                "major_axis": [5.0, 6.0],
                "semi_minor_axis": [3.0, 4.0],
                "minor_axis": [2.0, 3.0],
            }
        )

        # 2D segmentation (ndim=3: time + 2 spatial dims)
        seg = np.zeros((2, 100, 100), dtype=np.uint16)
        seg[0, 10, 15] = 1
        seg[1, 20, 25] = 2

        # Provide 3D ellipse_axis_radii mapping (3 values but seg is 2D spatial)
        name_map = {
            "id": "id",
            "parent_id": "parent_id",
            "time": "time",
            "pos": ["y", "x"],
            "seg_id": "seg_id",
            "ellipse_axis_radii": ["major_axis", "semi_minor_axis", "minor_axis"],
        }

        with pytest.raises(ValueError, match="ellipse_axis_radii.*has 3 values"):
            tracks_from_df(df, segmentation=seg, node_name_map=name_map)

    def test_ellipse_axis_radii_correct_dimensions(self):
        """Test that ellipse_axis_radii with correct dimensions passes validation."""
        df = pd.DataFrame(
            {
                "time": [0, 1],
                "y": [10.0, 20.0],
                "x": [15.0, 25.0],
                "id": [1, 2],
                "parent_id": [-1, 1],
                "seg_id": [1, 2],
                "major_axis": [5.0, 6.0],
                "minor_axis": [2.0, 3.0],
            }
        )

        # 2D segmentation (ndim=3: time + 2 spatial dims)
        seg = np.zeros((2, 100, 100), dtype=np.uint16)
        seg[0, 10, 15] = 1
        seg[1, 20, 25] = 2

        # Provide 2D ellipse_axis_radii mapping (2 values matching 2D spatial)
        name_map = {
            "id": "id",
            "parent_id": "parent_id",
            "time": "time",
            "pos": ["y", "x"],
            "seg_id": "seg_id",
            "ellipse_axis_radii": ["major_axis", "minor_axis"],
        }

        # Should not raise
        tracks = tracks_from_df(df, segmentation=seg, node_name_map=name_map)
        assert tracks.ndim == 3

    def test_pos_spatial_dims_mismatch_with_segmentation(self):
        """Test that position with wrong number of values raises error.

        Position has spatial_dims=True, so the validation catches mismatches
        between pos mapping and segmentation dimensions.
        """
        df = pd.DataFrame(
            {
                "time": [0, 1],
                "z": [5.0, 10.0],
                "y": [10.0, 20.0],
                "x": [15.0, 25.0],
                "id": [1, 2],
                "parent_id": [-1, 1],
                "seg_id": [1, 2],
            }
        )

        # 2D segmentation (ndim=3: time + 2 spatial dims)
        seg = np.zeros((2, 100, 100), dtype=np.uint16)
        seg[0, 10, 15] = 1
        seg[1, 20, 25] = 2

        # Provide 3D position mapping (3 coords but seg is 2D spatial)
        name_map = {
            "id": "id",
            "parent_id": "parent_id",
            "time": "time",
            "pos": ["z", "y", "x"],  # 3 coords but seg is 2D
            "seg_id": "seg_id",
        }

        with pytest.raises(ValueError, match="pos.*has 3 values"):
            tracks_from_df(df, segmentation=seg, node_name_map=name_map)

    def test_spatial_dims_mismatch_without_segmentation(self):
        """Test that spatial_dims validation uses position as fallback.

        When no segmentation is provided, position mapping is used as the
        source of truth for spatial dimensions.
        """
        df = pd.DataFrame(
            {
                "time": [0, 1],
                "y": [10.0, 20.0],
                "x": [15.0, 25.0],
                "id": [1, 2],
                "parent_id": [-1, 1],
                "major_axis": [5.0, 6.0],
                "semi_minor_axis": [3.0, 4.0],
                "minor_axis": [2.0, 3.0],
            }
        )

        # Provide 3D ellipse_axis_radii but 2D position (no segmentation)
        name_map = {
            "id": "id",
            "parent_id": "parent_id",
            "time": "time",
            "pos": ["y", "x"],  # 2D position
            "ellipse_axis_radii": ["major_axis", "semi_minor_axis", "minor_axis"],  # 3D
        }

        # Should raise - ellipse_axis_radii has 3 values but expected 2
        with pytest.raises(
            ValueError, match="ellipse_axis_radii.*has 3 values.*expected 2"
        ):
            tracks_from_df(df, node_name_map=name_map)

    def test_spatial_dims_consistent_without_segmentation(self):
        """Test that consistent spatial_dims features pass without segmentation."""
        df = pd.DataFrame(
            {
                "time": [0, 1],
                "y": [10.0, 20.0],
                "x": [15.0, 25.0],
                "id": [1, 2],
                "parent_id": [-1, 1],
                "major_axis": [5.0, 6.0],
                "minor_axis": [2.0, 3.0],
            }
        )

        # Both pos and ellipse_axis_radii have 2 values (consistent)
        name_map = {
            "id": "id",
            "parent_id": "parent_id",
            "time": "time",
            "pos": ["y", "x"],
            "ellipse_axis_radii": ["major_axis", "minor_axis"],
        }

        # Should not raise - dimensions are consistent
        tracks = tracks_from_df(df, node_name_map=name_map)
        assert tracks is not None

    def test_empty_list_in_name_map_removed(self):
        """Test that empty lists in name_map are removed during preprocessing."""
        df = pd.DataFrame(
            {
                "time": [0, 1],
                "y": [10.0, 20.0],
                "x": [15.0, 25.0],
                "id": [1, 2],
                "parent_id": [-1, 1],
            }
        )

        # Include an empty list for ellipse_axis_radii
        name_map = {
            "id": "id",
            "parent_id": "parent_id",
            "time": "time",
            "pos": ["y", "x"],
            "ellipse_axis_radii": [],  # Empty list - should be removed
        }

        # Should not raise - empty list is removed during preprocessing
        tracks = tracks_from_df(df, node_name_map=name_map)
        assert tracks is not None
        # The empty mapping should not result in a feature being added
        assert not tracks.graph.nodes[1].get("ellipse_axis_radii")

    def test_import_without_position_with_segmentation(self):
        """Test that position can be omitted when segmentation is provided.

        When segmentation is provided, position can be computed from centroids,
        so it's not required in the name_map.
        """
        # Create a simple 2D+T segmentation with two labeled regions
        segmentation = np.zeros((2, 10, 10), dtype=np.int32)
        segmentation[0, 2:5, 2:5] = 1  # Label 1 at t=0
        segmentation[1, 4:7, 4:7] = 2  # Label 2 at t=1

        df = pd.DataFrame(
            {
                "time": [0, 1],
                "id": [1, 2],
                "parent_id": [-1, 1],
            }
        )

        # No position in name_map - should be computed from segmentation
        name_map = {
            "id": "id",
            "parent_id": "parent_id",
            "time": "time",
            # No "pos" mapping
        }

        tracks = tracks_from_df(df, node_name_map=name_map, segmentation=segmentation)
        assert tracks is not None

        # Position should be computed from segmentation centroids
        assert "pos" in tracks.graph.nodes[1]
        pos_1 = tracks.graph.nodes[1]["pos"]
        # Centroid of 3x3 region at [2:5, 2:5] is approximately [3, 3]
        np.testing.assert_array_almost_equal(pos_1, [3.0, 3.0], decimal=0)

    def test_import_without_position_without_segmentation_fails(self):
        """Test that position is required when no segmentation is provided."""
        df = pd.DataFrame(
            {
                "time": [0, 1],
                "id": [1, 2],
                "parent_id": [-1, 1],
            }
        )

        # No position in name_map and no segmentation
        name_map = {
            "id": "id",
            "parent_id": "parent_id",
            "time": "time",
        }

        with pytest.raises(ValueError, match="pos.*mapping.*segmentation"):
            tracks_from_df(df, node_name_map=name_map)
