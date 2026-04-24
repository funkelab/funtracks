import numpy as np
import pytest
from tracksdata.nodes import Mask

from funtracks.actions import UpdateNodeSeg
from funtracks.annotators import RegionpropsAnnotator
from funtracks.data_model import Tracks

track_attrs = {"time_attr": "t", "tracklet_attr": "track_id"}


@pytest.mark.parametrize("ndim", [3, 4])
class TestRegionpropsAnnotator:
    def test_init(self, get_graph, ndim):
        graph = get_graph(ndim, with_seg=True)
        tracks = Tracks(
            graph,
            ndim=ndim,
            **track_attrs,
        )
        rp_ann = RegionpropsAnnotator(tracks)
        # Features start disabled by default
        assert len(rp_ann.all_features) == 5
        assert len(rp_ann.features) == 0
        # Enable features
        rp_ann.activate_features(list(rp_ann.all_features.keys()))
        assert (
            len(rp_ann.features) == 5
        )  # pos, area, ellipse_axis_radii, circularity, perimeter

    def test_compute_all(self, get_graph, ndim):
        graph = get_graph(ndim, with_seg=True)
        tracks = Tracks(
            graph,
            ndim=ndim,
            **track_attrs,
        )
        rp_ann = RegionpropsAnnotator(tracks)
        tracks.enable_features(list(rp_ann.all_features.keys()))

        for key in rp_ann.all_features:
            assert key in tracks.graph.node_attr_keys()
            for node_id in tracks.graph.node_ids():
                value = tracks.graph.nodes[node_id][key]
                assert value is not None

    def test_update_all(self, get_graph, ndim):
        graph = get_graph(ndim, with_seg=True)
        tracks = Tracks(
            graph,
            ndim=ndim,
            **track_attrs,
        )
        node_id = 3

        # Get the RegionpropsAnnotator from the registry
        rp_ann = next(
            ann for ann in tracks.annotators if isinstance(ann, RegionpropsAnnotator)
        )
        # Enable features through tracks
        tracks.enable_features(list(rp_ann.all_features.keys()))

        node_mask = tracks.get_mask(node_id)
        removal = Mask(node_mask.mask.copy(), node_mask.bbox)
        removal.mask.flat[np.argmax(removal.mask.flat)] = False
        expected_area = 1

        # Use UpdateNodeSeg action to modify segmentation and update features
        UpdateNodeSeg(tracks, node_id, removal, added=False)
        assert tracks.get_node_attr(node_id, "area") == expected_area
        for key in rp_ann.features:
            assert key in tracks.graph.node_attr_keys()

        # segmentation is fully erased and you try to update
        node_id = 1
        mask = tracks.get_mask(node_id)
        with pytest.warns(
            match="Cannot find label 1 in frame .*: updating regionprops values to None"
        ):
            UpdateNodeSeg(tracks, node_id, mask, added=False)
        # all regionprops features should be the defaults, because seg doesn't exist
        for key in rp_ann.features:
            actual = tracks.graph.nodes[node_id][key]
            expected = tracks.graph._node_attr_schemas()[key].default_value
            # Convert to numpy arrays for comparison (handles both scalar and array types)
            actual_np = np.asarray(actual)
            expected_np = np.asarray(expected)
            assert np.array_equal(actual_np, expected_np)

    def test_add_remove_feature(self, get_graph, ndim):
        graph = get_graph(ndim, with_seg=True)
        tracks = Tracks(
            graph,
            ndim=ndim,
            **track_attrs,
        )
        # Get the RegionpropsAnnotator from the registry
        rp_ann = next(
            ann for ann in tracks.annotators if isinstance(ann, RegionpropsAnnotator)
        )
        all_feature_keys = list(rp_ann.all_features.keys())
        to_remove_key = all_feature_keys[1]  # area
        tracks.disable_features([to_remove_key])

        rp_ann.compute()
        assert to_remove_key not in tracks.graph.node_attr_keys()

        # add it back in
        tracks.enable_features([to_remove_key])
        # but remove a different one
        second_remove_key = all_feature_keys[2]  # ellipse_axis_radii
        tracks.disable_features([second_remove_key])

        # remove all but one pixel
        node_id = 3
        node_mask = tracks.get_mask(node_id)
        assert node_mask is not None
        removal = Mask(node_mask.mask.copy(), node_mask.bbox)
        removal.mask.flat[np.argmax(removal.mask.flat)] = False
        # Use UpdateNodeSeg action to modify segmentation and update features
        UpdateNodeSeg(tracks, node_id, removal, added=False)
        # the one we added back in is now present
        assert tracks.get_node_attr(node_id, to_remove_key) is not None

    def test_missing_seg(self, get_graph, ndim):
        """Test that RegionpropsAnnotator gracefully handles missing segmentation."""
        graph = get_graph(ndim, with_seg=False)
        tracks = Tracks(graph, ndim=ndim, **track_attrs)
        rp_ann = RegionpropsAnnotator(tracks)
        assert len(rp_ann.features) == 0
        # Should not raise an error, just return silently
        rp_ann.compute()  # No error expected

    def test_centroid_world_coords_with_scale(self, get_graph, ndim):
        """Centroid in 'pos' must be pixel_centroid * scale (world units).

        Without the fix, skimage returns local_centroid * spacing + bbox_min_pixel
        (mixed units). The correct formula is (local_centroid + bbox_min_pixel) *
        spacing = pixel_centroid * spacing.

        Node 6 has a cube/square at corner (96, 96, ...) with width 4,
        so pixel centroid = 97.5 in each spatial axis.
        """
        graph = get_graph(ndim, with_seg=True)
        if ndim == 3:
            scale = [1.0, 2.0, 3.0]
            pixel_centroid = np.array([97.5, 97.5])
        else:
            scale = [1.0, 2.0, 3.0, 4.0]
            pixel_centroid = np.array([97.5, 97.5, 97.5])

        tracks = Tracks(graph, ndim=ndim, scale=scale, **track_attrs)
        # Force recomputation so regionprops runs with the given scale as spacing
        tracks.enable_features(["pos"])

        pos = np.array(tracks.graph.nodes[6]["pos"])
        expected = pixel_centroid * np.array(scale[1:])

        bug_value = np.array([1.5] * len(pixel_centroid)) * np.array(
            scale[1:]
        ) + np.array([96.0] * len(pixel_centroid))
        np.testing.assert_allclose(
            pos,
            expected,
            atol=0.1,
            err_msg=(
                f"World centroid must be pixel_centroid * scale. "
                f"Got {pos}, expected {expected}. "
                f"Bug value would be local_centroid * scale + bbox_min = {bug_value}"
            ),
        )
