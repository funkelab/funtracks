import pytest

from funtracks.actions import UpdateNodeSeg, UpdateTrackID
from funtracks.annotators import RegionpropsAnnotator
from funtracks.data_model import SolutionTracks, Tracks

track_attrs = {"time_attr": "t", "tracklet_attr": "track_id"}


@pytest.mark.parametrize("ndim", [3, 4])
class TestRegionpropsAnnotator:
    def test_init(self, get_graph, get_segmentation, ndim):
        graph = get_graph(ndim, with_features="clean")
        seg = get_segmentation(ndim)
        tracks = Tracks(graph, segmentation=seg, ndim=ndim, **track_attrs)
        rp_ann = RegionpropsAnnotator(tracks)
        # Features start disabled by default
        assert len(rp_ann.all_features) == 5
        assert len(rp_ann.features) == 0
        # Enable features
        rp_ann.activate_features(list(rp_ann.all_features.keys()))
        assert (
            len(rp_ann.features) == 5
        )  # pos, area, ellipse_axis_radii, circularity, perimeter

    def test_compute_all(self, get_graph, get_segmentation, ndim):
        graph = get_graph(ndim, with_features="clean")
        seg = get_segmentation(ndim)
        tracks = Tracks(graph, segmentation=seg, ndim=ndim, **track_attrs)
        rp_ann = RegionpropsAnnotator(tracks)
        # Enable features
        rp_ann.activate_features(list(rp_ann.all_features.keys()))

        # Compute values
        rp_ann.compute()
        for node in tracks.nodes():
            for key in rp_ann.features:
                assert key in tracks.graph.nodes[node]

    def test_update_all(self, get_graph, get_segmentation, ndim):
        graph = get_graph(ndim, with_features="clean")
        seg = get_segmentation(ndim)
        tracks = Tracks(graph, segmentation=seg, ndim=ndim, **track_attrs)
        node_id = 3

        # Get the RegionpropsAnnotator from the registry
        rp_ann = next(
            ann for ann in tracks.annotators if isinstance(ann, RegionpropsAnnotator)
        )
        # Enable features through tracks
        tracks.enable_features(list(rp_ann.all_features.keys()))

        orig_pixels = tracks.get_pixels(node_id)
        # remove all but one pixel
        pixels_to_remove = tuple(orig_pixels[d][1:] for d in range(len(orig_pixels)))
        expected_area = 1

        # Use UpdateNodeSeg action to modify segmentation and update features
        UpdateNodeSeg(tracks, node_id, pixels_to_remove, added=False)
        assert tracks.get_node_attr(node_id, "area") == expected_area
        for key in rp_ann.features:
            assert key in tracks.graph.nodes[node_id]

        # segmentation is fully erased and you try to update
        node_id = 1
        pixels = tracks.get_pixels(node_id)
        with pytest.warns(
            match="Cannot find label 1 in frame .*: updating regionprops values to None"
        ):
            UpdateNodeSeg(tracks, node_id, pixels, added=False)

        for key in rp_ann.features:
            assert tracks.graph.nodes[node_id][key] is None

    def test_add_remove_feature(self, get_graph, get_segmentation, ndim):
        graph = get_graph(ndim, with_features="clean")
        seg = get_segmentation(ndim)
        tracks = Tracks(graph, segmentation=seg, ndim=ndim, **track_attrs)
        # Get the RegionpropsAnnotator from the registry
        rp_ann = next(
            ann for ann in tracks.annotators if isinstance(ann, RegionpropsAnnotator)
        )
        all_feature_keys = list(rp_ann.all_features.keys())
        to_remove_key = all_feature_keys[1]  # area
        rp_ann.deactivate_features([to_remove_key])

        # Clear existing area attributes from graph (from fixture)
        for node in tracks.nodes():
            if to_remove_key in tracks.graph.nodes[node]:
                del tracks.graph.nodes[node][to_remove_key]

        rp_ann.compute()
        for node in tracks.nodes():
            assert to_remove_key not in tracks.graph.nodes[node]

        # add it back in
        rp_ann.activate_features([to_remove_key])
        # but remove a different one
        second_remove_key = all_feature_keys[2]  # ellipse_axis_radii
        rp_ann.deactivate_features([second_remove_key])

        # remove all but one pixel
        node_id = 3
        prev_value = tracks.get_node_attr(node_id, second_remove_key)
        orig_pixels = tracks.get_pixels(node_id)
        assert orig_pixels is not None
        pixels_to_remove = tuple(orig_pixels[d][1:] for d in range(len(orig_pixels)))
        # Use UpdateNodeSeg action to modify segmentation and update features
        UpdateNodeSeg(tracks, node_id, pixels_to_remove, added=False)
        # the new one we removed is not updated
        assert tracks.get_node_attr(node_id, second_remove_key) == prev_value
        # the one we added back in is now present
        assert tracks.get_node_attr(node_id, to_remove_key) is not None

    def test_missing_seg(self, get_graph, ndim):
        """Test that RegionpropsAnnotator gracefully handles missing segmentation."""
        graph = get_graph(ndim, with_features="clean")
        tracks = Tracks(graph, segmentation=None, ndim=ndim, **track_attrs)
        rp_ann = RegionpropsAnnotator(tracks)
        assert len(rp_ann.features) == 0
        # Should not raise an error, just return silently
        rp_ann.compute()  # No error expected

    def test_ignores_irrelevant_actions(self, get_graph, get_segmentation, ndim):
        """Test that RegionpropsAnnotator ignores actions that don't affect
        segmentation.
        """
        graph = get_graph(ndim, with_features="clean")
        seg = get_segmentation(ndim)
        tracks = SolutionTracks(graph, segmentation=seg, ndim=ndim, **track_attrs)
        tracks.enable_features(["area", "track_id"])

        node_id = 1
        initial_area = tracks.get_node_attr(node_id, "area")

        # Manually modify segmentation (without triggering an action)
        # Remove half the pixels from node 1
        orig_pixels = tracks.get_pixels(node_id)
        assert orig_pixels is not None
        pixels_to_remove = tuple(
            orig_pixels[d][: len(orig_pixels[d]) // 2] for d in range(len(orig_pixels))
        )
        tracks.set_pixels(pixels_to_remove, 0)

        # If we recomputed area now, it would be different
        # But we won't - we'll just call UpdateTrackID

        # Get original track_id
        original_track_id = tracks.get_track_id(node_id)
        new_track_id = original_track_id + 100

        # Perform UpdateTrackID action
        UpdateTrackID(tracks, node_id, new_track_id)

        # Area should remain unchanged (no recomputation happened despite seg change)
        assert tracks.get_node_attr(node_id, "area") == initial_area
        # But track_id should be updated
        assert tracks.get_track_id(node_id) == new_track_id
