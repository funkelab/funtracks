import pytest

from funtracks.actions import UpdateNodeSeg, UpdateTrackID
from funtracks.annotators import EdgeAnnotator
from funtracks.data_model import SolutionTracks, Tracks

track_attrs = {"time_attr": "t", "tracklet_attr": "track_id"}


@pytest.mark.parametrize("ndim", [3, 4])
class TestEdgeAnnotator:
    def test_init(self, get_graph, get_segmentation, ndim):
        # Start with clean graph, no existing features
        graph = get_graph(ndim, with_features="clean")
        seg = get_segmentation(ndim)
        tracks = Tracks(graph, segmentation=seg, ndim=ndim, **track_attrs)
        ann = EdgeAnnotator(tracks)
        # Features start disabled by default
        assert len(ann.all_features) == 1
        assert len(ann.features) == 0
        # Enable features to test
        ann.activate_features(list(ann.all_features.keys()))
        assert len(ann.features) == 1

    def test_compute_all(self, get_graph, get_segmentation, ndim):
        graph = get_graph(ndim, with_features="clean")
        seg = get_segmentation(ndim)
        tracks = Tracks(graph, segmentation=seg, ndim=ndim, **track_attrs)
        ann = EdgeAnnotator(tracks)
        # Enable features
        ann.activate_features(list(ann.all_features.keys()))
        all_features = ann.features

        # Compute values
        ann.compute()
        for edge in tracks.edges():
            for key in all_features:
                assert key in tracks.graph.edges[edge]

    def test_update_all(self, get_graph, get_segmentation, ndim) -> None:
        graph = get_graph(ndim, with_features="clean")
        seg = get_segmentation(ndim)
        tracks = Tracks(graph, segmentation=seg, ndim=ndim, **track_attrs)  # type: ignore
        # Get the EdgeAnnotator from the registry
        ann = next(ann for ann in tracks.annotators if isinstance(ann, EdgeAnnotator))
        # Enable features through tracks (which updates the registry)
        tracks.enable_features(list(ann.all_features.keys()))

        node_id = 3
        edge_id = (1, 3)

        orig_pixels = tracks.get_pixels(node_id)
        assert orig_pixels is not None
        # remove all but one pixel
        pixels_to_remove = tuple(orig_pixels[d][1:] for d in range(len(orig_pixels)))
        expected_iou = pytest.approx(0.0, abs=0.001)

        # Use UpdateNodeSeg action to modify segmentation and update edge
        UpdateNodeSeg(tracks, node_id, pixels_to_remove, added=False)
        assert tracks.get_edge_attr(edge_id, "iou", required=True) == expected_iou

        # segmentation is fully erased and you try to update
        node_id = 1
        pixels = tracks.get_pixels(node_id)
        assert pixels is not None
        with pytest.warns(
            match="Cannot find label 1 in frame .*: updating edge IOU value to 0"
        ):
            UpdateNodeSeg(tracks, node_id, pixels, added=False)

        assert tracks.graph.edges[edge_id]["iou"] == 0

    def test_add_remove_feature(self, get_graph, get_segmentation, ndim):
        graph = get_graph(ndim, with_features="clean")
        seg = get_segmentation(ndim)
        tracks = Tracks(graph, segmentation=seg, ndim=ndim, **track_attrs)
        # Get the EdgeAnnotator from the registry
        ann = next(ann for ann in tracks.annotators if isinstance(ann, EdgeAnnotator))
        # Enable features through tracks
        tracks.enable_features(list(ann.all_features.keys()))

        node_id = 3
        edge_id = (1, 3)
        to_remove_key = next(iter(ann.features))
        orig_iou = tracks.get_edge_attr(edge_id, to_remove_key, required=True)

        # remove the IOU from computation (tracks level)
        tracks.disable_features([to_remove_key])
        # remove all but one pixel
        orig_pixels = tracks.get_pixels(node_id)
        assert orig_pixels is not None
        pixels_to_remove = tuple(orig_pixels[d][1:] for d in range(len(orig_pixels)))
        tracks.set_pixels(pixels_to_remove, 0)

        # Compute at tracks level - this should not update the removed feature
        for a in tracks.annotators:
            if isinstance(a, EdgeAnnotator):
                a.compute()
        # IoU was computed before removal, so value is still there
        assert tracks.get_edge_attr(edge_id, to_remove_key, required=True) == orig_iou

        # add it back in
        tracks.enable_features([to_remove_key])
        # Use UpdateNodeSeg action to modify segmentation and update edge
        UpdateNodeSeg(tracks, node_id, pixels_to_remove, added=False)
        new_iou = pytest.approx(0.0, abs=0.001)
        # the feature is now updated
        assert tracks.get_edge_attr(edge_id, to_remove_key, required=True) == new_iou

    def test_missing_seg(self, get_graph, ndim) -> None:
        """Test that EdgeAnnotator gracefully handles missing segmentation."""
        graph = get_graph(ndim, with_features="clean")
        tracks = Tracks(graph, segmentation=None, ndim=ndim, **track_attrs)  # type: ignore

        ann = EdgeAnnotator(tracks)
        assert len(ann.features) == 0
        # Should not raise an error, just return silently
        ann.compute()  # No error expected

    def test_ignores_irrelevant_actions(self, get_graph, get_segmentation, ndim):
        """Test that EdgeAnnotator ignores actions that don't affect edges."""
        graph = get_graph(ndim, with_features="clean")
        seg = get_segmentation(ndim)
        tracks = SolutionTracks(graph, segmentation=seg, ndim=ndim, **track_attrs)
        tracks.enable_features(["iou", track_attrs["tracklet_attr"]])

        edge_id = (1, 3)
        initial_iou = tracks.graph.edges[edge_id]["iou"]

        # Manually modify segmentation (without triggering an action)
        # Remove half the pixels from node 3 (target of the edge)
        node_id = 3
        orig_pixels = tracks.get_pixels(node_id)
        assert orig_pixels is not None
        pixels_to_remove = tuple(
            orig_pixels[d][: len(orig_pixels[d]) // 2] for d in range(len(orig_pixels))
        )
        tracks.set_pixels(pixels_to_remove, 0)

        # If we recomputed IoU now, it would be different
        # But we won't - we'll just call UpdateTrackID on node 1

        # UpdateTrackID should not trigger edge update
        node_id = 1
        original_track_id = tracks.get_track_id(node_id)
        new_track_id = original_track_id + 100

        # Perform UpdateTrackID action
        UpdateTrackID(tracks, node_id, new_track_id)

        # IoU should remain unchanged (no recomputation happened despite seg change)
        assert tracks.graph.edges[edge_id]["iou"] == initial_iou
        # But track_id should be updated
        assert tracks.get_track_id(node_id) == new_track_id
