import pytest

from funtracks.actions import TracksAction
from funtracks.annotators import EdgeAnnotator
from funtracks.data_model import Tracks


@pytest.mark.parametrize("ndim", [3, 4])
class TestEdgeAnnotator:
    def test_init(self, get_graph, get_segmentation, ndim):
        # Start with clean graph, no existing features
        graph = get_graph(ndim, with_features="clean")
        seg = get_segmentation(ndim)
        tracks = Tracks(graph, segmentation=seg, ndim=ndim)
        ann = EdgeAnnotator(tracks)
        # Features start disabled by default
        assert len(ann.all_features) == 1
        assert len(ann.features) == 0
        # Enable features to test
        ann.enable_features(list(ann.all_features.keys()))
        assert len(ann.features) == 1

    def test_compute_all(self, get_graph, get_segmentation, ndim):
        graph = get_graph(ndim, with_features="clean")
        seg = get_segmentation(ndim)
        tracks = Tracks(graph, segmentation=seg, ndim=ndim)
        ann = EdgeAnnotator(tracks)
        # Enable features
        ann.enable_features(list(ann.all_features.keys()))
        all_features = ann.features

        # Compute values
        ann.compute()
        for edge in tracks.edges():
            for key in all_features:
                assert key in tracks.graph.edges[edge]

    def test_update_all(self, get_graph, get_segmentation, ndim) -> None:
        graph = get_graph(ndim, with_features="clean")
        seg = get_segmentation(ndim)
        tracks = Tracks(graph, segmentation=seg, ndim=ndim)
        ann = EdgeAnnotator(tracks)
        # Enable features
        ann.enable_features(list(ann.all_features.keys()))

        node_id = 3
        edge_id = (1, 3)

        # Compute initial values
        ann.compute()

        orig_pixels = tracks.get_pixels(node_id)
        assert orig_pixels is not None
        # remove all but one pixel
        pixels_to_remove = tuple(orig_pixels[d][1:] for d in range(len(orig_pixels)))
        tracks.set_pixels(pixels_to_remove, 0)
        expected_iou = pytest.approx(0.0, abs=0.001)

        action = TracksAction(tracks)
        ann.update(edge_id, action)
        assert tracks.get_edge_attr(edge_id, "iou", required=True) == expected_iou
        # update a node - should be silently ignored
        ann.update(3, action)

        # segmentation is fully erased and you try to update
        node_id = 1
        pixels = tracks.get_pixels(node_id)
        assert pixels is not None
        tracks.set_pixels(pixels, 0)
        with pytest.warns(
            match="Cannot find label 1 in frame .*: updating edge IOU value to 0"
        ):
            ann.update(edge_id, action)

        assert tracks.graph.edges[edge_id]["iou"] == 0

    def test_add_remove_feature(self, get_graph, get_segmentation, ndim):
        graph = get_graph(ndim, with_features="clean")
        seg = get_segmentation(ndim)
        tracks = Tracks(graph, segmentation=seg, ndim=ndim)
        ann = EdgeAnnotator(tracks)
        # Enable features
        ann.enable_features(list(ann.all_features.keys()))

        # Compute initial values
        ann.compute()

        node_id = 3
        edge_id = (1, 3)
        to_remove_key = next(iter(ann.features))
        orig_iou = tracks.get_edge_attr(edge_id, to_remove_key, required=True)

        # remove the IOU from computation (annotator level only)
        ann.disable_features([to_remove_key])
        # remove all but one pixel
        orig_pixels = tracks.get_pixels(node_id)
        assert orig_pixels is not None
        pixels_to_remove = tuple(orig_pixels[d][1:] for d in range(len(orig_pixels)))
        tracks.set_pixels(pixels_to_remove, 0)

        ann.compute()  # this should not update the removed feature
        # IoU was computed before removal, so value is still there
        assert tracks.get_edge_attr(edge_id, to_remove_key, required=True) == orig_iou

        # add it back in
        ann.enable_features([to_remove_key])
        action = TracksAction(tracks)
        ann.update(edge_id, action)
        new_iou = pytest.approx(0.0, abs=0.001)
        # the feature is now updated
        assert tracks.get_edge_attr(edge_id, to_remove_key, required=True) == new_iou

    def test_missing_seg(self, get_graph, ndim) -> None:
        graph = get_graph(ndim, with_features="clean")
        tracks = Tracks(graph, segmentation=None, ndim=ndim)

        ann = EdgeAnnotator(tracks)
        assert len(ann.features) == 0
        with pytest.raises(
            ValueError, match="Cannot compute edge features without segmentation."
        ):
            ann.compute()
        with pytest.raises(
            ValueError, match="Cannot update edge features without segmentation."
        ):
            action = TracksAction(tracks)
            ann.update((1, 3), action)
