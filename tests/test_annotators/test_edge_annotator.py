import pytest

from funtracks.annotators import EdgeAnnotator
from funtracks.data_model import Tracks
from funtracks.features import FeatureSet, Position, Time


@pytest.mark.parametrize("ndim", [3, 4])
class TestEdgeAnnotator:
    def get_tracks(self, request, ndim) -> Tracks:
        seg_name = "segmentation_2d" if ndim == 3 else "segmentation_3d"
        graph_name = "graph_2d" if ndim == 3 else "graph_3d"
        seg = request.getfixturevalue(seg_name)
        graph = request.getfixturevalue(graph_name)
        axes = ("y", "x") if ndim == 3 else ("z", "y", "x")
        features = FeatureSet(Time(), Position(axes))
        return Tracks(graph, segmentation=seg, features=features)

    def test_init(self, request, ndim):
        tracks = self.get_tracks(request, ndim)
        ann = EdgeAnnotator(tracks)
        assert len(ann.features) == 1

    def test_compute_all(self, request, ndim):
        tracks = self.get_tracks(request, ndim)
        assert len(tracks.features._features) == 2

        ann = EdgeAnnotator(tracks)
        all_features = ann.features

        ann.compute(add_to_set=True)
        assert len(tracks.features._features) == 3
        for edge in tracks.edges():
            for feature in all_features:
                assert feature.key in tracks.graph.edges[edge]

        with pytest.raises(KeyError, match="Key .* already in feature set"):
            ann.compute(add_to_set=True)

    def test_update_all(self, request, ndim) -> None:
        tracks = self.get_tracks(request, ndim)
        node_id = 3
        edge_id = (1, 3)

        orig_pixels = tracks.get_pixels(node_id)
        assert orig_pixels is not None
        # remove all but one pixel
        pixels_to_remove = tuple(orig_pixels[d][1:] for d in range(len(orig_pixels)))
        tracks.set_pixels(pixels_to_remove, 0)
        expected_iou = pytest.approx(0.0, abs=0.001)

        ann = EdgeAnnotator(tracks)
        ann.update(edge_id)
        assert tracks.get_edge_attr(edge_id, "IoU", required=True) == expected_iou
        # update a node
        with pytest.raises(
            ValueError, match="EdgeAnnotator update expected an edge, got node"
        ):
            ann.update(3)

        # segmentation is fully erased and you try to update
        node_id = 1
        pixels = tracks.get_pixels(node_id)
        assert pixels is not None
        tracks.set_pixels(pixels, 0)
        with pytest.warns(
            match="Cannot find label 1 in frame .*: updating edge IOU value to 0"
        ):
            ann.update(edge_id)

        assert tracks.graph.edges[edge_id]["IoU"] == 0

    def test_add_remove_feature(self, request, ndim: int):
        tracks = self.get_tracks(request, ndim)
        ann = EdgeAnnotator(tracks)
        # compute the original iou
        ann.compute(add_to_set=True)
        node_id = 3
        edge_id = (1, 3)
        to_remove = ann.features[0]
        orig_iou = tracks.get_edge_attr(edge_id, to_remove.key, required=True)

        # remove the IOU from computation
        ann.remove_feature(to_remove, update_set=True)
        # remove all but one pixel
        orig_pixels = tracks.get_pixels(node_id)
        assert orig_pixels is not None
        pixels_to_remove = tuple(orig_pixels[d][1:] for d in range(len(orig_pixels)))
        tracks.set_pixels(pixels_to_remove, 0)

        ann.compute(add_to_set=True)  # this should do nothing
        assert len(tracks.features._features) == 2  # iou not added to set
        assert tracks.get_edge_attr(edge_id, to_remove.key, required=True) == orig_iou

        # add it back in
        ann.add_feature(to_remove, update_set=True)
        ann.update(edge_id)
        new_iou = pytest.approx(0.0, abs=0.001)
        # the new one we removed is not updated
        assert tracks.get_edge_attr(edge_id, to_remove.key, required=True) == new_iou

    def test_missing_seg(self, request, ndim) -> None:
        tracks = self.get_tracks(request, ndim)
        tracks.segmentation = None

        ann = EdgeAnnotator(tracks)
        assert len(ann.features) == 0
        with pytest.raises(
            ValueError, match="Cannot compute edge features without segmentation."
        ):
            ann.compute()
        with pytest.raises(
            ValueError, match="Cannot update edge features without segmentation."
        ):
            ann.update((1, 3))
