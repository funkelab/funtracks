import pytest

from funtracks.data_model import Tracks
from funtracks.features import FeatureSet, Position, RegionpropsAnnotator, Time


@pytest.mark.parametrize("ndim", [3, 4])
class TestRegionpropsAnnotator:
    def get_tracks(self, request, ndim):
        seg_name = "segmentation_2d" if ndim == 3 else "segmentation_3d"
        graph_name = "graph_2d" if ndim == 3 else "graph_3d"
        seg = request.getfixturevalue(seg_name)
        graph = request.getfixturevalue(graph_name)
        axes = ("y", "x") if ndim == 3 else ("z", "y", "x")
        features = FeatureSet(Time(), Position(axes))
        return Tracks(graph, segmentation=seg, features=features)

    def test_init(self, request, ndim):
        tracks = self.get_tracks(request, ndim)
        rp_ann = RegionpropsAnnotator(tracks)
        assert len(rp_ann.features) == 4

    def test_area_compute(self, request, ndim):
        tracks = self.get_tracks(request, ndim)
        nodes = list(tracks.nodes())
        areas = tracks.get_areas(nodes)
        for node in nodes:
            if "area" in tracks.graph.nodes[node]:
                del tracks.graph.nodes[node]["area"]
        for node in nodes:
            assert tracks.get_area(node) is None
        assert len(tracks.features._features) == 2

        rp_ann = RegionpropsAnnotator(tracks)

        rp_ann.compute()
        assert len(tracks.features._features) == 6
        for node, area in zip(nodes, areas, strict=True):
            if area is not None:
                assert tracks.graph.nodes[node]["area"] == area
            else:
                assert tracks.graph.nodes[node]["area"] > 0  # TODO: get real values

    def test_area_update(self, request, ndim):
        tracks = self.get_tracks(request, ndim)
        node_id = 3

        orig_pixels = tracks.get_pixels([node_id])[0]
        # remove all but one pixel
        pixels_to_remove = tuple(orig_pixels[d][1:] for d in range(len(orig_pixels)))
        tracks.set_pixels([pixels_to_remove], [0])
        expected_area = 1

        rp_ann = RegionpropsAnnotator(tracks)
        rp_ann.update(node_id)
        assert tracks.get_area(node_id) == expected_area

    # TODO: test values for the other properties
