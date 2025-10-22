import pytest

from funtracks.annotators import RegionpropsAnnotator
from funtracks.data_model import Tracks


@pytest.mark.parametrize("ndim", [3, 4])
class TestRegionpropsAnnotator:
    def test_init(self, get_graph, get_segmentation, ndim):
        graph = get_graph(ndim, with_features="clean")
        seg = get_segmentation(ndim)
        tracks = Tracks(graph, segmentation=seg, ndim=ndim)
        rp_ann = RegionpropsAnnotator(tracks)
        assert (
            len(rp_ann.features) == 5
        )  # pos, area, ellipse_axis_radii, circularity, perimeter

    def test_compute_all(self, get_graph, get_segmentation, ndim):
        graph = get_graph(ndim, with_features="clean")
        seg = get_segmentation(ndim)
        tracks = Tracks(graph, segmentation=seg, ndim=ndim)
        rp_ann = RegionpropsAnnotator(tracks)
        all_feature_specs = RegionpropsAnnotator.get_feature_specs(tracks)

        # Compute values
        rp_ann.compute()
        for node in tracks.nodes():
            for spec in all_feature_specs:
                assert spec.key in tracks.graph.nodes[node]

    def test_update_all(self, get_graph, get_segmentation, ndim):
        graph = get_graph(ndim, with_features="clean")
        seg = get_segmentation(ndim)
        tracks = Tracks(graph, segmentation=seg, ndim=ndim)
        node_id = 3

        orig_pixels = tracks.get_pixels(node_id)
        # remove all but one pixel
        pixels_to_remove = tuple(orig_pixels[d][1:] for d in range(len(orig_pixels)))
        tracks.set_pixels(pixels_to_remove, 0)
        expected_area = 1

        rp_ann = RegionpropsAnnotator(tracks)
        all_feature_specs = RegionpropsAnnotator.get_feature_specs(tracks)
        rp_ann.update(node_id)
        assert tracks.get_area(node_id) == expected_area
        for spec in all_feature_specs:
            assert spec.key in tracks.graph.nodes[node_id]
        # update an edge
        with pytest.raises(
            ValueError, match="RegionpropsAnnotator update expected a node, got edge"
        ):
            rp_ann.update((3, 4))

        # segmentation is fully erased and you try to update
        node_id = 1
        pixels = tracks.get_pixels(node_id)
        tracks.set_pixels(pixels, 0)
        with pytest.warns(
            match="Cannot find label 1 in frame .*: updating regionprops values to None"
        ):
            rp_ann.update(node_id)

        for spec in all_feature_specs:
            assert tracks.graph.nodes[node_id][spec.key] is None

    def test_add_remove_feature(self, get_graph, get_segmentation, ndim):
        graph = get_graph(ndim, with_features="clean")
        seg = get_segmentation(ndim)
        tracks = Tracks(graph, segmentation=seg, ndim=ndim)
        rp_ann = tracks.annotator_manager.annotators["regionprops"]
        all_feature_specs = RegionpropsAnnotator.get_feature_specs(tracks)
        to_remove_key = all_feature_specs[1].key  # area
        rp_ann.remove_feature(to_remove_key)

        # Clear existing area attributes from graph (from fixture)
        for node in tracks.nodes():
            if to_remove_key in tracks.graph.nodes[node]:
                del tracks.graph.nodes[node][to_remove_key]

        rp_ann.compute()
        for node in tracks.nodes():
            assert to_remove_key not in tracks.graph.nodes[node]

        # add it back in
        rp_ann.add_feature(to_remove_key)
        # but remove a different one
        second_remove_key = all_feature_specs[2].key  # ellipse_axis_radii
        rp_ann.remove_feature(second_remove_key)

        # remove all but one pixel
        node_id = 3
        prev_value = tracks.get_node_attr(node_id, second_remove_key)
        orig_pixels = tracks.get_pixels(node_id)
        assert orig_pixels is not None
        pixels_to_remove = tuple(orig_pixels[d][1:] for d in range(len(orig_pixels)))
        tracks.set_pixels(pixels_to_remove, 0)
        rp_ann.update(node_id)
        # the new one we removed is not updated
        assert tracks.get_node_attr(node_id, second_remove_key) == prev_value
        # the one we added back in is now present
        assert tracks.get_node_attr(node_id, to_remove_key) is not None

    def test_missing_seg(self, get_graph, ndim):
        graph = get_graph(ndim, with_features="clean")
        tracks = Tracks(graph, segmentation=None, ndim=ndim)
        assert RegionpropsAnnotator.get_feature_specs(tracks) == []
        rp_ann = RegionpropsAnnotator(tracks)
        with pytest.raises(
            ValueError, match="Cannot compute regionprops features without segmentation."
        ):
            rp_ann.compute()
        with pytest.raises(
            ValueError, match="Cannot update regionprops features without segmentation."
        ):
            rp_ann.update(3)
