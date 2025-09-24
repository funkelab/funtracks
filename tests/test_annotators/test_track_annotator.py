import pytest

from funtracks.annotators import TrackAnnotator
from funtracks.data_model import SolutionTracks, Tracks
from funtracks.features import FeatureSet, Position, Time


@pytest.mark.parametrize("ndim", [3, 4])
class TestTrackAnnotator:
    def get_tracks(self, request, ndim) -> Tracks:
        seg_name = "segmentation_2d" if ndim == 3 else "segmentation_3d"
        graph_name = "graph_2d" if ndim == 3 else "graph_3d"
        seg = request.getfixturevalue(seg_name)
        graph = request.getfixturevalue(graph_name)
        axes = ("y", "x") if ndim == 3 else ("z", "y", "x")
        features = FeatureSet(Time(), Position(axes))
        return Tracks(graph, segmentation=seg, features=features)

    def get_soln_tracks(self, request, ndim) -> SolutionTracks:
        return SolutionTracks.from_tracks(self.get_tracks(request, ndim))

    def test_init(self, request, ndim) -> None:
        tracks = self.get_soln_tracks(request, ndim)
        ann = TrackAnnotator(tracks)
        assert len(ann.features) == 2
        assert len(ann.lineage_id_to_node) == 0

        ann = TrackAnnotator(tracks, tracklet_key="track_id")
        assert len(ann.features) == 2
        assert len(ann.lineage_id_to_node) == 0
        assert len(ann.tracklet_id_to_node) == 4
        assert ann.max_lineage_id == 0
        assert ann.max_tracklet_id == 5

    def test_compute_all(self, request, ndim) -> None:
        tracks = self.get_soln_tracks(request, ndim)
        assert len(tracks.features._features) == 2

        ann = TrackAnnotator(tracks)
        all_features = ann.features

        ann.compute(add_to_set=True)
        assert len(tracks.features._features) == 2 + len(all_features)
        for node in tracks.nodes():
            for feature in all_features:
                assert feature.key in tracks.graph.nodes[node]

        with pytest.raises(KeyError, match="Key .* already in feature set"):
            ann.compute(add_to_set=True)

        lineages = [
            [1, 2, 3, 4, 5],
            [6],
        ]
        tracklets = [
            [1],
            [2],
            [3, 4, 5],
            [6],
        ]
        for components, key in zip(
            [lineages, tracklets], [ann.lineage.key, ann.tracklet.key], strict=True
        ):
            # one unique id per component
            id_sets = [
                list(set(tracks.get_nodes_attr(component, key, required=True)))
                for component in components
            ]
            for id_set in id_sets:
                assert len(id_set) == 1
            # no shared ids across components
            assert len({id_set[0] for id_set in id_sets}) == len(id_sets)

    def test_add_remove_feature(self, request, ndim: int):
        tracks = self.get_soln_tracks(request, ndim)
        ann = TrackAnnotator(tracks)
        # compute the original tracklet and lineage ids
        ann.compute(add_to_set=True)
        # add an edge
        node_id = 6
        edge_id = (4, 6)
        tracks.graph.add_edge(*edge_id)
        to_remove = ann.lineage
        orig_lin = tracks.get_node_attr(node_id, ann.lineage.key, required=True)
        orig_tra = tracks.get_node_attr(node_id, ann.tracklet.key, required=True)

        # remove one feature from computation
        ann.remove_feature(to_remove, update_set=True)
        ann.compute(add_to_set=False)  # this should update tra but not lin
        assert len(tracks.features._features) == 3  # lin not added to set
        assert tracks.get_node_attr(node_id, ann.lineage.key, required=True) == orig_lin
        assert tracks.get_node_attr(node_id, ann.tracklet.key, required=True) != orig_tra

        # add it back in
        ann.add_feature(to_remove, update_set=True)
        ann.compute(add_to_set=False)
        # now both are updated
        assert tracks.get_node_attr(node_id, ann.lineage.key, required=True) != orig_lin
        assert tracks.get_node_attr(node_id, ann.tracklet.key, required=True) != orig_tra

    def test_invalid(self, request, ndim) -> None:
        tracks = self.get_tracks(request, ndim)
        with pytest.raises(
            ValueError, match="Currently the TrackAnnotator only works on SolutionTracks"
        ):
            TrackAnnotator(tracks)  # type: ignore
