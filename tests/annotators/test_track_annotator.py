import pytest

from funtracks.actions import UpdateNodeSeg
from funtracks.annotators import TrackAnnotator
from funtracks.user_actions import (
    UserAddEdge,
    UserAddNode,
    UserDeleteEdge,
    UserDeleteNode,
)


@pytest.mark.parametrize("ndim", [3, 4])
@pytest.mark.parametrize("with_seg", [True, False])
class TestTrackAnnotator:
    def test_init(self, get_tracks, ndim, with_seg) -> None:
        tracks = get_tracks(ndim=ndim, with_seg=with_seg, is_solution=True)
        ann = TrackAnnotator(tracks)
        # Features start disabled by default
        assert len(ann.all_features) == 2
        assert len(ann.features) == 0
        assert len(ann.lineage_id_to_nodes) == 2

        ann = TrackAnnotator(tracks, tracklet_key="track_id")
        assert len(ann.all_features) == 2
        assert len(ann.features) == 0
        assert len(ann.lineage_id_to_nodes) == 2
        assert len(ann.tracklet_id_to_nodes) == 4
        assert ann.max_lineage_id == 2
        assert ann.max_tracklet_id == 5

    def test_compute_all(self, get_tracks, ndim, with_seg) -> None:
        tracks = get_tracks(ndim=ndim, with_seg=with_seg, is_solution=True)

        ann = TrackAnnotator(tracks)
        # Enable features
        ann.activate_features(list(ann.all_features.keys()))
        all_features = ann.features

        # Compute values
        ann.compute()
        for node in tracks.nodes():
            for key in all_features:
                assert key in tracks.graph.nodes[node]

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
            [lineages, tracklets], [ann.lineage_key, ann.tracklet_key], strict=True
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

    def test_add_remove_feature(self, get_tracks, ndim, with_seg):
        tracks = get_tracks(ndim=ndim, with_seg=with_seg, is_solution=True)
        ann = TrackAnnotator(tracks)
        # Enable features
        ann.activate_features(list(ann.all_features.keys()))
        # compute the original tracklet and lineage ids
        ann.compute()
        # add an edge
        node_id = 6
        edge_id = (4, 6)
        tracks.graph.add_edge(*edge_id)
        to_remove_key = ann.lineage_key
        orig_lin = tracks.get_node_attr(node_id, ann.lineage_key, required=True)
        orig_tra = tracks.get_node_attr(node_id, ann.tracklet_key, required=True)

        # remove one feature from computation (annotator level, not FeatureDict)
        ann.deactivate_features([to_remove_key])
        ann.compute()  # this should update tra but not lin
        # lineage_id is still in tracks.features but not recomputed
        assert tracks.get_node_attr(node_id, ann.lineage_key, required=True) == orig_lin
        assert tracks.get_node_attr(node_id, ann.tracklet_key, required=True) != orig_tra

        # add it back in
        ann.activate_features([to_remove_key])
        ann.compute()
        # now both are updated
        assert tracks.get_node_attr(node_id, ann.lineage_key, required=True) != orig_lin
        assert tracks.get_node_attr(node_id, ann.tracklet_key, required=True) != orig_tra

    def test_invalid(self, get_tracks, ndim, with_seg) -> None:
        # Create regular Tracks (not SolutionTracks) to test error handling
        tracks = get_tracks(ndim=ndim, with_seg=with_seg, is_solution=False)
        with pytest.raises(
            ValueError, match="Currently the TrackAnnotator only works on SolutionTracks"
        ):
            TrackAnnotator(tracks)  # type: ignore

    def test_ignores_irrelevant_actions(self, get_tracks, ndim, with_seg):
        """Test that TrackAnnotator ignores actions that don't affect track IDs."""
        if not with_seg:
            pytest.skip("Test requires segmentation")

        tracks = get_tracks(ndim=ndim, with_seg=with_seg, is_solution=True)
        tracks.enable_features(["area", tracks.features.tracklet_key])

        node_id = 3
        initial_track_id = tracks.get_track_id(node_id)

        # UpdateNodeSeg should not trigger track ID update
        orig_pixels = tracks.get_pixels(node_id)
        assert orig_pixels is not None
        pixels_to_remove = tuple(orig_pixels[d][1:] for d in range(len(orig_pixels)))

        # Perform UpdateNodeSeg action
        UpdateNodeSeg(tracks, node_id, pixels_to_remove, added=False)

        # Track ID should remain unchanged (no track update happened)
        assert tracks.get_track_id(node_id) == initial_track_id
        # But area should be updated
        assert tracks.get_node_attr(node_id, "area") == 1

    def test_lineage_id_updated_on_add_and_delete_edge(
        self, get_tracks, ndim, with_seg
    ) -> None:
        tracks = get_tracks(ndim=3, with_seg=False, is_solution=True)
        tracks.enable_features(["lineage_id"])

        # get the existing TrackAnnotator
        ann = next(a for a in tracks.annotators if isinstance(a, TrackAnnotator))

        # ---- UserAddEdge: merge lineages ----
        source_node = 2
        target_node = 6
        UserAddEdge(tracks, edge=(source_node, target_node))

        # Assert target component adopts source lineage id
        assert tracks.get_node_attr(target_node, ann.lineage_key) == tracks.get_node_attr(
            source_node, ann.lineage_key
        )
        assert set(ann.lineage_id_to_nodes[1]) == {1, 2, 3, 4, 5, 6}
        assert 2 not in ann.lineage_id_to_nodes

        # ---- UserDeleteEdge: split lineage ----
        source_node = 3
        target_node = 4

        edge = next(e for e in tracks.graph.edges if set(e) == {3, 4})

        expected_lineage_id = ann.max_lineage_id + 1
        UserDeleteEdge(tracks, edge=edge)

        # Assert target component gets a new lineage id
        component = [4, 5]
        for node in component:
            assert tracks.get_node_attr(node, ann.lineage_key) == expected_lineage_id

        # Assert source component keeps original lineage id
        component = [1, 3, 2, 6]
        for node in component:
            assert tracks.get_node_attr(node, ann.lineage_key) == tracks.get_node_attr(
                source_node, ann.lineage_key
            )

        assert set(ann.lineage_id_to_nodes[1]) == {1, 2, 3, 6}
        assert set(ann.lineage_id_to_nodes[expected_lineage_id]) == {4, 5}

        # ---- Add a node with existing track id ----
        # After the split, only node 3 has track_id=3, and it has lineage_id=1
        # (nodes 4,5 got new track_id=6 and lineage_id=3)
        attrs = {"pos": ([5, 8]), tracks.features.time_key: (5), "track_id": (3)}
        UserAddNode(tracks, node=7, attributes=attrs)

        # Assert new node adopts lineage of existing track (track_id=3 -> lineage_id=1)
        assert tracks.get_node_attr(7, ann.lineage_key) == 1
        assert 7 in ann.lineage_id_to_nodes[1]

        # ---- Add a node with a new track id ----
        attrs = {"pos": ([5, 8]), tracks.features.time_key: (5), "track_id": (4)}
        expected_lineage_id = ann.max_lineage_id + 1
        UserAddNode(tracks, node=8, attributes=attrs)

        # Assert new node adopts a new lineage id
        assert tracks.get_node_attr(8, ann.lineage_key) == expected_lineage_id
        assert 8 in ann.lineage_id_to_nodes[expected_lineage_id]

        # ---- Ensure that deleting a node updates lineage bookkeeping ----
        UserDeleteNode(tracks, node=8)
        assert expected_lineage_id not in ann.lineage_id_to_nodes  # whole list removed

    def test_disabled_tracklet_key_does_nothing(self, get_tracks, ndim, with_seg) -> None:
        """Test that TrackAnnotator does nothing when tracklet_key is disabled."""
        tracks = get_tracks(ndim=ndim, with_seg=with_seg, is_solution=True)
        ann = TrackAnnotator(tracks)

        # Don't activate any features - they should all be disabled
        assert len(ann.features) == 0

        # Store original bookkeeping state
        original_tracklet_map = dict(ann.tracklet_id_to_nodes)
        original_lineage_map = dict(ann.lineage_id_to_nodes)
        original_max_tracklet = ann.max_tracklet_id
        original_max_lineage = ann.max_lineage_id

        # Perform an action that would normally update track IDs
        UserAddEdge(tracks, edge=(4, 6))

        # Bookkeeping should remain unchanged since features are disabled
        assert ann.tracklet_id_to_nodes == original_tracklet_map
        assert ann.lineage_id_to_nodes == original_lineage_map
        assert ann.max_tracklet_id == original_max_tracklet
        assert ann.max_lineage_id == original_max_lineage
