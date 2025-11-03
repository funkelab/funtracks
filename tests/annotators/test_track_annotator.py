import pytest

from funtracks.actions import UpdateNodeSeg
from funtracks.annotators import TrackAnnotator


@pytest.mark.parametrize("ndim", [3, 4])
@pytest.mark.parametrize("with_seg", [True, False])
class TestTrackAnnotator:
    def test_init(self, get_tracks, ndim, with_seg) -> None:
        tracks = get_tracks(ndim=ndim, with_seg=with_seg, is_solution=True)
        ann = TrackAnnotator(tracks)
        # Features start disabled by default
        assert len(ann.all_features) == 2
        assert len(ann.features) == 0
        assert len(ann.lineage_id_to_nodes) == 0

        ann = TrackAnnotator(tracks, tracklet_key="track_id")
        assert len(ann.all_features) == 2
        assert len(ann.features) == 0
        assert len(ann.lineage_id_to_nodes) == 0
        assert len(ann.tracklet_id_to_nodes) == 4
        assert ann.max_lineage_id == 0
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


def test_tracklet_and_lineage_assignment_with_merges(graph_with_merges):
    """Test that both tracklet and lineage IDs work correctly with merge events.

    The graph has:
    - One merge at node 3 (from nodes 1 and 2)
    - One division at node 4 (to nodes 5 and 6)
    - One merge at node 8 (from nodes 6 and 9)

    Expected tracklets (after removing division and merge edges):
    - [1], [2], [3, 4], [5, 7], [6], [9], [8] = 7 tracklets

    Expected lineages (weakly connected components):
    - All 9 nodes in one lineage = 1 lineage
    """
    from funtracks.data_model import SolutionTracks

    # Create SolutionTracks from the graph with merges
    tracks = SolutionTracks(graph_with_merges, segmentation=None, ndim=3)

    # Create annotator and activate both features
    ann = TrackAnnotator(tracks)
    ann.activate_features([ann.tracklet_key, ann.lineage_key])

    # Compute both features at once
    ann.compute()

    # Define expected tracklets based on merge/division structure
    expected_tracklets = [
        [1],  # ends at merge into node 3
        [2],  # ends at merge into node 3
        [3, 4],  # starts from merge, ends at division
        [5, 7],  # starts from division from node 4
        [6],  # starts from division, ends at merge into node 8
        [9],  # ends at merge into node 8
        [8],  # starts from merge
    ]

    # Verify we have the expected number of tracklets
    assert len(ann.tracklet_id_to_nodes) == len(expected_tracklets), (
        f"Expected {len(expected_tracklets)} tracklets, "
        f"got {len(ann.tracklet_id_to_nodes)}"
    )

    # For each expected tracklet, verify nodes have same tracklet_id
    for expected_nodes in expected_tracklets:
        tracklet_ids = [tracks.get_track_id(node) for node in expected_nodes]
        assert len(set(tracklet_ids)) == 1, (
            f"Nodes {expected_nodes} should all have the same tracklet_id, "
            f"got {tracklet_ids}"
        )
        tracklet_id = tracklet_ids[0]
        actual_nodes = sorted(ann.tracklet_id_to_nodes[tracklet_id])
        assert actual_nodes == sorted(expected_nodes), (
            f"For tracklet_id {tracklet_id}, expected nodes {sorted(expected_nodes)}, "
            f"got {actual_nodes}"
        )

    # Verify lineages: should be 1 lineage containing all nodes
    assert len(ann.lineage_id_to_nodes) == 1, (
        f"Expected 1 lineage (all nodes connected), got {len(ann.lineage_id_to_nodes)}"
    )

    # Verify all nodes have both tracklet_id and lineage_id
    all_nodes = list(graph_with_merges.nodes())
    for node in all_nodes:
        tracklet_id = tracks.get_node_attr(node, ann.tracklet_key)
        lineage_id = tracks.get_node_attr(node, ann.lineage_key)
        assert tracklet_id is not None, f"Node {node} missing tracklet_id"
        assert lineage_id is not None, f"Node {node} missing lineage_id"

    # Verify the single lineage contains all 9 nodes
    lineage_id = list(ann.lineage_id_to_nodes.keys())[0]
    assert sorted(ann.lineage_id_to_nodes[lineage_id]) == sorted(all_nodes), (
        f"Lineage should contain all nodes {sorted(all_nodes)}, "
        f"got {sorted(ann.lineage_id_to_nodes[lineage_id])}"
    )
