import pytest

from funtracks.data_model import SolutionTracks
from funtracks.user_actions import UserAddEdge, UserDeleteEdge


@pytest.mark.parametrize("ndim", [3, 4])
@pytest.mark.parametrize("use_seg", [True, False])
class TestUserAddDeleteEdge:
    def get_tracks(self, request, ndim, use_seg):
        seg_name = "segmentation_2d" if ndim == 3 else "segmentation_3d"
        seg = request.getfixturevalue(seg_name) if use_seg else None

        gt_graph = self.get_gt_graph(request, ndim)
        tracks = SolutionTracks(gt_graph, segmentation=seg, ndim=ndim)
        return tracks

    def get_gt_graph(self, request, ndim):
        graph_name = "graph_2d" if ndim == 3 else "graph_3d"
        gt_graph = request.getfixturevalue(graph_name)
        return gt_graph

    def test_user_add_edge(self, request, ndim, use_seg):
        tracks = self.get_tracks(request, ndim, use_seg)
        # add an edge from 4 to 6 (will make 4 a division and 5 will need to relabel
        # track id)
        edge = (4, 6)
        old_child = 5
        old_track_id = tracks.get_track_id(old_child)
        assert not tracks.graph.has_edge(*edge)
        action = UserAddEdge(tracks, edge)
        assert tracks.graph.has_edge(*edge)
        assert tracks.get_track_id(old_child) != old_track_id

        inverse = action.inverse()
        assert not tracks.graph.has_edge(*edge)
        assert tracks.get_track_id(old_child) == old_track_id

        inverse.inverse()
        assert tracks.graph.has_edge(*edge)
        assert tracks.get_track_id(old_child) != old_track_id

    def test_user_delete_edge(self, request, ndim, use_seg):
        tracks = self.get_tracks(request, ndim, use_seg)
        # delete edge (1, 3). (1,2) is now not a division anymore
        edge = (1, 3)
        old_child = 2

        old_track_id = tracks.get_track_id(old_child)
        new_track_id = tracks.get_track_id(1)
        assert tracks.graph.has_edge(*edge)

        action = UserDeleteEdge(tracks, edge)
        assert not tracks.graph.has_edge(*edge)
        assert tracks.get_track_id(old_child) == new_track_id

        inverse = action.inverse()
        assert tracks.graph.has_edge(*edge)
        assert tracks.get_track_id(old_child) == old_track_id

        double_inv = inverse.inverse()
        assert not tracks.graph.has_edge(*edge)
        assert tracks.get_track_id(old_child) == new_track_id

        # TODO: error if edge doesn't exist?
        double_inv.inverse()

        # delete edge (3, 4). 4 and 5 should get new track id
        edge = (3, 4)
        old_child = 5

        old_track_id = tracks.get_track_id(old_child)
        assert tracks.graph.has_edge(*edge)

        action = UserDeleteEdge(tracks, edge)
        assert not tracks.graph.has_edge(*edge)
        assert tracks.get_track_id(old_child) != old_track_id

        inverse = action.inverse()
        assert tracks.graph.has_edge(*edge)
        assert tracks.get_track_id(old_child) == old_track_id

        inverse.inverse()
        assert not tracks.graph.has_edge(*edge)
        assert tracks.get_track_id(old_child) != old_track_id
