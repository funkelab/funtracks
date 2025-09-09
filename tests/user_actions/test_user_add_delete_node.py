import numpy as np
import pytest

from funtracks.data_model import NodeAttr, SolutionTracks
from funtracks.exceptions import InvalidActionError
from funtracks.user_actions import UserAddNode, UserDeleteNode


@pytest.mark.parametrize("ndim", [3, 4])
@pytest.mark.parametrize("use_seg", [True, False])
class TestUserAddDeleteNode:
    def get_tracks(self, request, ndim, use_seg) -> SolutionTracks:
        seg_name = "segmentation_2d" if ndim == 3 else "segmentation_3d"
        seg = request.getfixturevalue(seg_name) if use_seg else None

        gt_graph = self.get_gt_graph(request, ndim)
        tracks = SolutionTracks(gt_graph, segmentation=seg, ndim=ndim)
        return tracks

    def get_gt_graph(self, request, ndim):
        graph_name = "graph_2d" if ndim == 3 else "graph_3d"
        gt_graph = request.getfixturevalue(graph_name)
        return gt_graph

    def test_user_add_invalid_node(self, request, ndim, use_seg):
        tracks = self.get_tracks(request, ndim, use_seg=use_seg)
        # duplicate node
        with pytest.raises(ValueError, match="Node .* already exists"):
            attrs = {"time": 5, "track_id": 1}
            UserAddNode(tracks, node=1, attributes=attrs)

        # no time
        with pytest.raises(ValueError, match="Cannot add node without time"):
            attrs = {"track_id": 1}
            UserAddNode(tracks, node=7, attributes=attrs)

        # no track_id
        with pytest.raises(ValueError, match="Cannot add node without track id"):
            attrs = {"time": 1}
            UserAddNode(tracks, node=7, attributes=attrs)

        # upstream division
        with pytest.raises(
            InvalidActionError,
            match="Cannot add node here - upstream division event detected",
        ):
            attrs = {"time": 2, "track_id": 1}
            UserAddNode(tracks, node=7, attributes=attrs)

    def test_user_add_node(self, request, ndim, use_seg):
        tracks = self.get_tracks(request, ndim, use_seg)
        # add a node to replace a skip edge between node 4 in time 2 and node 5 in time 4
        node_id = 7
        track_id = 3
        time = 3
        position = [50, 50, 50] if ndim == 4 else [50, 50]
        attributes = {
            NodeAttr.TRACK_ID.value: track_id,
            NodeAttr.POS.value: position,
            NodeAttr.TIME.value: time,
        }
        if use_seg:
            seg_copy = tracks.segmentation.copy()
            if ndim == 3:
                seg_copy[time, position[0], position[1]] = node_id
            else:
                seg_copy[time, position[0], position[1], position[2]] = node_id
            pixels = np.nonzero(seg_copy == node_id)
            del attributes[NodeAttr.POS.value]
        else:
            pixels = None
        graph = tracks.graph
        assert not graph.has_node(node_id)
        assert graph.has_edge(4, 5)
        action = UserAddNode(tracks, node_id, attributes, pixels=pixels)
        assert graph.has_node(node_id)
        assert not graph.has_edge(4, 5)
        assert graph.has_edge(4, node_id)
        assert graph.has_edge(node_id, 5)
        assert tracks.get_position(node_id) == position
        assert tracks.get_track_id(node_id) == track_id
        if use_seg:
            assert tracks.get_area(node_id) == 1

        inverse = action.inverse()
        assert not graph.has_node(node_id)
        assert graph.has_edge(4, 5)

        inverse.inverse()
        assert graph.has_node(node_id)
        assert not graph.has_edge(4, 5)
        assert graph.has_edge(4, node_id)
        assert graph.has_edge(node_id, 5)
        assert tracks.get_position(node_id) == position
        assert tracks.get_track_id(node_id) == track_id
        if use_seg:
            assert tracks.get_area(node_id) == 1
        # TODO: error if node already exists?

    def test_user_delete_node(self, request, ndim, use_seg):
        tracks = self.get_tracks(request, ndim, use_seg)
        # delete node in middle of track. Should skip-connect 3 and 5 with span 3
        node_id = 4

        graph = tracks.graph
        assert graph.has_node(node_id)
        assert graph.has_edge(3, node_id)
        assert graph.has_edge(node_id, 5)
        assert not graph.has_edge(3, 5)

        action = UserDeleteNode(tracks, node_id)
        assert not graph.has_node(node_id)
        assert not graph.has_edge(3, node_id)
        assert not graph.has_edge(node_id, 5)
        assert graph.has_edge(3, 5)

        inverse = action.inverse()
        assert graph.has_node(node_id)
        assert graph.has_edge(3, node_id)
        assert graph.has_edge(node_id, 5)
        assert not graph.has_edge(3, 5)

        inverse.inverse()
        assert not graph.has_node(node_id)
        assert not graph.has_edge(3, node_id)
        assert not graph.has_edge(node_id, 5)
        assert graph.has_edge(3, 5)
        # TODO: error if node doesn't exist?

    def test_user_delete_node_after_division(self, request, ndim, use_seg: bool):
        tracks = self.get_tracks(request, ndim, use_seg)
        # delete first node after division. Should relabel the other child
        # to be the same track as parent
        parent_node = 1
        node_id = 2
        sib = 3

        graph = tracks.graph
        assert graph.has_node(node_id)
        assert graph.has_edge(parent_node, node_id)
        parent_track_id = tracks.get_track_id(parent_node)
        node_track_id = tracks.get_track_id(node_id)
        sib_track_id = tracks.get_track_id(sib)
        assert parent_track_id != node_track_id
        assert parent_track_id != sib_track_id
        assert node_track_id != sib_track_id

        action = UserDeleteNode(tracks, node_id)
        assert not graph.has_node(node_id)
        assert graph.has_edge(parent_node, sib)
        assert tracks.get_track_id(sib) == parent_track_id

        inverse = action.inverse()
        assert graph.has_node(node_id)
        assert graph.has_edge(parent_node, node_id)
        assert tracks.get_track_id(parent_node) == parent_track_id
        assert tracks.get_track_id(node_id) == node_track_id
        assert tracks.get_track_id(sib) == sib_track_id

        inverse.inverse()
        assert not graph.has_node(node_id)
        assert graph.has_edge(parent_node, sib)
        assert tracks.get_track_id(sib) == parent_track_id
