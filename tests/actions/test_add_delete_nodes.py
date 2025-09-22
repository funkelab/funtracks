import networkx as nx
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from funtracks.actions import (
    ActionGroup,
    AddNode,
)
from funtracks.data_model import SolutionTracks


class TestAddDeleteNodes:
    @staticmethod
    @pytest.mark.parametrize("use_seg", [True, False])
    def test_2d_seg(segmentation_2d, graph_2d, use_seg):
        # start with an empty Tracks
        empty_graph = nx.DiGraph()
        empty_seg = np.zeros_like(segmentation_2d) if use_seg else None
        tracks = SolutionTracks(empty_graph, segmentation=empty_seg, ndim=3)
        # add all the nodes from graph_2d/seg_2d

        nodes = list(graph_2d.nodes())
        actions = []
        for node in nodes:
            pixels = np.nonzero(segmentation_2d == node) if use_seg else None
            actions.append(
                AddNode(tracks, node, dict(graph_2d.nodes[node]), pixels=pixels)
            )
        action = ActionGroup(tracks=tracks, actions=actions)

        assert set(tracks.graph.nodes()) == set(graph_2d.nodes())
        for node, data in tracks.graph.nodes(data=True):
            graph_2d_data = graph_2d.nodes[node]
            assert data == graph_2d_data
        if use_seg:
            assert_array_almost_equal(tracks.segmentation, segmentation_2d)

        # invert the action to delete all the nodes
        del_nodes = action.inverse()
        assert set(tracks.graph.nodes()) == set(empty_graph.nodes())
        if use_seg:
            assert_array_almost_equal(tracks.segmentation, empty_seg)

        # re-invert the action to add back all the nodes and their attributes
        del_nodes.inverse()
        assert set(tracks.graph.nodes()) == set(graph_2d.nodes())
        for node, data in tracks.graph.nodes(data=True):
            graph_2d_data = graph_2d.nodes[node]
            # TODO: get back custom attrs https://github.com/funkelab/funtracks/issues/1
            if not use_seg:
                del graph_2d_data["area"]
            assert data == graph_2d_data
        if use_seg:
            assert_array_almost_equal(tracks.segmentation, segmentation_2d)


def test_add_node_missing_time(graph_2d):
    tracks = SolutionTracks(graph_2d, ndim=3)
    with pytest.raises(ValueError, match="Must provide a time attribute for each node"):
        AddNode(tracks, 8, {})


def test_add_node_missing_pos(graph_2d):
    tracks = SolutionTracks(graph_2d, ndim=3)
    with pytest.raises(
        ValueError, match="Must provide positions or segmentation and ids"
    ):
        AddNode(tracks, 8, {"time": 2})
