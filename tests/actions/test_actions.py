import networkx as nx
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from funtracks.actions import (
    ActionGroup,
    AddEdge,
    AddNode,
    UpdateNodeSeg,
)
from funtracks.data_model import Tracks
from funtracks.data_model.graph_attributes import EdgeAttr, NodeAttr


class TestAddDeleteNodes:
    @staticmethod
    @pytest.mark.parametrize("use_seg", [True, False])
    def test_2d_seg(segmentation_2d, graph_2d, use_seg):
        # start with an empty Tracks
        empty_graph = nx.DiGraph()
        empty_seg = np.zeros_like(segmentation_2d) if use_seg else None
        tracks = Tracks(empty_graph, segmentation=empty_seg, ndim=3)
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


def test_update_node_segs(segmentation_2d, graph_2d):
    tracks = Tracks(graph_2d.copy(), segmentation=segmentation_2d.copy())

    # add a couple pixels to the first node
    new_seg = segmentation_2d.copy()
    new_seg[0][0] = 1
    node = 1

    pixels = np.nonzero(segmentation_2d != new_seg)
    action = UpdateNodeSeg(tracks, node, pixels=pixels, added=True)

    assert set(tracks.graph.nodes()) == set(graph_2d.nodes())
    assert tracks.graph.nodes[1][NodeAttr.AREA.value] == 1345
    assert (
        tracks.graph.nodes[1][NodeAttr.POS.value] != graph_2d.nodes[1][NodeAttr.POS.value]
    )
    assert_array_almost_equal(tracks.segmentation, new_seg)

    inverse = action.inverse()
    assert set(tracks.graph.nodes()) == set(graph_2d.nodes())
    for node, data in tracks.graph.nodes(data=True):
        assert data == graph_2d.nodes[node]
    assert_array_almost_equal(tracks.segmentation, segmentation_2d)

    inverse.inverse()

    assert set(tracks.graph.nodes()) == set(graph_2d.nodes())
    assert tracks.graph.nodes[1][NodeAttr.AREA.value] == 1345
    assert (
        tracks.graph.nodes[1][NodeAttr.POS.value] != graph_2d.nodes[1][NodeAttr.POS.value]
    )
    assert_array_almost_equal(tracks.segmentation, new_seg)


def test_add_delete_edges(graph_2d, segmentation_2d):
    node_graph = nx.create_empty_copy(graph_2d, with_data=True)
    tracks = Tracks(node_graph, segmentation_2d)

    edges = [[1, 2], [1, 3], [3, 4], [4, 5]]

    action = ActionGroup(tracks=tracks, actions=[AddEdge(tracks, edge) for edge in edges])
    # TODO: What if adding an edge that already exists?
    # TODO: test all the edge cases, invalid operations, etc. for all actions
    assert set(tracks.graph.nodes()) == set(graph_2d.nodes())
    for edge in tracks.graph.edges():
        assert tracks.graph.edges[edge][EdgeAttr.IOU.value] == pytest.approx(
            graph_2d.edges[edge][EdgeAttr.IOU.value], abs=0.01
        )
    assert_array_almost_equal(tracks.segmentation, segmentation_2d)

    inverse = action.inverse()
    assert set(tracks.graph.edges()) == set()
    assert_array_almost_equal(tracks.segmentation, segmentation_2d)

    inverse.inverse()
    assert set(tracks.graph.nodes()) == set(graph_2d.nodes())
    assert set(tracks.graph.edges()) == set(graph_2d.edges())
    for edge in tracks.graph.edges():
        assert tracks.graph.edges[edge][EdgeAttr.IOU.value] == pytest.approx(
            graph_2d.edges[edge][EdgeAttr.IOU.value], abs=0.01
        )
    assert_array_almost_equal(tracks.segmentation, segmentation_2d)
