import networkx as nx
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from funtracks.data_model import Tracks
from funtracks.data_model.actions import (
    AddEdges,
    AddNodes,
    UpdateNodeSegs,
)
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
        attrs = {}
        attrs[NodeAttr.TIME.value] = [
            graph_2d.nodes[node][NodeAttr.TIME.value] for node in nodes
        ]
        attrs[NodeAttr.POS.value] = [
            graph_2d.nodes[node][NodeAttr.POS.value] for node in nodes
        ]
        attrs[NodeAttr.TRACK_ID.value] = [
            graph_2d.nodes[node][NodeAttr.TRACK_ID.value] for node in nodes
        ]
        if use_seg:
            pixels = [
                np.nonzero(segmentation_2d[time] == node_id)
                for time, node_id in zip(attrs[NodeAttr.TIME.value], nodes, strict=True)
            ]
            pixels = [
                (np.ones_like(pix[0]) * time, *pix)
                for time, pix in zip(attrs[NodeAttr.TIME.value], pixels, strict=True)
            ]
        else:
            pixels = None
            attrs[NodeAttr.AREA.value] = [
                graph_2d.nodes[node][NodeAttr.AREA.value] for node in nodes
            ]
        add_nodes = AddNodes(tracks, nodes, attributes=attrs, pixels=pixels)

        assert set(tracks.graph.nodes()) == set(graph_2d.nodes())
        for node, data in tracks.graph.nodes(data=True):
            graph_2d_data = graph_2d.nodes[node]
            assert data == graph_2d_data
        if use_seg:
            assert_array_almost_equal(tracks.segmentation, segmentation_2d)

        # invert the action to delete all the nodes
        del_nodes = add_nodes.inverse()
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
    nodes = list(graph_2d.nodes())

    # add a couple pixels to the first node
    new_seg = segmentation_2d.copy()
    new_seg[0][0] = 1
    nodes = [1]

    pixels = [np.nonzero(segmentation_2d != new_seg)]
    action = UpdateNodeSegs(tracks, nodes, pixels=pixels, added=True)

    assert set(tracks.graph.nodes()) == set(graph_2d.nodes())
    assert tracks.graph.nodes[1][NodeAttr.AREA.value] == 1345
    assert (
        tracks.graph.nodes[1][NodeAttr.POS.value]
        != graph_2d.nodes[1][NodeAttr.POS.value]
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
        tracks.graph.nodes[1][NodeAttr.POS.value]
        != graph_2d.nodes[1][NodeAttr.POS.value]
    )
    assert_array_almost_equal(tracks.segmentation, new_seg)


def test_add_delete_edges(graph_2d, segmentation_2d):
    node_graph = nx.create_empty_copy(graph_2d, with_data=True)
    tracks = Tracks(node_graph, segmentation_2d)

    edges = [[1, 2], [1, 3], [3, 4], [4, 5]]

    action = AddEdges(tracks, edges)
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
