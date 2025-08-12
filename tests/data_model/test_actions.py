import networkx as nx
import numpy as np
import polars as pl
import pytest
from numpy.testing import assert_array_almost_equal
from polars.testing import assert_frame_equal

from funtracks.data_model import Tracks
from funtracks.data_model.actions import (
    AddEdges,
    AddNodes,
    DeleteEdges,
    UpdateNodeSegs,
)
from funtracks.data_model.graph_attributes import EdgeAttr, NodeAttr
from funtracks.data_model.utils import (
    convert_nx_to_td_indexedrxgraph,
    td_get_single_attr_from_node,
    td_graph_edge_list,
)


class TestAddDeleteNodes:
    @staticmethod
    @pytest.mark.parametrize("use_seg", [True, False])
    def test_2d_seg(segmentation_2d, graph_2d, use_seg):
        # start with an empty Tracks
        empty_td_graph = convert_nx_to_td_indexedrxgraph(nx.DiGraph())
        empty_td_graph.add_node_attr_key(key="pos", default_value=[0, 0, 0])
        empty_td_graph.add_node_attr_key(key="t", default_value=0)
        empty_td_graph.add_node_attr_key(key="track_id", default_value=0)
        empty_td_graph.add_node_attr_key(key="area", default_value=0)
        empty_td_graph.add_node_attr_key(key="solution", default_value=1)

        empty_seg = np.zeros_like(segmentation_2d) if use_seg else None
        tracks = Tracks(empty_td_graph, segmentation=empty_seg, ndim=3)
        # add all the nodes from graph_2d/seg_2d
        nodes = list(graph_2d.node_ids())
        attrs = {}
        attrs[NodeAttr.TIME.value] = [
            # graph_2d.nodes[node][NodeAttr.TIME.value] for node in nodes
            td_get_single_attr_from_node(graph_2d, [node], [NodeAttr.TIME.value])
            for node in nodes
        ]
        attrs[NodeAttr.POS.value] = [
            td_get_single_attr_from_node(graph_2d, [node], [NodeAttr.POS.value])
            for node in nodes
        ]
        attrs[NodeAttr.TRACK_ID.value] = [
            td_get_single_attr_from_node(graph_2d, [node], [NodeAttr.TRACK_ID.value])
            for node in nodes
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
                td_get_single_attr_from_node(graph_2d, [node], [NodeAttr.AREA.value])
                for node in nodes
            ]
        add_nodes = AddNodes(tracks, nodes, attributes=attrs, pixels=pixels)

        assert set(tracks.graph.node_ids()) == set(graph_2d.node_ids())

        data_graph_2d = graph_2d.node_attrs()[tracks.graph.node_attrs().columns]
        data_tracks = tracks.graph.node_attrs()
        assert data_graph_2d.equals(data_tracks)
        if use_seg:
            assert_array_almost_equal(tracks.segmentation, segmentation_2d)

        # TODO: somehow, graph.copy() doesn't work for IndexedRXGraph,
        # because it messes up with the internal mapping, so we just
        # create a new empty_td_graph, purely for the assert
        empty_td_graph2 = convert_nx_to_td_indexedrxgraph(nx.DiGraph())
        empty_td_graph2.add_node_attr_key(key="pos", default_value=[0, 0, 0])
        empty_td_graph2.add_node_attr_key(key="t", default_value=0)
        empty_td_graph2.add_node_attr_key(key="track_id", default_value=0)
        empty_td_graph2.add_node_attr_key(key="area", default_value=0)
        empty_td_graph2.add_node_attr_key(key="solution", default_value=1)

        # invert the action to delete all the nodes
        del_nodes = add_nodes.inverse()
        assert set(tracks.graph.node_ids()) == set(empty_td_graph2.node_ids())
        if use_seg:
            assert_array_almost_equal(tracks.segmentation, empty_seg)

        # re-invert the action to add back all the nodes and their attributes
        del_nodes.inverse()
        assert set(tracks.graph.node_ids()) == set(graph_2d.node_ids())

        data_graph_2d = graph_2d.node_attrs()[tracks.graph.node_attrs().columns]
        data_tracks = tracks.graph.node_attrs()
        assert data_graph_2d.equals(data_tracks)

        # for node, data in tracks.graph.nodes(data=True):
        #     graph_2d_data = graph_2d.nodes[node]
        #     # TODO: get back custom attrs https://github.com/funkelab/funtracks/issues/1
        #     if not use_seg:
        #         del graph_2d_data["area"]
        #     assert data == graph_2d_data
        if use_seg:
            assert_array_almost_equal(tracks.segmentation, segmentation_2d)


def test_update_node_segs(segmentation_2d, graph_2d):
    tracks = Tracks(graph=graph_2d.copy(), segmentation=segmentation_2d.copy())
    # TODO: add copies back?
    nodes = list(graph_2d.node_ids())

    # add a couple pixels to the first node
    new_seg = segmentation_2d.copy()
    new_seg[0][0] = 1
    nodes = [1]

    pixels = [np.nonzero(segmentation_2d != new_seg)]
    action = UpdateNodeSegs(tracks, nodes, pixels=pixels, added=True)

    assert set(tracks.graph.node_ids()) == set(graph_2d.node_ids())
    assert (
        td_get_single_attr_from_node(tracks.graph, nodes, [NodeAttr.AREA.value]) == 1345
    )
    assert td_get_single_attr_from_node(
        tracks.graph, nodes, [NodeAttr.POS.value]
    ) != td_get_single_attr_from_node(graph_2d, nodes, [NodeAttr.POS.value])
    assert_array_almost_equal(tracks.segmentation, new_seg)

    inverse = action.inverse()
    assert set(tracks.graph.node_ids()) == set(graph_2d.node_ids())
    assert_frame_equal(
        tracks.graph.node_attrs(), graph_2d.node_attrs(), check_column_order=False
    )
    assert_array_almost_equal(tracks.segmentation, segmentation_2d)

    inverse.inverse()

    assert set(tracks.graph.node_ids()) == set(graph_2d.node_ids())
    assert (
        td_get_single_attr_from_node(tracks.graph, nodes, [NodeAttr.AREA.value]) == 1345
    )
    assert td_get_single_attr_from_node(
        tracks.graph, nodes, [NodeAttr.POS.value]
    ) != td_get_single_attr_from_node(graph_2d, nodes, [NodeAttr.POS.value])
    assert_array_almost_equal(tracks.segmentation, new_seg)


def test_add_delete_edges(graph_2d, segmentation_2d):
    # Create a fresh copy of the graph for this test

    node_graph = graph_2d.copy()
    tracks = Tracks(node_graph, segmentation_2d.copy())

    edges = [[1, 2], [1, 3], [3, 4], [4, 5]]

    # first delete the edges, before we can add them again
    action = DeleteEdges(tracks, edges)

    action = AddEdges(tracks, edges)
    # TODO: What if adding an edge that already exists?
    # TODO: test all the edge cases, invalid operations, etc. for all actions
    assert set(tracks.graph.node_ids()) == set(graph_2d.node_ids())

    # edge_ids are not preserved in td.graph.copy(), edges get re-assigned edge_ids.
    # so, we check the actual edges, not using edge_ids
    for edge in td_graph_edge_list(tracks.graph):
        edge_id_tracks = tracks.graph.edge_id(edge[0], edge[1])
        edge_id_graph = graph_2d.edge_id(edge[0], edge[1])

        assert tracks.graph.edge_attrs().filter(pl.col("edge_id") == edge_id_tracks)[
            EdgeAttr.IOU.value
        ].item() == pytest.approx(
            graph_2d.edge_attrs()
            .filter(pl.col("edge_id") == edge_id_graph)[EdgeAttr.IOU.value]
            .item(),
            abs=0.01,
        )
    assert_array_almost_equal(tracks.segmentation, segmentation_2d)

    inverse = action.inverse()
    assert set(tracks.graph.edge_ids()) == set()
    assert_array_almost_equal(tracks.segmentation, segmentation_2d)

    inverse.inverse()
    assert set(tracks.graph.node_ids()) == set(graph_2d.node_ids())
    assert td_graph_edge_list(tracks.graph) == td_graph_edge_list(graph_2d)
    for edge in td_graph_edge_list(tracks.graph):
        edge_id_tracks = tracks.graph.edge_id(edge[0], edge[1])
        edge_id_graph = graph_2d.edge_id(edge[0], edge[1])

        assert tracks.graph.edge_attrs().filter(pl.col("edge_id") == edge_id_tracks)[
            EdgeAttr.IOU.value
        ].item() == pytest.approx(
            graph_2d.edge_attrs()
            .filter(pl.col("edge_id") == edge_id_graph)[EdgeAttr.IOU.value]
            .item(),
            abs=0.01,
        )
    assert_array_almost_equal(tracks.segmentation, segmentation_2d)
