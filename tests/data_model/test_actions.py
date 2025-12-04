import numpy as np
import polars as pl
import pytest
import tracksdata as td
from numpy.testing import assert_array_almost_equal
from polars.testing import assert_frame_equal, assert_series_not_equal
from tracksdata.array import GraphArrayView

from funtracks.data_model import Tracks
from funtracks.data_model.actions import (
    AddEdges,
    AddNodes,
    DeleteEdges,
    UpdateNodeSegs,
)
from funtracks.data_model.graph_attributes import EdgeAttr, NodeAttr
from funtracks.data_model.tracksdata_utils import (
    assert_node_attrs_equal_with_masks,
    create_empty_graphview_graph,
    pixels_to_td_mask,
    td_graph_edge_list,
)


class TestAddDeleteNodes:
    @staticmethod
    @pytest.mark.parametrize("use_seg", [True, False])
    def test_2d_seg(graph_2d, use_seg):
        # start with an empty Tracks
        empty_td_graph = create_empty_graphview_graph(
            database=":memory:", position_attrs=["pos"]
        )

        empty_td_graph_original = td.graph.IndexedRXGraph.from_other(empty_td_graph)

        # empty_array_view = (
        #     GraphArrayView(
        #         graph=empty_td_graph, shape=(5, 100, 100), attr_key="node_id", offset=0
        #     )
        #     if use_seg
        #     else None
        # )
        filled_array_view = GraphArrayView(
            graph=graph_2d, shape=(5, 100, 100), attr_key="node_id", offset=0
        )

        # empty_seg = np.zeros_like(np.asarray(array_view)) if use_seg else None
        tracks = Tracks(empty_td_graph, segmentation_shape=(5, 100, 100), ndim=3)
        # add all the nodes from graph_2d/seg_2d
        nodes = list(graph_2d.node_ids())
        attrs = {}
        attrs[NodeAttr.TIME.value] = [
            graph_2d[node][NodeAttr.TIME.value] for node in nodes
        ]
        if NodeAttr.POS.value == "pos":
            attrs[NodeAttr.POS.value] = [
                graph_2d[node][NodeAttr.POS.value].to_list() for node in nodes
            ]
        else:
            attrs[NodeAttr.POS.value] = [
                graph_2d[node][NodeAttr.POS.value] for node in nodes
            ]
        attrs[NodeAttr.TRACK_ID.value] = [
            graph_2d[node][NodeAttr.TRACK_ID.value] for node in nodes
        ]
        if use_seg:
            pixels = [
                np.nonzero(np.asarray(filled_array_view)[time] == node_id)
                for time, node_id in zip(attrs[NodeAttr.TIME.value], nodes, strict=True)
            ]
            pixels = [
                (np.ones_like(pix[0]) * time, *pix)
                for time, pix in zip(attrs[NodeAttr.TIME.value], pixels, strict=True)
            ]
            mask_list = []
            for pix in pixels:
                mask, _ = pixels_to_td_mask(pix, tracks.ndim, tracks.scale)
                mask_list.append(mask)
            attrs[td.DEFAULT_ATTR_KEYS.MASK] = mask_list
            attrs[td.DEFAULT_ATTR_KEYS.BBOX] = [
                mask.bbox for mask in attrs[td.DEFAULT_ATTR_KEYS.MASK]
            ]
        else:
            pixels = None
            attrs[NodeAttr.AREA.value] = [
                graph_2d[node][NodeAttr.AREA.value] for node in nodes
            ]

        #TODO: Teun: this fails when use_seg is false (pixels=None), AddNodes should handle this
        add_nodes = AddNodes(tracks, nodes, attributes=attrs, pixels=pixels)
        assert set(tracks.graph.node_ids()) == set(graph_2d.node_ids())

        data_graph_2d = graph_2d.node_attrs()[tracks.graph.node_attrs().columns]
        data_tracks = tracks.graph.node_attrs()

        if use_seg:
            assert_array_almost_equal(
                np.asarray(tracks.segmentation), np.asarray(filled_array_view)
            )
            assert_node_attrs_equal_with_masks(data_graph_2d, data_tracks)

        else:
            assert data_graph_2d.drop(["mask", "bbox"]).equals(
                data_tracks.drop(["mask", "bbox"])
            )

        # invert the action to delete all the nodes
        del_nodes = add_nodes.inverse()
        assert set(tracks.graph.node_ids()) == set(empty_td_graph_original.node_ids())
        if use_seg:
            assert np.asarray(tracks.segmentation).sum() == 0
            assert np.asarray(tracks.segmentation).max() == 0

        # re-invert the action to add back all the nodes and their attributes
        del_nodes.inverse()

        assert set(tracks.graph.node_ids()) == set(graph_2d.node_ids())

        data_graph_2d = graph_2d.node_attrs()[tracks.graph.node_attrs().columns]
        data_tracks = tracks.graph.node_attrs()
        if use_seg:
            assert_node_attrs_equal_with_masks(data_graph_2d, data_tracks)
        else:
            assert_frame_equal(
                data_graph_2d.drop(["mask", "bbox", "area"]),
                data_tracks.drop(["mask", "bbox", "area"]),
                check_column_order=False,
                check_row_order=False,
            )

        # TODO: graph.nodes it not allowed with tracksdata
        # for node, data in tracks.graph.nodes(data=True):
        #     graph_2d_data = graph_2d.nodes[node]
        #     # TODO: get back custom attrs https://github.com/funkelab/funtracks/issues/1
        #     if not use_seg:
        #         del graph_2d_data["area"]
        #     assert data == graph_2d_data
        if use_seg:
            assert_array_almost_equal(
                np.asarray(tracks.segmentation), np.asarray(filled_array_view)
            )


def test_update_node_segs(graph_2d):
    graph_2d_original = td.graph.IndexedRXGraph.from_other(graph_2d).filter().subgraph()
    tracks = Tracks(graph=graph_2d, segmentation_shape=(5, 100, 100))
    nodes = list(graph_2d.node_ids())

    array_view_copy = np.asarray(tracks.segmentation).copy()

    # add a couple pixels to the first node
    new_seg = np.asarray(array_view_copy).copy()
    new_seg[0][0] = 1
    nodes = [1]

    pixels = [np.nonzero(np.asarray(array_view_copy) != new_seg)]
    action = UpdateNodeSegs(tracks, nodes, pixels=pixels)

    assert set(tracks.graph.node_ids()) == set(graph_2d.node_ids())
    assert tracks.graph[nodes[0]][NodeAttr.AREA.value] == 1345
    assert_series_not_equal(
        graph_2d_original[nodes[0]][NodeAttr.POS.value],
        tracks.graph[nodes[0]][NodeAttr.POS.value],
    )
    assert_array_almost_equal(tracks.segmentation, new_seg)

    inverse = action.inverse()
    assert set(tracks.graph.node_ids()) == set(graph_2d_original.node_ids())
    assert_node_attrs_equal_with_masks(
        tracks.graph,
        graph_2d_original,
        check_column_order=False,
    )
    assert_array_almost_equal(tracks.segmentation, array_view_copy)

    inverse.inverse()

    assert set(tracks.graph.node_ids()) == set(graph_2d_original.node_ids())
    assert tracks.graph[nodes[0]][NodeAttr.AREA.value] == 1345
    assert_series_not_equal(
        graph_2d_original[nodes[0]][NodeAttr.POS.value],
        tracks.graph[nodes[0]][NodeAttr.POS.value],
    )
    assert_array_almost_equal(tracks.segmentation, new_seg)


def test_duplicate_edges(graph_2d):
    tracks = Tracks(graph_2d, segmentation_shape=(5, 100, 100))
    edges = [[1, 2], [1, 3], [3, 4], [4, 5]]
    for edge in edges:
        with pytest.raises(ValueError):
            AddEdges(tracks, [edge])
    assert set(tracks.graph.edge_ids()) == set(graph_2d.edge_ids())


def test_add_delete_edges(graph_2d):
    # Create a fresh copy of the graph for this test
    node_graph = graph_2d
    tracks = Tracks(node_graph, segmentation_shape=(5, 100, 100))

    segmentation_original = np.asarray(tracks.segmentation).copy()

    edges = [[1, 2], [1, 3], [3, 4], [4, 5]]

    # first delete the edges, before we can add them again
    action = DeleteEdges(tracks, edges)

    action = AddEdges(tracks, edges)
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
    assert_array_almost_equal(tracks.segmentation, segmentation_original)

    inverse = action.inverse()
    assert set(tracks.graph.edge_ids()) == set()
    assert_array_almost_equal(tracks.segmentation, segmentation_original)

    inverse.inverse()
    assert set(tracks.graph.node_ids()) == set(graph_2d.node_ids())
    assert sorted(td_graph_edge_list(tracks.graph)) == sorted(
        td_graph_edge_list(graph_2d)
    )
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
    assert_array_almost_equal(tracks.segmentation, segmentation_original)
