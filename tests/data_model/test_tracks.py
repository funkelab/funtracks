import networkx as nx
import numpy as np
import pytest
from networkx.utils import graphs_equal
from numpy.testing import assert_array_almost_equal

from funtracks.data_model import EdgeAttr, NodeAttr, Tracks


def test_create_tracks(graph_3d, segmentation_3d):
    # create empty tracks
    tracks = Tracks(graph=nx.DiGraph(), ndim=3)
    with pytest.raises(KeyError):
        tracks.get_positions([1])

    # create tracks with graph only
    tracks = Tracks(graph=graph_3d, ndim=4)
    assert tracks.get_positions([1]).tolist() == [[50, 50, 50]]
    assert tracks.get_time(1) == 0
    with pytest.raises(KeyError):
        tracks.get_positions(["0"])

    # create track with graph and seg
    tracks = Tracks(graph=graph_3d, segmentation=segmentation_3d)
    assert tracks.get_positions([1]).tolist() == [[50, 50, 50]]
    assert tracks.get_time(1) == 0
    assert tracks.get_positions([1], incl_time=True).tolist() == [[0, 50, 50, 50]]
    tracks.set_time(1, 1)
    assert tracks.get_positions([1], incl_time=True).tolist() == [[1, 50, 50, 50]]

    tracks_wrong_attr = Tracks(
        graph=graph_3d, segmentation=segmentation_3d, time_attr="test"
    )
    with pytest.raises(KeyError):  # raises error at access if time is wrong
        tracks_wrong_attr.get_times([1])

    tracks_wrong_attr = Tracks(graph=graph_3d, pos_attr="test", ndim=3)
    with pytest.raises(KeyError):  # raises error at access if pos is wrong
        tracks_wrong_attr.get_positions([1])

    # test multiple position attrs
    pos_attr = ("z", "y", "x")
    for node in graph_3d.nodes():
        pos = graph_3d.nodes[node][NodeAttr.POS.value]
        z, y, x = pos
        del graph_3d.nodes[node][NodeAttr.POS.value]
        graph_3d.nodes[node]["z"] = z
        graph_3d.nodes[node]["y"] = y
        graph_3d.nodes[node]["x"] = x

    tracks = Tracks(graph=graph_3d, pos_attr=pos_attr, ndim=4)
    assert tracks.get_positions([1]).tolist() == [[50, 50, 50]]
    tracks.set_position(1, [55, 56, 57])
    assert tracks.get_position(1) == [55, 56, 57]

    tracks.set_position(1, [1, 50, 50, 50], incl_time=True)
    assert tracks.get_time(1) == 1


def test_pixels_and_seg_id(graph_3d, segmentation_3d):
    # create track with graph and seg
    tracks = Tracks(graph=graph_3d, segmentation=segmentation_3d)

    # changing a segmentation id changes it in the mapping
    pix = tracks.get_pixels(1)
    new_seg_id = 10
    tracks.set_pixels(pix, new_seg_id)


def test_save_load_delete(tmp_path, graph_2d, segmentation_2d):
    tracks_dir = tmp_path / "tracks"
    tracks = Tracks(graph_2d, segmentation_2d)
    with pytest.warns(
        DeprecationWarning,
        match="`Tracks.save` is deprecated and will be removed in 2.0",
    ):
        tracks.save(tracks_dir)
    with pytest.warns(
        DeprecationWarning,
        match="`Tracks.load` is deprecated and will be removed in 2.0",
    ):
        loaded = Tracks.load(tracks_dir)
        assert graphs_equal(loaded.graph, tracks.graph)
        assert_array_almost_equal(loaded.segmentation, tracks.segmentation)
    with pytest.warns(
        DeprecationWarning,
        match="`Tracks.delete` is deprecated and will be removed in 2.0",
    ):
        Tracks.delete(tracks_dir)


def test_nodes_edges(graph_2d):
    tracks = Tracks(graph_2d, ndim=3)
    assert set(tracks.nodes()) == {1, 2, 3, 4, 5, 6}
    assert set(map(tuple, tracks.edges())) == {(1, 2), (1, 3), (3, 4), (4, 5)}


def test_degrees(graph_2d):
    tracks = Tracks(graph_2d, ndim=3)
    assert tracks.in_degree(np.array([1])) == 0
    assert tracks.in_degree(np.array([4])) == 1
    assert np.array_equal(
        tracks.in_degree(None), np.array([[1, 0], [2, 1], [3, 1], [4, 1], [5, 1], [6, 0]])
    )
    assert np.array_equal(tracks.out_degree(np.array([1, 4])), np.array([2, 1]))
    assert np.array_equal(
        tracks.out_degree(None),
        np.array([[1, 2], [2, 0], [3, 1], [4, 1], [5, 0], [6, 0]]),
    )


def test_predecessors_successors(graph_2d):
    tracks = Tracks(graph_2d, ndim=3)
    assert tracks.predecessors(2) == [1]
    assert tracks.successors(1) == [2, 3]
    assert tracks.predecessors(1) == []
    assert tracks.successors(2) == []


def test_area_methods(graph_2d):
    tracks = Tracks(graph_2d, ndim=3)
    assert tracks.get_area(1) == 1245
    assert tracks.get_areas([1, 2]) == [1245, 305]


def test_iou_methods(graph_2d):
    tracks = Tracks(graph_2d, ndim=3)
    assert tracks.get_iou((1, 2)) == 0.0
    assert tracks.get_ious([(1, 2)]) == [0.0]
    assert tracks.get_ious([(1, 2), (1, 3)]) == [0.0, 0.395]


def test_get_set_node_attr(graph_2d):
    tracks = Tracks(graph_2d, ndim=3)

    tracks._set_node_attr(1, "a", 42)
    # test deprecated functions
    with pytest.warns(
        DeprecationWarning,
        match="_get_node_attr deprecated in favor of public method get_node_attr",
    ):
        assert tracks._get_node_attr(1, "a") == 42

    tracks._set_nodes_attr([1, 2], "b", [7, 8])
    with pytest.warns(
        DeprecationWarning,
        match="_get_nodes_attr deprecated in favor of public method get_nodes_attr",
    ):
        assert tracks._get_nodes_attr([1, 2], "b") == [7, 8]

    # test new functions
    assert tracks.get_node_attr(1, "a", required=True) == 42
    assert tracks.get_nodes_attr([1, 2], "b", required=True) == [7, 8]
    assert tracks.get_nodes_attr([1, 2], "b", required=False) == [7, 8]
    with pytest.raises(KeyError):
        tracks.get_node_attr(1, "not_present", required=True)
    assert tracks.get_node_attr(1, "not_present", required=False) is None
    with pytest.raises(KeyError):
        tracks.get_nodes_attr([1, 2], "not_present", required=True)
    assert all(
        x is None for x in tracks.get_nodes_attr([1, 2], "not_present", required=False)
    )

    # test array attributes
    tracks._set_node_attr(1, "array_attr", np.array([1, 2, 3]))
    tracks._set_nodes_attr((1, 2), "array_attr2", np.array(([1, 2, 3], [4, 5, 6])))


def test_get_set_edge_attr(graph_2d):
    tracks = Tracks(graph_2d, ndim=3)
    tracks._set_edge_attr((1, 2), "c", 99)
    assert tracks.get_edge_attr((1, 2), "c") == 99
    assert tracks.get_edge_attr((1, 2), "iou", required=True) == 0.0
    tracks._set_edges_attr([(1, 2), (1, 3)], "d", [123, 5])
    assert tracks.get_edges_attr([(1, 2), (1, 3)], "d", required=True) == [123, 5]
    assert tracks.get_edges_attr([(1, 2), (1, 3)], "d", required=False) == [123, 5]
    with pytest.raises(KeyError):
        tracks.get_edge_attr((1, 2), "not_present", required=True)
    assert tracks.get_edge_attr((1, 2), "not_present", required=False) is None
    with pytest.raises(KeyError):
        tracks.get_edges_attr([(1, 2), (1, 3)], "not_present", required=True)
    assert all(
        x is None
        for x in tracks.get_edges_attr(((1, 2), (1, 3)), "not_present", required=False)
    )


def test_set_positions_str(graph_2d):
    tracks = Tracks(graph_2d, ndim=3)
    tracks.set_positions((1, 2), [(1, 2), (3, 4)])
    assert np.array_equal(
        tracks.get_positions((1, 2), incl_time=False), np.array([[1, 2], [3, 4]])
    )
    assert np.array_equal(
        tracks.get_positions((1, 2), incl_time=True), np.array([[0, 1, 2], [1, 3, 4]])
    )

    # test invalid node id
    with pytest.raises(KeyError):
        tracks.get_positions(["0"])


def test_set_positions_list(graph_2d_list):
    tracks = Tracks(graph_2d_list, pos_attr=["y", "x"], ndim=3)
    tracks.set_positions((1, 2), [(1, 2), (3, 4)])
    assert np.array_equal(
        tracks.get_positions((1, 2), incl_time=False), np.array([[1, 2], [3, 4]])
    )
    assert np.array_equal(
        tracks.get_positions((1, 2), incl_time=True), np.array([[0, 1, 2], [1, 3, 4]])
    )


def test_set_node_attributes(graph_2d, caplog):
    tracks = Tracks(graph_2d, ndim=3)
    attrs = {"attr_1": 1, "attr_2": ["a", "b", "c", "d", "e", "f"]}
    tracks._set_node_attributes(1, attrs)
    assert tracks.get_node_attr(1, "attr_1") == 1
    assert tracks.get_node_attr(1, "attr_2") == ["a", "b", "c", "d", "e", "f"]
    with caplog.at_level("INFO"):
        tracks._set_node_attributes(7, attrs)
    assert any("Node 7 not found in the graph." in message for message in caplog.messages)


def test_set_edge_attributes(graph_2d, caplog):
    tracks = Tracks(graph_2d, ndim=3)
    attrs = {"attr_1": 1, "attr_2": ["a", "b", "c", "d"]}
    tracks._set_edge_attributes((1, 2), attrs)
    assert tracks.get_edge_attr((1, 2), "attr_1") == 1
    assert tracks.get_edge_attr((1, 2), "attr_2") == ["a", "b", "c", "d"]
    with caplog.at_level("INFO"):
        tracks._set_edge_attributes((4, 6), attrs)
    assert any(
        "Edge (4, 6) not found in the graph." in message for message in caplog.messages
    )


def test_compute_node_attrs(graph_2d, segmentation_2d):
    tracks = Tracks(graph_2d, segmentation=segmentation_2d, ndim=3, scale=(1, 2, 2))
    attrs = tracks._compute_node_attrs(1, 0)
    assert NodeAttr.POS.value in attrs
    assert NodeAttr.AREA.value in attrs
    assert attrs[NodeAttr.AREA.value] == 1245 * 4
    attrs = tracks._compute_node_attrs(2, 1)
    assert attrs[NodeAttr.AREA.value] == 305 * 4

    # cannot compute node attributes without segmentation
    tracks = Tracks(graph_2d, segmentation=None, ndim=3)
    attrs = tracks._compute_node_attrs(1, 0)
    assert not bool(attrs)


def test_compute_edge_attrs(graph_2d, segmentation_2d):
    tracks = Tracks(graph_2d, segmentation_2d, ndim=3)
    attrs = tracks._compute_edge_attrs((1, 2))
    assert EdgeAttr.IOU.value in attrs
    assert attrs[EdgeAttr.IOU.value] == 0.0
    attrs = tracks._compute_edge_attrs((1, 3))
    assert np.isclose(attrs[EdgeAttr.IOU.value], 0.395, rtol=1e-2)

    # cannot compute IOU without segmentation
    tracks = Tracks(graph_2d, segmentation=None, ndim=3)
    attrs = tracks._compute_edge_attrs((1, 2))
    assert not bool(attrs)


def test_get_pixels_and_set_pixels(graph_2d, segmentation_2d):
    tracks = Tracks(graph_2d, segmentation_2d, ndim=3)
    pix = tracks.get_pixels(1)
    assert isinstance(pix, tuple)
    tracks.set_pixels(pix, 99)
    assert tracks.segmentation[0, 50, 50] == 99


def test_get_pixels_none(graph_2d):
    tracks = Tracks(graph_2d, segmentation=None, ndim=3)
    assert tracks.get_pixels([1]) is None


def test_set_pixels_no_segmentation(graph_2d):
    tracks = Tracks(graph_2d, segmentation=None, ndim=3)
    pix = [(np.array([0]), np.array([10]), np.array([20]))]
    with pytest.raises(ValueError):
        tracks.set_pixels(pix, [1])


def test_compute_ndim_errors():
    g = nx.DiGraph()
    g.add_node(1, time=0, pos=[0, 0, 0])
    # seg ndim = 3, scale ndim = 2, provided ndim = 4 -> mismatch
    seg = np.zeros((2, 2, 2))
    with pytest.raises(ValueError, match="Dimensions from segmentation"):
        Tracks(g, segmentation=seg, scale=[1, 2], ndim=4)

    with pytest.raises(
        ValueError, match="Cannot compute dimensions from segmentation or scale"
    ):
        Tracks(g)
