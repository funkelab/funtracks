import pytest
from numpy.testing import assert_array_almost_equal
from polars.testing import assert_frame_equal

from funtracks.data_model import NodeAttr, Tracks
from funtracks.data_model.utils import td_get_single_attr_from_node


def test_create_tracks(graph_3d, segmentation_3d):
    # create tracks with graph only
    tracks = Tracks(graph=graph_3d.copy(), ndim=4)
    assert tracks.get_positions([1]).tolist() == [[50, 50, 50]]
    assert tracks.get_time(1) == 0
    with pytest.raises(KeyError):
        tracks.get_positions(["0"])

    # create track with graph and seg
    tracks = Tracks(graph=graph_3d.copy(), segmentation=segmentation_3d.copy())
    assert tracks.get_positions([1]).tolist() == [[50, 50, 50]]
    assert tracks.get_time(1) == 0
    assert tracks.get_positions([1], incl_time=True).tolist() == [[0, 50, 50, 50]]
    tracks.set_time(1, 1)
    assert tracks.get_positions([1], incl_time=True).tolist() == [[1, 50, 50, 50]]

    tracks_wrong_attr = Tracks(
        graph=graph_3d.copy(),
        segmentation=segmentation_3d.copy(),
        time_attr="test",
    )
    with pytest.raises(KeyError):  # raises error at access if time is wrong
        tracks_wrong_attr.get_times([1])

    tracks_wrong_attr = Tracks(graph=graph_3d.copy(), pos_attr="test", ndim=3)
    with pytest.raises(KeyError):  # raises error at access if pos is wrong
        tracks_wrong_attr.get_positions([1])

    # test multiple position attrs
    pos_attr = ("z", "y", "x")
    graph_3d_copy = graph_3d.copy()
    graph_3d_copy.add_node_attr_key(key="z", default_value=0)
    graph_3d_copy.add_node_attr_key(key="y", default_value=0)
    graph_3d_copy.add_node_attr_key(key="x", default_value=0)
    for node in graph_3d_copy.node_ids():
        pos = td_get_single_attr_from_node(
            graph_3d_copy, node_ids=[node], attrs=[NodeAttr.POS.value]
        )
        z, y, x = pos
        # del graph_3d.nodes[node][NodeAttr.POS.value]
        graph_3d_copy.update_node_attrs(attrs={"z": z, "y": y, "x": x}, node_ids=[node])
    # remove node attr pos

    tracks = Tracks(graph=graph_3d_copy, pos_attr=pos_attr, ndim=4)
    assert tracks.get_positions([1]).tolist() == [[50, 50, 50]]
    tracks.set_position(1, [55, 56, 57])
    assert tracks.get_position(1) == [55, 56, 57]

    tracks.set_position(1, [1, 50, 50, 50], incl_time=True)
    assert tracks.get_time(1) == 1


def test_pixels_and_seg_id(graph_3d, segmentation_3d):
    # create track with graph and seg
    tracks = Tracks(graph=graph_3d.copy(), segmentation=segmentation_3d.copy())

    # changing a segmentation id changes it in the mapping
    pix = tracks.get_pixels([1])
    new_seg_id = 10
    tracks.set_pixels(pix, [new_seg_id])

    with pytest.raises(KeyError):
        tracks.get_positions(["0"])


def test_save_load_delete(tmp_path, graph_2d, segmentation_2d):
    tracks_dir = tmp_path / "tracks"
    tracks = Tracks(graph=graph_2d.copy(), segmentation=segmentation_2d.copy())
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
        assert_frame_equal(
            loaded.graph.node_attrs(), tracks.graph.node_attrs(), check_column_order=False
        )
        assert_frame_equal(
            loaded.graph.edge_attrs(), tracks.graph.edge_attrs(), check_column_order=False
        )
        assert_array_almost_equal(loaded.segmentation, tracks.segmentation)
    with pytest.warns(
        DeprecationWarning,
        match="`Tracks.delete` is deprecated and will be removed in 2.0",
    ):
        Tracks.delete(tracks_dir)
