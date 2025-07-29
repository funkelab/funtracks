import pytest
from networkx.utils import graphs_equal
from numpy.testing import assert_array_almost_equal

from funtracks.data_model import NodeAttr, Tracks


def test_create_tracks(graph_3d, segmentation_3d):
    # create empty tracks
    # tracks = Tracks(graph=nx.DiGraph(), ndim=3)
    # with pytest.raises(KeyError):
    #     tracks.get_positions([1])

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
    for node in graph_3d.node_ids():
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
    pix = tracks.get_pixels([1])
    new_seg_id = 10
    tracks.set_pixels(pix, [new_seg_id])

    with pytest.raises(KeyError):
        tracks.get_positions(["0"])


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
