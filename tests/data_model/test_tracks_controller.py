import numpy as np

from funtracks.data_model.graph_attributes import NodeAttr
from funtracks.data_model.solution_tracks import SolutionTracks
from funtracks.data_model.tracks_controller import TracksController


def test__add_nodes_no_seg(graph_2d):
    # add without segmentation
    tracks = SolutionTracks(graph_2d, ndim=3)
    controller = TracksController(tracks)

    num_edges = tracks.graph.number_of_edges()

    # start a new track with multiple nodes
    attrs = {
        NodeAttr.TIME.value: [0, 1],
        NodeAttr.POS.value: np.array([[1, 3], [1, 3]]),
        NodeAttr.TRACK_ID.value: [6, 6],
    }

    action, node_ids = controller._add_nodes(attrs)

    node = node_ids[0]
    assert tracks.graph.has_node(node)
    assert tracks.get_position(node) == [1, 3]
    assert tracks.get_track_id(node) == 6

    assert tracks.graph.number_of_edges() == num_edges + 1  # one edge added

    # add nodes to end of existing track
    attrs = {
        NodeAttr.TIME.value: [2, 3],
        NodeAttr.POS.value: np.array([[1, 3], [1, 3]]),
        NodeAttr.TRACK_ID.value: [2, 2],
    }

    action, node_ids = controller._add_nodes(attrs)

    node1 = node_ids[0]
    node2 = node_ids[1]
    assert tracks.get_position(node1) == [1, 3]
    assert tracks.get_track_id(node1) == 2
    assert tracks.graph.has_edge(2, node1)
    assert tracks.graph.has_edge(node1, node2)

    # add node to middle of existing track
    attrs = {
        NodeAttr.TIME.value: [3],
        NodeAttr.POS.value: np.array([[1, 3]]),
        NodeAttr.TRACK_ID.value: [3],
    }

    action, node_ids = controller._add_nodes(attrs)

    node = node_ids[0]
    assert tracks.get_position(node) == [1, 3]
    assert tracks.get_track_id(node) == 3

    assert tracks.graph.has_edge(4, node)
    assert tracks.graph.has_edge(node, 5)
    assert not tracks.graph.has_edge(4, 5)


def test__add_nodes_with_seg(graph_2d, segmentation_2d):
    # add with segmentation
    tracks = SolutionTracks(graph_2d, segmentation=segmentation_2d)
    controller = TracksController(tracks)

    num_edges = tracks.graph.number_of_edges()

    new_seg = segmentation_2d.copy()
    time = 0
    track_id = 6
    node1 = 7
    node2 = 8
    new_seg[time : time + 1, 90:100, 0:4] = node1
    new_seg[time + 1 : time + 2, 90:100, 0:4] = node2
    expected_center = [94.5, 1.5]
    # start a new track
    attrs = {
        NodeAttr.TIME.value: [time, time + 1],
        NodeAttr.TRACK_ID.value: [track_id, track_id],
        "node_id": [node1, node2],
    }

    loc_pix = np.where(new_seg[time] == node1)
    time_pix = np.ones_like(loc_pix[0]) * time
    time_pix2 = np.ones_like(loc_pix[0]) * (time + 1)
    pixels = [
        (time_pix, *loc_pix),
        (time_pix2, *loc_pix),
    ]  # TODO: get time from pixels?

    action, node_ids = controller._add_nodes(attrs, pixels=pixels)

    node1, node2 = node_ids
    assert tracks.get_time(node1) == 0
    assert tracks.get_position(node1) == expected_center
    assert tracks.get_track_id(node1) == 6
    assert tracks.get_time(node2) == 1
    assert tracks.get_position(node2) == expected_center
    assert tracks.get_track_id(node2) == 6
    assert np.sum(tracks.segmentation != new_seg) == 0

    assert tracks.graph.number_of_edges() == num_edges + 1  # one edge added

    # add nodes to end of existing track
    time = 2
    track_id = 2
    node1 = 9
    node2 = 10
    new_seg[time : time + 1, 0:10, 0:4] = node1
    new_seg[time + 1 : time + 2, 0:10, 0:4] = node2
    expected_center = [4.5, 1.5]
    # start a new track
    attrs = {
        NodeAttr.TIME.value: [time, time + 1],
        NodeAttr.TRACK_ID.value: [track_id, track_id],
        "node_id": [node1, node2],
    }

    loc_pix = np.where(new_seg[time] == node1)
    time_pix = np.ones_like(loc_pix[0]) * time
    time_pix2 = np.ones_like(loc_pix[0]) * (time + 1)
    pixels = [(time_pix, *loc_pix), (time_pix2, *loc_pix)]

    _, node_ids = controller._add_nodes(attrs, pixels)

    node = node_ids[0]
    assert tracks.get_position(node) == expected_center
    assert tracks.get_track_id(node) == 2
    assert np.sum(tracks.segmentation != new_seg) == 0

    assert tracks.graph.has_edge(2, node)
    assert tracks.graph.has_edge(node, node_ids[1])

    # add node to middle of existing track
    time = 3
    track_id = 3
    node1 = 11
    new_seg[time, 0:10, 0:4] = node1
    expected_center = [4.5, 1.5]
    attrs = {
        NodeAttr.TIME.value: [time],
        NodeAttr.TRACK_ID.value: [track_id],
        "node_id": [node1],
    }

    loc_pix = np.where(new_seg[time] == node1)
    time_pix = np.ones_like(loc_pix[0]) * time
    pixels = [(time_pix, *loc_pix)]

    action, node_ids = controller._add_nodes(attrs, pixels=pixels)

    node = node_ids[0]
    assert tracks.get_position(node) == expected_center
    assert tracks.get_track_id(node) == 3
    assert np.sum(tracks.segmentation != new_seg) == 0

    assert tracks.graph.has_edge(4, node)
    assert tracks.graph.has_edge(node, 5)
    assert not tracks.graph.has_edge(4, 5)


def test__delete_nodes_no_seg(graph_2d):
    tracks = SolutionTracks(graph_2d, ndim=3)
    controller = TracksController(tracks)
    num_edges = tracks.graph.number_of_edges()

    # delete unconnected node
    node = 6
    action = controller._delete_nodes([node])
    assert not tracks.graph.has_node(node)
    assert tracks.graph.number_of_edges() == num_edges
    action.inverse()

    # delete end node
    node = 5
    action = controller._delete_nodes([node])
    assert not tracks.graph.has_node(node)
    assert not tracks.graph.has_edge(4, node)
    action.inverse()

    # delete continuation node
    node = 4
    action = controller._delete_nodes([node])
    assert not tracks.graph.has_node(node)
    assert not tracks.graph.has_edge(3, node)
    assert not tracks.graph.has_edge(node, 5)
    assert tracks.graph.has_edge(3, 5)
    assert tracks.get_track_id(5) == 3
    action.inverse()

    # delete div parent
    node = 1
    action = controller._delete_nodes([node])
    assert not tracks.graph.has_node(node)
    assert not tracks.graph.has_edge(node, 2)
    assert not tracks.graph.has_edge(node, 3)
    action.inverse()

    # delete div child
    node = 3
    action = controller._delete_nodes([node])
    assert not tracks.graph.has_node(node)
    assert tracks.get_track_id(2) == 1  # update track id for other child


def test__delete_nodes_with_seg(graph_2d, segmentation_2d):
    tracks = SolutionTracks(graph_2d, segmentation=segmentation_2d)
    controller = TracksController(tracks)
    num_edges = tracks.graph.number_of_edges()

    # delete unconnected node
    node = 6
    track_id = 6
    time = 4
    action = controller._delete_nodes([node])
    assert not tracks.graph.has_node(node)
    assert track_id not in np.unique(tracks.segmentation[time])
    assert tracks.graph.number_of_edges() == num_edges
    action.inverse()

    # delete end node
    node = 5
    track_id = 3
    time = 4
    action = controller._delete_nodes([node])
    assert not tracks.graph.has_node(node)
    assert track_id not in np.unique(tracks.segmentation[time])
    assert not tracks.graph.has_edge(4, node)
    action.inverse()

    # delete continuation node
    node = 4
    track_id = 3
    time = 2
    action = controller._delete_nodes([node])
    assert not tracks.graph.has_node(node)
    assert track_id not in np.unique(tracks.segmentation[time])
    assert not tracks.graph.has_edge(3, node)
    assert not tracks.graph.has_edge(node, 5)
    assert tracks.graph.has_edge(3, 5)
    assert tracks.get_track_id(5) == 3
    action.inverse()

    # delete div parent
    node = 1
    track_id = 1
    time = 0
    action = controller._delete_nodes([node])
    assert not tracks.graph.has_node(node)
    assert track_id not in np.unique(tracks.segmentation[time])
    assert not tracks.graph.has_edge(node, 2)
    assert not tracks.graph.has_edge(node, 3)
    action.inverse()

    # delete div child
    node = 2
    track_id = 2
    time = 1
    action = controller._delete_nodes([node])
    assert not tracks.graph.has_node(node)
    assert track_id not in np.unique(tracks.segmentation[time])
    assert tracks.get_track_id(3) == 1  # update track id for other child
    assert tracks.get_track_id(5) == 1  # update track id for other child


def test__add_remove_edges_no_seg(graph_2d):
    tracks = SolutionTracks(graph_2d, ndim=3)
    controller = TracksController(tracks)
    num_edges = tracks.graph.number_of_edges()

    # delete continuation edge
    edge = (3, 4)
    track_id = 3
    controller._delete_edges([edge])
    assert not tracks.graph.has_edge(*edge)
    assert tracks.get_track_id(edge[1]) != track_id  # relabeled the rest of the track
    assert tracks.graph.number_of_edges() == num_edges - 1

    # add back in continuation edge
    controller._add_edges([edge])
    assert tracks.graph.has_edge(*edge)
    assert tracks.get_track_id(edge[1]) == track_id  # track id was changed back
    assert tracks.graph.number_of_edges() == num_edges

    # delete division edge
    edge = (1, 3)
    track_id = 3
    controller._delete_edges([edge])
    assert not tracks.graph.has_edge(*edge)
    assert tracks.get_track_id(edge[1]) == track_id  # dont relabel after removal
    assert tracks.get_track_id(2) == 1  # but do relabel the sibling
    assert tracks.graph.number_of_edges() == num_edges - 1

    # add back in division edge
    edge = (1, 3)
    track_id = 3
    controller._add_edges([edge])
    assert tracks.graph.has_edge(*edge)
    assert tracks.get_track_id(edge[1]) == track_id  # dont relabel after removal
    assert tracks.get_track_id(2) != 1  # give sibling new id again (not necessarily 2)
    assert tracks.graph.number_of_edges() == num_edges
