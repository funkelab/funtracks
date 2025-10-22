import copy

import networkx as nx
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from funtracks.actions import (
    ActionGroup,
    AddNode,
)


@pytest.mark.parametrize("ndim", [3, 4])
@pytest.mark.parametrize("with_seg", [True, False])
def test_add_delete_nodes(get_tracks, ndim, with_seg):
    # Get a tracks instance
    tracks = get_tracks(ndim=ndim, with_seg=with_seg, is_solution=True)
    reference_graph = tracks.graph
    reference_seg = copy.deepcopy(tracks.segmentation)

    # Start with an empty Tracks
    empty_graph = nx.DiGraph()
    empty_seg = np.zeros_like(tracks.segmentation) if with_seg else None
    tracks.graph = empty_graph
    if with_seg:
        tracks.segmentation = empty_seg

    nodes = list(reference_graph.nodes())
    actions = []
    for node in nodes:
        pixels = np.nonzero(reference_seg == node) if with_seg else None
        actions.append(
            AddNode(tracks, node, dict(reference_graph.nodes[node]), pixels=pixels)
        )
    action = ActionGroup(tracks=tracks, actions=actions)

    assert set(tracks.graph.nodes()) == set(reference_graph.nodes())
    for node, data in tracks.graph.nodes(data=True):
        reference_data = reference_graph.nodes[node]
        assert data == reference_data
    if with_seg:
        assert_array_almost_equal(tracks.segmentation, reference_seg)

    # Invert the action to delete all the nodes
    del_nodes = action.inverse()
    assert set(tracks.graph.nodes()) == set(empty_graph.nodes())
    if with_seg:
        assert_array_almost_equal(tracks.segmentation, empty_seg)

    # Re-invert the action to add back all the nodes and their attributes
    del_nodes.inverse()
    assert set(tracks.graph.nodes()) == set(reference_graph.nodes())
    for node, data in tracks.graph.nodes(data=True):
        reference_data = copy.deepcopy(reference_graph.nodes[node])
        # TODO: get back custom attrs https://github.com/funkelab/funtracks/issues/1
        if not with_seg and "area" in reference_data:
            del reference_data["area"]
        assert data == reference_data
    if with_seg:
        assert_array_almost_equal(tracks.segmentation, reference_seg)


def test_add_node_missing_time(get_tracks):
    tracks = get_tracks(ndim=3, with_seg=True, is_solution=True)
    with pytest.raises(ValueError, match="Must provide a time attribute for node"):
        AddNode(tracks, 8, {})


def test_add_node_missing_pos(get_tracks):
    tracks = get_tracks(ndim=3, with_seg=True, is_solution=True)
    # First test: missing track_id raises an error
    with pytest.raises(ValueError, match="Must provide a track_id attribute for node"):
        AddNode(tracks, 8, {"time": 2})

    # Second test: with track_id but without segmentation, missing pos raises an error
    tracks_no_seg = get_tracks(ndim=3, with_seg=False, is_solution=True)
    with pytest.raises(
        ValueError, match="Must provide position or segmentation for node"
    ):
        AddNode(tracks_no_seg, 8, {"time": 2, "track_id": 1})
