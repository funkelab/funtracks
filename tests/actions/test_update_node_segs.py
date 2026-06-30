import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from polars.testing import assert_series_equal
from tracksdata.nodes import Mask

from funtracks.actions import UpdateNodeSeg


@pytest.mark.parametrize("ndim", [3, 4])
def test_update_node_segs(get_tracks, ndim):
    # Get tracks with segmentation
    tracks = get_tracks(ndim=ndim, with_seg=True, prefill_track_ids=True)
    reference_graph = tracks.graph_solution.detach().filter().subgraph()

    node = 1
    time = tracks.get_time(node)

    original_seg = np.asarray(tracks.segmentation).copy()
    original_area = tracks.graph_solution.nodes[1]["area"]
    original_pos = tracks.graph_solution.nodes[1]["pos"]

    # Add a couple pixels to the first node
    new_seg = np.asarray(tracks.segmentation).copy()
    if ndim == 3:
        new_seg[time][0][0] = node
        mask = Mask(np.ones((1, 1), dtype=bool), np.array([0, 0, 1, 1]))
    else:
        new_seg[time][0][0][0] = node
        mask = Mask(np.ones((1, 1, 1), dtype=bool), np.array([0, 0, 0, 1, 1, 1]))

    action = UpdateNodeSeg(tracks, node, mask=mask, added=True)

    assert set(tracks.graph_solution.node_ids()) == set(reference_graph.node_ids())
    assert tracks.graph_solution.nodes[1]["area"] == original_area + 1
    assert not np.allclose(tracks.graph_solution.nodes[1]["pos"], original_pos)
    assert_array_almost_equal(tracks.segmentation, new_seg)

    inverse = action.inverse()
    assert set(tracks.graph_solution.node_ids()) == set(reference_graph.node_ids())
    assert_series_equal(
        reference_graph.nodes[1]["pos"],
        tracks.graph_solution.nodes[1]["pos"],
    )
    assert_array_almost_equal(tracks.segmentation, original_seg)

    inverse.inverse()

    assert set(tracks.graph_solution.node_ids()) == set(reference_graph.node_ids())
    assert tracks.graph_solution.nodes[1]["area"] == original_area + 1
    assert not np.allclose(tracks.graph_solution.nodes[1]["pos"], original_pos)
    assert_array_almost_equal(tracks.segmentation, new_seg)
