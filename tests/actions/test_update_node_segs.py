import copy

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from polars.testing import assert_series_not_equal

from funtracks.actions import (
    UpdateNodeSeg,
)


@pytest.mark.parametrize("ndim", [3, 4])
def test_update_node_segs(get_tracks, ndim):
    # Get tracks with segmentation
    tracks = get_tracks(ndim=ndim, with_seg=True, is_solution=True)
    reference_graph = copy.deepcopy(tracks.graph)

    original_seg = tracks.segmentation.copy()
    original_area = tracks.graph[1]["area"]
    original_pos = tracks.graph[1]["pos"]

    # Add a couple pixels to the first node
    new_seg = tracks.segmentation.copy()
    if ndim == 3:
        new_seg[0][0][0] = 1  # 2D spatial
    else:
        new_seg[0][0][0][0] = 1  # 3D spatial
    node = 1

    pixels = np.nonzero(original_seg != new_seg)
    action = UpdateNodeSeg(tracks, node, pixels=pixels, added=True)

    assert set(tracks.graph.node_ids()) == set(reference_graph.node_ids())
    assert tracks.graph[1]["area"] == original_area + 1
    assert not np.allclose(tracks.graph[1]["pos"], original_pos)
    assert_array_almost_equal(tracks.segmentation, new_seg)

    inverse = action.inverse()
    assert set(tracks.graph.node_ids()) == set(reference_graph.node_ids())
    assert_series_not_equal(
        reference_graph[1]["pos"],
        tracks.graph[1]["pos"],
    )
    assert_array_almost_equal(tracks.segmentation, original_seg)

    inverse.inverse()

    assert set(tracks.graph.node_ids()) == set(reference_graph.node_ids())
    assert tracks.graph[1]["area"] == original_area + 1
    assert not np.allclose(tracks.graph[1]["pos"], original_pos)
    assert_array_almost_equal(tracks.segmentation, new_seg)
