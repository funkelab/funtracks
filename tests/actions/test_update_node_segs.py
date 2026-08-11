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


def test_update_node_segs_erase_interior_pixel(get_tracks):
    """Erasing a pixel *interior* to a node must clear it in the segmentation view.

    Node 1 is a disk centred at (50, 50); erasing its centre pixel does not shrink
    the bbox, so it exercises exactly the bbox-unchanged path that regressed.
    """
    tracks = get_tracks(ndim=3, with_seg=True, prefill_track_ids=True)
    node = 1
    time = tracks.get_time(node)

    original_bbox = np.asarray(tracks.graph_solution.nodes[node]["bbox"]).copy()
    original_area = tracks.graph_solution.nodes[node]["area"]
    # Precondition: the interior pixel currently belongs to the node.
    assert int(np.asarray(tracks.segmentation[time, 50, 50])) == node

    # Erase the single interior pixel (50, 50).
    erase_mask = Mask(np.ones((1, 1), dtype=bool), np.array([50, 50, 51, 51]))
    UpdateNodeSeg(tracks, node, mask=erase_mask, added=False)

    # The bbox must be unchanged (the erased pixel was interior), which is the
    # condition under which the stale-readback bug manifested.
    assert np.array_equal(
        np.asarray(tracks.graph_solution.nodes[node]["bbox"]), original_bbox
    )
    assert tracks.graph_solution.nodes[node]["area"] == original_area - 1
    # The segmentation view must reflect the erase.
    assert int(np.asarray(tracks.segmentation[time, 50, 50])) == 0
