import numpy as np
from numpy.testing import assert_array_almost_equal

from funtracks.actions import (
    UpdateNodeSeg,
)
from funtracks.data_model import SolutionTracks
from funtracks.data_model.graph_attributes import NodeAttr


def test_update_node_segs(segmentation_2d, graph_2d):
    tracks = SolutionTracks(graph_2d.copy(), segmentation=segmentation_2d.copy())

    # add a couple pixels to the first node
    new_seg = segmentation_2d.copy()
    new_seg[0][0] = 1
    node = 1

    pixels = np.nonzero(segmentation_2d != new_seg)
    action = UpdateNodeSeg(tracks, node, pixels=pixels, added=True)

    assert set(tracks.graph.nodes()) == set(graph_2d.nodes())
    assert tracks.graph.nodes[1][NodeAttr.AREA.value] == 1345
    assert (
        tracks.graph.nodes[1][NodeAttr.POS.value] != graph_2d.nodes[1][NodeAttr.POS.value]
    )
    assert_array_almost_equal(tracks.segmentation, new_seg)

    inverse = action.inverse()
    assert set(tracks.graph.nodes()) == set(graph_2d.nodes())
    for node, data in tracks.graph.nodes(data=True):
        assert data == graph_2d.nodes[node]
    assert_array_almost_equal(tracks.segmentation, segmentation_2d)

    inverse.inverse()

    assert set(tracks.graph.nodes()) == set(graph_2d.nodes())
    assert tracks.graph.nodes[1][NodeAttr.AREA.value] == 1345
    assert (
        tracks.graph.nodes[1][NodeAttr.POS.value] != graph_2d.nodes[1][NodeAttr.POS.value]
    )
    assert_array_almost_equal(tracks.segmentation, new_seg)
