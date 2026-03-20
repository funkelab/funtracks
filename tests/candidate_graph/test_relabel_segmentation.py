import numpy as np
import pytest
from numpy.testing import assert_array_equal
from skimage.draw import disk

from funtracks.utils import ensure_unique_labels, relabel_segmentation_with_track_id


@pytest.fixture
def segmentation_2d_repeat_labels():
    frame_shape = (100, 100)
    total_shape = (2, *frame_shape)
    segmentation = np.zeros(total_shape, dtype="int32")
    # make frame with one cell in center with label 1
    rr, cc = disk(center=(50, 50), radius=20, shape=(100, 100))
    segmentation[0][rr, cc] = 1

    # make frame with two cells
    # first cell centered at (20, 80) with label 1
    # second cell centered at (60, 45) with label 2
    rr, cc = disk(center=(20, 80), radius=10, shape=frame_shape)
    segmentation[1][rr, cc] = 1
    rr, cc = disk(center=(60, 45), radius=15, shape=frame_shape)
    segmentation[1][rr, cc] = 2
    return segmentation


def test_relabel_segmentation(get_tracks):
    tracks = get_tracks(ndim=3, with_seg=True)
    segmentation = np.asarray(tracks.segmentation)

    # Use only nodes 1 and 2 (single tracklet: node 1 at t=0, node 2 at t=1)
    subgraph = tracks.graph.filter(node_ids=[1, 2]).subgraph()
    relabeled = relabel_segmentation_with_track_id(subgraph, segmentation)

    # Nodes 1 and 2 form one tracklet → both get label 1
    assert (relabeled[0][segmentation[0] == 1] == 1).all()
    assert (relabeled[1][segmentation[1] == 2] == 1).all()
    # Node 3 not in subgraph → pixels become 0
    assert (relabeled[1][segmentation[1] == 3] == 0).all()


def test_ensure_unique_labels_2d(segmentation_2d_repeat_labels):
    expected = segmentation_2d_repeat_labels.copy().astype(np.uint64)
    frame = expected[1]
    frame[frame == 2] = 3
    frame[frame == 1] = 2
    expected[1] = frame

    print(np.unique(expected[1], return_counts=True))  # noqa
    result = ensure_unique_labels(segmentation_2d_repeat_labels)
    assert_array_equal(expected, result)
