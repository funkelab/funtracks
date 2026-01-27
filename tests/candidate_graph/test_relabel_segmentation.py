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


def test_relabel_segmentation(segmentation_2d, graph_2d):
    frame_shape = (100, 100)
    expected = np.zeros(segmentation_2d.shape, dtype="int32")
    # make frame with one cell in center with label 1
    rr, cc = disk(center=(50, 50), radius=20, shape=(100, 100))
    expected[0][rr, cc] = 1

    # make frame with cell centered at (20, 80) with label 1
    rr, cc = disk(center=(20, 80), radius=10, shape=frame_shape)
    expected[1][rr, cc] = 1

    graph_2d.remove_node(3)
    relabeled_seg = relabel_segmentation_with_track_id(graph_2d, segmentation_2d)
    print(f"Nonzero relabeled: {np.count_nonzero(relabeled_seg)}")  # noqa
    print(f"Nonzero expected: {np.count_nonzero(expected)}")  # noqa
    print(f"Max relabeled: {np.max(relabeled_seg)}")  # noqa
    print(f"Max expected: {np.max(expected)}")  # noqa

    assert_array_equal(relabeled_seg, expected)


def test_ensure_unique_labels_2d(segmentation_2d_repeat_labels):
    expected = segmentation_2d_repeat_labels.copy().astype(np.uint64)
    frame = expected[1]
    frame[frame == 2] = 3
    frame[frame == 1] = 2
    expected[1] = frame

    print(np.unique(expected[1], return_counts=True))  # noqa
    result = ensure_unique_labels(segmentation_2d_repeat_labels)
    assert_array_equal(expected, result)
