"""Benchmarks for candidate graph generation using synthetic segmentation data.

To diagnose regressions, we recommend running line-profiler locally as shown in
profile_candidate_graph.py
"""

import numpy as np
import pytest
from skimage.draw import disk

from funtracks.candidate_graph.compute_graph import compute_graph_from_seg

NUM_FRAMES = 50
FRAME_SHAPE = (700, 1100)
CELLS_PER_FRAME = 150
MAX_EDGE_DISTANCE = 50.0


def _generate_segmentation(
    num_frames=NUM_FRAMES,
    frame_shape=FRAME_SHAPE,
    cells_per_frame=CELLS_PER_FRAME,
    seed=42,
):
    """Generate a synthetic segmentation array with random disks."""
    rng = np.random.default_rng(seed)
    seg = np.zeros((num_frames, *frame_shape), dtype=np.uint16)
    label = 1
    for t in range(num_frames):
        for _ in range(cells_per_frame):
            cy = rng.integers(20, frame_shape[0] - 20)
            cx = rng.integers(20, frame_shape[1] - 20)
            radius = rng.integers(10, 30)
            rr, cc = disk((cy, cx), radius, shape=frame_shape)
            seg[t, rr, cc] = label
            label += 1
    return seg


@pytest.fixture(scope="module")
def seg_data():
    return _generate_segmentation()


def test_compute_graph_from_seg(benchmark, seg_data):
    benchmark.pedantic(
        compute_graph_from_seg,
        args=(seg_data, MAX_EDGE_DISTANCE),
        kwargs={"iou": True},
        rounds=1,
        iterations=1,
    )
