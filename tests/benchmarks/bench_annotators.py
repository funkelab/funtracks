"""Benchmarks for annotator bulk-compute paths.

Both RegionpropsAnnotator.compute() and EdgeAnnotator.compute() are the bulk
recompute-everything path (as opposed to the single-node/edge update() path
exercised incidentally by test_update_segmentation, test_add_delete_node, and
test_add_delete_edges in bench_actions.py). Neither feature is active by default --
_graph_builders.make_tracks writes raw "pos"/"iou" values into the graph at build
time, but that is plain data, not a registered feature the annotators track -- so
both benchmarks enable their annotator's feature(s) first (recompute=False, so
enabling itself isn't what's timed) and then measure compute().
"""

import platform

import numpy as np
import pytest

from funtracks.annotators import EdgeAnnotator, RegionpropsAnnotator

from ._graph_builders import make_tracks

NUM_FRAMES = 20
FRAME_SHAPE = (500, 500)
CELLS_PER_FRAME = 50

ROUNDS = 3
if platform.system() == "Darwin":
    ROUNDS = ROUNDS * 2


@pytest.fixture(scope="module")
def tracks(_warm_jit):
    return make_tracks(
        n_frames=NUM_FRAMES,
        cells_per_frame=CELLS_PER_FRAME,
        frame_shape=FRAME_SHAPE,
    )


@pytest.fixture(scope="module")
def regionprops_annotator(tracks):
    """The Tracks' own RegionpropsAnnotator, with every feature enabled.

    Enabling via tracks.enable_features(..., recompute=False) registers the columns
    without paying for a compute -- the benchmark times compute() itself, not this
    one-time setup. An intensity image is attached first so intensity is included
    rather than skipped-with-a-warning, matching what get_available_features
    advertises as the full feature set.
    """
    (rp_ann,) = (a for a in tracks.annotators if isinstance(a, RegionpropsAnnotator))
    rng = np.random.default_rng(42)
    intensity_image = rng.random(tracks.segmentation.shape, dtype=np.float32)
    rp_ann.set_intensity_images([intensity_image])
    tracks.enable_features(list(rp_ann.all_features.keys()), recompute=False)
    return rp_ann


def test_regionprops_compute_all_features(benchmark, regionprops_annotator):
    """Bulk recompute of every regionprops feature (pos, area, intensity,
    ellipse_axis_radii, circularity, perimeter) across the whole graph.
    """

    def run():
        regionprops_annotator.compute()

    benchmark.pedantic(run, rounds=ROUNDS, iterations=1)


@pytest.fixture(scope="module")
def edge_annotator(tracks):
    """The Tracks' own EdgeAnnotator, with the iou feature enabled (registered, not
    computed -- same recompute=False rationale as regionprops_annotator above).
    """
    (edge_ann,) = (a for a in tracks.annotators if isinstance(a, EdgeAnnotator))
    tracks.enable_features(list(edge_ann.all_features.keys()), recompute=False)
    return edge_ann


def test_edge_compute_iou(benchmark, edge_annotator):
    """Bulk recompute of IoU across every edge in the graph."""

    def run():
        edge_annotator.compute()

    benchmark.pedantic(run, rounds=ROUNDS, iterations=1)
