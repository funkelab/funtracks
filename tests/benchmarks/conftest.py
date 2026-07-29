"""Shared fixtures for the benchmark suite."""

import pytest

from funtracks.user_actions import UserUpdateNodeAttrs, UserUpdateSegmentation
from funtracks.utils.tracksdata_utils import td_mask_to_pixels

from ._graph_builders import make_solution_tracks


@pytest.fixture(scope="session")
def _warm_jit():
    """Pay the one-time JIT compile costs before any timed benchmark.

    Several lazy compiles would otherwise land inside whichever timed region hits them
    first, and they are large enough to skew or dwarf the measurement:

    1. Building a SolutionTracks with segmentation creates a GraphArrayView, which builds
       a spatial_graph rtree whose Cython module is JIT-compiled by witty on first use
       (seconds-to-tens-of-seconds on a cold Windows runner).
    2. UserUpdateSegmentation calls Mask.intersection to decide whether a paint stroke
       fully covers a node, which lands in tracksdata's numba kernels
       (``@njit fast_intersection_with_bbox`` / ``fast_iou_with_bbox`` in
       tracksdata/functional/_iou.py). Nothing in funtracks itself is numba-compiled.
       First call ~450ms against ~0.5ms steady state -- an 800x first-call penalty.

    Both caches are keyed on types, not sizes: numba specialises per argument type
    signature and witty per generated source. That is what makes a tiny warm-up graph
    sufficient -- it installs the ``(int64[:], int64[:], bool[:,:], bool[:,:])`` signature
    that the full-size benchmarks then reuse. It also means the warm-up only works while
    it shares the real graph's dtypes, so make_solution_tracks is the single source of
    geometry for both; do not hand-roll a graph for this fixture.
    """
    tracks = make_solution_tracks(n_frames=2, cells_per_frame=2, frame_shape=(64, 64))

    # Exercise the update-segmentation path too. Building the tracks is not enough: the
    # numba compile happens on the first Mask.intersection call, not during construction.
    node = next(iter(sorted(int(n) for n in tracks.graph.node_ids())))
    mask_pixels = td_mask_to_pixels(
        tracks.get_mask(node), tracks.get_time(node), ndim=tracks.ndim
    )
    n_patch = max(1, len(mask_pixels[0]) // 3)
    patch = tuple(dim_pixels[:n_patch] for dim_pixels in mask_pixels)
    UserUpdateSegmentation(
        tracks, new_value=0, updated_pixels=[(patch, node)], current_track_id=1
    )

    # Same story, smaller magnitude (~2x rather than ~800x): the first attribute update
    # is slower than steady state, enough to skew a 3-round benchmark.
    UserUpdateNodeAttrs(tracks, node, {"score": 1.0})
