"""Benchmarks for Tracks construction from an already-built graph.

Separate from test_graph_to_solution in bench_candidate_graph.py, which measures
construction from a *candidate* graph (multi-successor, produced by
compute_graph_from_seg). This module measures construction from a graph that is
already a valid tracking solution (one successor per node, the shape user_actions and
import/export expect), which is the path GEFF import and the local-file `funtracks`
usage of Tracks() takes.
"""

import platform

import pytest

from funtracks.data_model import Tracks

from ._graph_builders import make_tracks

NUM_FRAMES = 20
FRAME_SHAPE = (500, 500)
CELLS_PER_FRAME = 50

ROUNDS = 3
if platform.system() == "Darwin":
    ROUNDS = ROUNDS * 2


@pytest.fixture(scope="module")
def built_graph(_warm_jit):
    """A solution graph, pre-built and detached from any Tracks object.

    make_tracks both builds the graph and wraps it in Tracks (which is what's under
    test), so the graph is built once here via a throwaway Tracks and its underlying
    tracksdata graph is reused as construction input below -- constructing that graph
    is not part of what these benchmarks measure.
    """
    return make_tracks(
        n_frames=NUM_FRAMES,
        cells_per_frame=CELLS_PER_FRAME,
        frame_shape=FRAME_SHAPE,
    ).graph_full


def test_construct_with_track_ids(benchmark, built_graph):
    """Tracks() on a pre-built solution graph, with tracklet/lineage assignment.

    Setting tracklet_attr/lineage_attr registers a TrackAnnotator, which computes
    tracklet and lineage ids on construction -- this is the dominant cost over the
    no-track-ids case below.
    """

    def run():
        Tracks(
            built_graph,
            time_attr="t",
            pos_attr="pos",
            tracklet_attr="tracklet_id",
            lineage_attr="lineage_id",
        )

    benchmark.pedantic(run, rounds=ROUNDS, iterations=1)


def test_construct_without_track_ids(benchmark, built_graph):
    """Tracks() on a pre-built solution graph, without track-id assignment."""

    def run():
        Tracks(built_graph, time_attr="t", pos_attr="pos")

    benchmark.pedantic(run, rounds=ROUNDS, iterations=1)
