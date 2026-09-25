"""Benchmarks for CSV import.

tracks_from_df is the entry point tested here, fed a DataFrame read back from a real
CSV file (rather than built in memory) to include pandas' own CSV parsing cost. CSV
import is graph-only -- there is no segmentation or mask/bbox data in the export, so
name mapping is auto-inferred by fuzzy matching over the plain t/y/x/id/parent_id/
tracklet_id columns export_to_csv writes.
"""

import platform

import pandas as pd
import pytest

from funtracks.import_export import export_to_csv, tracks_from_df

from ._graph_builders import make_tracks

NUM_FRAMES = 20
FRAME_SHAPE = (500, 500)
CELLS_PER_FRAME = 50

ROUNDS = 3
if platform.system() == "Darwin":
    ROUNDS = ROUNDS * 2


@pytest.fixture(scope="module")
def csv_path(tmp_path_factory, _warm_jit):
    """A CSV export of a synthetic Tracks, written once and read by every round."""
    tracks = make_tracks(
        n_frames=NUM_FRAMES,
        cells_per_frame=CELLS_PER_FRAME,
        frame_shape=FRAME_SHAPE,
    )
    out = tmp_path_factory.mktemp("csv") / "tracks.csv"
    export_to_csv(tracks, out)
    return out


def test_import_from_csv(benchmark, csv_path):
    def run():
        df = pd.read_csv(csv_path)
        tracks_from_df(df)

    benchmark.pedantic(run, rounds=ROUNDS, iterations=1)
