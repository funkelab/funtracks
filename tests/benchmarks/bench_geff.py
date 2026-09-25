"""Benchmarks for GEFF export/import and save/load.

Two distinct paths, kept in one file but grouped separately:
  - export_to_geff / import_from_geff: the user-facing round-trip (napari-track-edit's
    File > Export/Import), which wraps the graph in a parent zarr container and
    optionally writes segmentation alongside it.
  - write_to_geff: the lighter internal save-to-store variant used by the
    save/load workflow -- no parent zarr container, no segmentation. There is no
    separate load_from_geff; loading a store written by write_to_geff goes back
    through import_from_geff, so only the write side has its own benchmark here.

Segmentation on/off is benchmarked separately within export/import since
export_to_geff's cost is dominated by segmentation relabeling and writing when a
segmentation is present.
"""

import platform
from pathlib import Path

import pytest
import zarr

from funtracks.import_export import export_to_geff, import_from_geff, write_to_geff

from ._graph_builders import make_tracks

NUM_FRAMES = 20
FRAME_SHAPE = (500, 500)
CELLS_PER_FRAME = 50

ROUNDS = 3
if platform.system() == "Darwin":
    ROUNDS = ROUNDS * 2


@pytest.fixture(scope="module")
def tracks(_warm_jit):
    """make_tracks builds a Tracks with a segmentation (mask/bbox node attrs), so
    it covers the with-segmentation benchmarks; the no-segmentation benchmarks below
    reuse it with save_segmentation=False to isolate the graph-only cost.
    """
    return make_tracks(
        n_frames=NUM_FRAMES,
        cells_per_frame=CELLS_PER_FRAME,
        frame_shape=FRAME_SHAPE,
    )


# ────────────────────────────────────────────────────────────────────────
# export_to_geff / import_from_geff -- user-facing export/import round-trip
# ────────────────────────────────────────────────────────────────────────


@pytest.fixture
def exported_store(tmp_path, tracks):
    """A geff store with embedded (mask/bbox-reconstructed) segmentation, for the
    import-side benchmarks. import_from_geff takes the tracks.geff store itself, not
    its parent export directory.
    """
    out = tmp_path / "exported.zarr"
    export_to_geff(tracks, out, overwrite=True, save_segmentation=True)
    return out / "tracks.geff"


@pytest.fixture
def exported_store_no_seg(tmp_path, tracks):
    """A geff store without segmentation, for the import-side benchmarks."""
    out = tmp_path / "exported_no_seg.zarr"
    export_to_geff(tracks, out, overwrite=True, save_segmentation=False)
    return out / "tracks.geff"


def _make_foreign(store: Path) -> None:
    """Rewrite a geff store in place to look like it came from an external tool (or
    an old funtracks version), rather than funtracks' own export.

    Two changes, both required to force the fuzzy-matching import path:
      1. Remove the funtracks FeatureDict from the store's metadata. Without a
         FeatureDict, import_from_geff falls back from the direct-mapping path to
         GeffTracksBuilder.infer_node_name_map's fuzzy string matching over
         axes/property names.
      2. Remove the tracklet_id/lineage_id node properties -- both the zarr arrays
         and their entries in the geff metadata's node_props_metadata (which a reader
         consults to know which properties exist, so it has to agree with the arrays
         or GeffMetadata.read chokes on the mismatch). A funtracks export always
         writes them, but an external tool's GEFF generally would not -- so their
         absence is itself part of what makes a store "foreign", and it means
         Tracks.__init__ must recompute them via TrackAnnotator on import instead of
         reading them off the graph.

    Mirrors the metadata-surgery in tests/import_export/test_import_from_geff.py.
    """
    z = zarr.open_group(str(store), mode="r+")
    attrs = dict(z.attrs)
    geff_attrs = dict(attrs.get("geff", {}))
    extra = dict(geff_attrs.get("extra", {}))
    funtracks_extra = dict(extra.get("funtracks", {}))
    funtracks_extra.pop("features", None)
    if funtracks_extra:
        extra["funtracks"] = funtracks_extra
    else:
        extra.pop("funtracks", None)
    if extra:
        geff_attrs["extra"] = extra
    else:
        geff_attrs.pop("extra", None)

    node_props_metadata = dict(geff_attrs.get("node_props_metadata", {}))
    for key in ("tracklet_id", "lineage_id"):
        node_props_metadata.pop(key, None)
    geff_attrs["node_props_metadata"] = node_props_metadata

    attrs["geff"] = geff_attrs
    z.attrs.clear()
    z.attrs.update(attrs)

    node_props = z["nodes/props"]
    for key in ("tracklet_id", "lineage_id"):
        if key in node_props:
            del node_props[key]


@pytest.fixture
def foreign_store(exported_store):
    """A funtracks-exported store rewritten to look externally authored, so import
    takes the fuzzy-matching branch and recomputes tracklet/lineage ids -- both more
    expensive than the direct-mapping, read-off-the-graph branch a funtracks-authored
    GEFF normally gets.
    """
    _make_foreign(exported_store)
    return exported_store


def test_export_to_geff_with_segmentation(benchmark, tracks, tmp_path):
    counter = [0]

    def setup():
        counter[0] += 1
        return (), {
            "tracks": tracks,
            "directory": tmp_path / f"export_seg_{counter[0]}.zarr",
            "save_segmentation": True,
        }

    benchmark.pedantic(export_to_geff, setup=setup, rounds=ROUNDS, iterations=1)


def test_import_from_geff_with_segmentation(benchmark, exported_store):
    """Import of a funtracks-authored GEFF: a FeatureDict is present, so this takes
    the direct-mapping path (see test_import_from_foreign_geff for the fuzzy-matching
    path a FeatureDict-less GEFF falls back to).
    """

    def run():
        import_from_geff(exported_store)

    benchmark.pedantic(run, rounds=ROUNDS, iterations=1)


def test_import_from_foreign_geff(benchmark, foreign_store):
    """Import of a GEFF with no FeatureDict, as if written by an external tool (or an
    old funtracks version). GeffTracksBuilder.infer_node_name_map falls back to fuzzy
    string matching over axes/property names instead of the FeatureDict's direct
    mapping -- a materially different and more expensive code path than
    test_import_from_geff_with_segmentation above.
    """

    def run():
        import_from_geff(foreign_store)

    benchmark.pedantic(run, rounds=ROUNDS, iterations=1)


# ────────────────────────────────────────────────────────────────────────
# write_to_geff -- internal save/load path
# ────────────────────────────────────────────────────────────────────────


def test_internal_save(benchmark, tracks, tmp_path):
    """write_to_geff writes the graph store directly, with no parent zarr container
    and no segmentation -- the internal save/load path, as opposed to
    export_to_geff's user-facing export.
    """
    counter = [0]

    def setup():
        counter[0] += 1
        return (), {"tracks": tracks, "path": tmp_path / f"write_{counter[0]}.geff"}

    benchmark.pedantic(write_to_geff, setup=setup, rounds=ROUNDS, iterations=1)


def test_internal_load(benchmark, exported_store_no_seg):
    def run():
        import_from_geff(exported_store_no_seg)

    benchmark.pedantic(run, rounds=ROUNDS, iterations=1)
