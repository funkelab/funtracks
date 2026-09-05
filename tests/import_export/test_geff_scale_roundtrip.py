"""Regression tests for the geff scale round-trip.

Two scales, two homes: ``Tracks.scale`` (segmentation voxel spacing) lives in
``graph.metadata["scale"]`` and is never written to axes.scale; GEFF's
per-axis ``scale`` means "multiply pos by this to get world units" and is
always applied to ``pos`` on import, independent of ``Tracks.scale``. GEFFs
written by pre-fix funtracks are the one exception -- see
``test_legacy_funtracks_geff_scale_migrates_without_double_scaling_points``.
"""

import numpy as np
import pytest
from geff.testing.data import create_mock_geff
from geff_spec import GeffMetadata

from funtracks.import_export import export_to_geff, import_from_geff, write_to_geff


def test_scale_survives_write_to_geff_roundtrip(get_tracks, tmp_path):
    tracks = get_tracks(ndim=3, with_seg=False, prefill_track_ids=True)
    tracks.scale = [1.0, 0.25, 0.5]

    geff_path = tmp_path / "tracks.geff"
    write_to_geff(tracks, geff_path)

    assert import_from_geff(geff_path).scale == [1.0, 0.25, 0.5]


def test_scale_survives_export_to_geff_roundtrip(get_tracks, tmp_path):
    tracks = get_tracks(ndim=4, with_seg=False, prefill_track_ids=True)
    tracks.scale = [1.0, 2.0, 0.25, 0.5]

    export_to_geff(tracks, tmp_path / "container")

    loaded = import_from_geff(tmp_path / "container" / "tracks.geff")
    assert loaded.scale == [1.0, 2.0, 0.25, 0.5]


def test_scale_not_written_to_axes(get_tracks, tmp_path):
    """Tracks.scale (segmentation spacing) must not leak into axes.scale,
    which per the geff spec means something else: how to convert pos to
    world units."""
    tracks = get_tracks(ndim=3, with_seg=False, prefill_track_ids=True)
    tracks.scale = [1.0, 0.25, 0.5]

    geff_path = tmp_path / "tracks.geff"
    write_to_geff(tracks, geff_path)

    meta = GeffMetadata.read(geff_path)
    assert all(ax.scale is None for ax in meta.axes)
    assert meta.extra["tracksdata"]["scale"] == [1.0, 0.25, 0.5]


def test_explicit_scale_overrides_metadata(get_tracks, tmp_path):
    """A caller-supplied scale must still win over what the file says."""
    tracks = get_tracks(ndim=3, with_seg=False, prefill_track_ids=True)
    tracks.scale = [1.0, 0.25, 0.5]

    geff_path = tmp_path / "tracks.geff"
    write_to_geff(tracks, geff_path)

    loaded = import_from_geff(geff_path, scale=[1.0, 3.0, 3.0])
    assert loaded.scale == [1.0, 3.0, 3.0]


def test_explicit_scale_does_not_suppress_points_scaling():
    """An explicit ``scale`` argument overrides Tracks.scale (segmentation
    spacing) only. It must not affect whether pos gets scaled by axes.scale --
    that's a separate, always-applied conversion to world units."""
    store, graph_data = create_mock_geff(
        node_id_dtype="uint",
        node_axis_dtypes={"position": "float64", "time": "int64"},
        directed=True,
        num_nodes=5,
        num_edges=2,
        include_t=True,
        include_z=False,
        include_y=True,
        include_x=True,
    )
    meta = GeffMetadata.read(store)
    for ax in meta.axes:
        if ax.type == "space":
            ax.scale = 2.0
    meta.write(store)

    orig_y = graph_data["node_props"]["y"]["values"]
    orig_x = graph_data["node_props"]["x"]["values"]

    tracks = import_from_geff(store, scale=[1.0, 9.0, 9.0])

    assert tracks.scale == [1.0, 9.0, 9.0]
    df = tracks.graph_solution.node_attrs(attr_keys=["pos"])
    pos = np.array(df["pos"].to_list())
    np.testing.assert_allclose(pos[:, 0], orig_y * 2.0)
    np.testing.assert_allclose(pos[:, 1], orig_x * 2.0)


def test_scale_none_when_axes_have_no_scale():
    """Third-party geff without per-axis scale stays 'unknown', not unity."""
    store, _ = create_mock_geff(
        node_id_dtype="uint",
        node_axis_dtypes={"position": "float64", "time": "int64"},
        directed=True,
        num_nodes=5,
        num_edges=2,
        include_t=True,
        include_z=False,
        include_y=True,
        include_x=True,
    )
    assert import_from_geff(store).scale is None


@pytest.mark.parametrize("ndim", [3, 4])
def test_scale_roundtrip_with_segmentation(get_tracks, tmp_path, ndim):
    """Positions are in world units; the scale must come back so that
    segmentation lookups keep matching."""
    tracks = get_tracks(ndim=ndim, with_seg=True, prefill_track_ids=True)
    scale = [1.0, 0.25, 0.5] if ndim == 3 else [1.0, 2.0, 0.25, 0.5]
    tracks.scale = scale

    geff_path = tmp_path / "tracks.geff"
    write_to_geff(tracks, geff_path)

    loaded = import_from_geff(geff_path)
    assert loaded.scale is not None
    np.testing.assert_allclose(loaded.scale, scale)


def test_axes_scale_applied_to_points_on_import():
    """A correctly-written GEFF's axes.scale means 'multiply pos by this to
    get world units', so it must be applied -- this is the geff spec's own
    convention, unrelated to funtracks' Tracks.scale (segmentation spacing)."""
    store, graph_data = create_mock_geff(
        node_id_dtype="uint",
        node_axis_dtypes={"position": "float64", "time": "int64"},
        directed=True,
        num_nodes=5,
        num_edges=2,
        include_t=True,
        include_z=False,
        include_y=True,
        include_x=True,
    )
    meta = GeffMetadata.read(store)
    for ax in meta.axes:
        if ax.type == "space":
            ax.scale = 2.0
    meta.write(store)

    orig_y = graph_data["node_props"]["y"]["values"]
    orig_x = graph_data["node_props"]["x"]["values"]

    tracks = import_from_geff(store)
    df = tracks.graph_solution.node_attrs(attr_keys=["pos"])
    pos = np.array(df["pos"].to_list())

    np.testing.assert_allclose(pos[:, 0], orig_y * 2.0)
    np.testing.assert_allclose(pos[:, 1], orig_x * 2.0)
    # axes.scale describes points, not the segmentation -- unrelated to Tracks.scale
    assert tracks.scale is None


def test_legacy_funtracks_geff_scale_migrates_without_double_scaling_points(
    get_tracks, tmp_path
):
    """Geffs written by pre-fix funtracks stored Tracks.scale (mislabeled) in
    axes.scale, while pos was already written in world units. Detect that
    combination (funtracks FeatureDict extra, no version string) and recover
    Tracks.scale from axes.scale without also applying it to pos, which would
    silently corrupt already-correct positions."""
    tracks = get_tracks(ndim=3, with_seg=False, prefill_track_ids=True)
    tracks.scale = [1.0, 0.25, 0.5]

    geff_path = tmp_path / "tracks.geff"
    write_to_geff(tracks, geff_path)
    orig_pos = tracks.graph_solution.node_attrs(attr_keys=["pos"])["pos"].to_list()

    # Simulate pre-fix funtracks output: scale lived in axes.scale, not
    # graph.metadata["scale"], and there was no version string.
    meta = GeffMetadata.read(geff_path)
    scale = meta.extra["tracksdata"].pop("scale")
    for ax, s in zip(meta.axes, scale, strict=True):
        ax.scale = s
    del meta.extra["funtracks"]["version"]
    meta.write(geff_path)

    loaded = import_from_geff(geff_path)
    assert loaded.scale == [1.0, 0.25, 0.5]
    loaded_pos = loaded.graph_solution.node_attrs(attr_keys=["pos"])["pos"].to_list()
    np.testing.assert_allclose(loaded_pos, orig_pos)
