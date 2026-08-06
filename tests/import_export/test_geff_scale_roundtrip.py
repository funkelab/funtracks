"""Regression tests: the geff axes metadata carries the scale, so a geff
round-trip must return it.

``write_to_geff`` / ``export_to_geff`` write ``scale`` into each geff axis.
``import_from_geff`` used to ignore it, so anisotropic data silently came back
with ``scale=None``.
"""

import numpy as np
import pytest
from geff.testing.data import create_mock_geff

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


def test_explicit_scale_overrides_metadata(get_tracks, tmp_path):
    """A caller-supplied scale must still win over what the file says."""
    tracks = get_tracks(ndim=3, with_seg=False, prefill_track_ids=True)
    tracks.scale = [1.0, 0.25, 0.5]

    geff_path = tmp_path / "tracks.geff"
    write_to_geff(tracks, geff_path)

    loaded = import_from_geff(geff_path, scale=[1.0, 3.0, 3.0])
    assert loaded.scale == [1.0, 3.0, 3.0]


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
