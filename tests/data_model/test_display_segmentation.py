import numpy as np
import pytest
import tracksdata as td
from tracksdata.array import GraphArrayView
from tracksdata.nodes import Mask

from funtracks.data_model import Tracks
from funtracks.utils.tracksdata_utils import create_empty_graph


@pytest.fixture
def thin_mask_tracks(tmp_path, backend) -> Tracks:
    """Tracks whose masks are single voxels, thinner than any downscale factor.

    The "every node renders" property only has teeth when the single-voxel fallback
    in ``Mask.paint_buffer`` actually fires, which chunky fixture masks never do.

    Nodes are spaced so that no two share an output voxel at the factors tested: a
    voxel covered by more than one object takes an unspecified value, so overlap
    would exercise tie-breaking rather than the fallback.
    """
    graph = create_empty_graph(
        node_attributes=[td.DEFAULT_ATTR_KEYS.MASK, td.DEFAULT_ATTR_KEYS.BBOX],
        node_default_values=[0.0, 0.0],
        database=":memory:" if backend == "sql" else str(tmp_path / "thin.db"),
        ndim=3,
        backend=backend,
    )
    node_ids, node_attrs = [], []
    for node_id, (t, y, x) in enumerate(
        [(0, 3, 3), (0, 40, 41), (0, 60, 61), (1, 3, 3), (1, 90, 90)], start=1
    ):
        mask = Mask(np.ones((1, 1), dtype=bool), bbox=np.array([y, x, y + 1, x + 1]))
        node_ids.append(node_id)
        node_attrs.append(
            {
                "t": t,
                "solution": True,
                td.DEFAULT_ATTR_KEYS.MASK: mask,
                td.DEFAULT_ATTR_KEYS.BBOX: mask.bbox,
            }
        )
    graph.bulk_add_nodes(nodes=node_attrs, indices=node_ids)
    graph.metadata["shape"] = (2, 100, 100)
    return Tracks(graph, ndim=3, time_attr="t")


def _labels_at(array, t: int) -> set[int]:
    return set(np.unique(np.array(array[t]))) - {0}


@pytest.mark.parametrize("downscale", [(2, 2), (4, 4), (1, 8), (16, 16)])
def test_every_node_renders_at_any_factor(thin_mask_tracks, downscale):
    """The coarse view draws every node, which is what keeps cells selectable."""
    tracks = thin_mask_tracks
    view = tracks.display_segmentation(downscale)
    for t in range(2):
        expected = {
            node
            for node in tracks.graph_solution.node_ids()
            if tracks.get_time(node) == t
        }
        assert _labels_at(view, t) == expected


def test_coarse_shape(get_tracks):
    tracks = get_tracks(ndim=4, with_seg=True)
    view = tracks.display_segmentation((1, 4, 4))
    assert tracks.segmentation.shape == (5, 100, 100, 100)
    assert view.shape == (5, 100, 25, 25)
    assert view.downscale == (1, 4, 4)


def test_scalar_factor_broadcasts(get_tracks):
    tracks = get_tracks(ndim=4, with_seg=True)
    assert tracks.display_segmentation(4).downscale == (4, 4, 4)


def test_views_are_cached_per_factor(get_tracks):
    tracks = get_tracks(ndim=3, with_seg=True)
    assert tracks.display_segmentation((4, 4)) is tracks.display_segmentation(4)
    assert tracks.display_segmentation((4, 4)) is not tracks.display_segmentation((2, 2))


def test_unit_factor_returns_the_full_resolution_array(get_tracks):
    """`view is tracks.segmentation` is the cheapest form of the "am I coarse" check."""
    tracks = get_tracks(ndim=3, with_seg=True)
    assert tracks.display_segmentation((1, 1)) is tracks.segmentation
    assert tracks.display_segmentation(1) is tracks.segmentation


def test_none_without_segmentation(get_tracks):
    tracks = get_tracks(ndim=3, with_seg=False)
    assert tracks.segmentation is None
    assert tracks.display_segmentation((4, 4)) is None


def test_time_first_length_is_rejected_with_a_hint(get_tracks):
    tracks = get_tracks(ndim=4, with_seg=True)
    with pytest.raises(ValueError, match="time-first"):
        tracks.display_segmentation((1, 1, 4, 4))


@pytest.mark.parametrize("bad", [(4, 4, 4, 4, 4), (4,), (1.5, 2, 2), (1, 0, 2)])
def test_invalid_factors_rejected(get_tracks, bad):
    tracks = get_tracks(ndim=4, with_seg=True)
    with pytest.raises(ValueError):
        tracks.display_segmentation(bad)


def test_full_resolution_array_is_untouched(get_tracks):
    """§4.1: requesting a display view must not change the authoritative array."""
    tracks = get_tracks(ndim=3, with_seg=True)
    before = np.array(tracks.segmentation[0])
    seg = tracks.segmentation
    tracks.display_segmentation((8, 8))
    assert tracks.segmentation is seg
    assert tracks.segmentation.shape == (5, 100, 100)
    assert tracks.segmentation.dtype == before.dtype
    np.testing.assert_array_equal(np.array(tracks.segmentation[0]), before)


def test_assigning_a_coarse_view_to_segmentation_is_rejected(get_tracks):
    """The one failure mode in this design that would otherwise be silent."""
    tracks = get_tracks(ndim=3, with_seg=True)
    coarse = tracks.display_segmentation((4, 4))
    with pytest.raises(ValueError, match="downscaled"):
        tracks.segmentation = coarse
    assert tracks.segmentation.shape == (5, 100, 100)


def test_assigning_a_full_resolution_view_is_allowed(get_tracks):
    tracks = get_tracks(ndim=3, with_seg=True)
    tracks.segmentation = GraphArrayView(
        graph=tracks.graph_solution,
        shape=(5, 100, 100),
        attr_key="node_id",
        offset=0,
    )
    assert tracks.segmentation.downscale == (1, 1)


def test_display_view_inherits_the_dtype(get_tracks):
    """Inherited rather than re-inferred: the probe is a full node_id column read."""
    tracks = get_tracks(ndim=3, with_seg=True)
    assert tracks.display_segmentation((4, 4)).dtype == tracks.segmentation.dtype


def test_display_scale(get_tracks):
    tracks = get_tracks(ndim=4, with_seg=True)
    tracks.scale = [1.0, 1.97, 0.485, 0.485]
    assert tracks.display_scale((1, 4, 4)) == pytest.approx([1.0, 1.97, 1.94, 1.94])
    assert tracks.display_scale((1, 1, 1)) == pytest.approx(tracks.scale)


def test_display_scale_is_none_without_scale_metadata(get_tracks):
    """Never manufacture a scale out of the factors alone: it would look valid."""
    tracks = get_tracks(ndim=4, with_seg=True)
    assert tracks.scale is None
    assert tracks.display_scale((1, 4, 4)) is None


def test_export_is_unaffected_by_an_existing_display_view(get_tracks, tmp_path):
    """§4.1's silent failure: a coarse shape sizing a full-resolution render."""
    import zarr

    from funtracks.import_export._export_segmentation import export_segmentation

    tracks = get_tracks(ndim=3, with_seg=True)
    tracks.display_segmentation((8, 8))
    out = tmp_path / "seg.zarr"
    export_segmentation(tracks, out, file_format="zarr")
    exported = zarr.open(str(out), mode="r")
    assert exported.shape == (5, 100, 100)
    np.testing.assert_array_equal(exported[0], np.array(tracks.segmentation[0]))
