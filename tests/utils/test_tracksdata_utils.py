"""Tests for tracksdata utility functions."""

import threading

import numpy as np
import pytest
import tracksdata as td
from tracksdata.nodes import Mask

from funtracks.utils.tracksdata_utils import (
    create_empty_graph,
    create_empty_graphview_graph,
    pixels_to_td_mask,
    td_mask_to_pixels,
    tighten_td_mask,
    union_td_masks,
)

# Import from conftest
from ..conftest import (
    make_2d_disk_mask,
    make_2d_square_mask,
    make_3d_cube_mask,
    make_3d_sphere_mask,
)


@pytest.mark.parametrize(
    "mask_func,ndim",
    [
        (lambda: make_2d_disk_mask(center=(50, 50), radius=20), 3),
        (lambda: make_2d_disk_mask(center=(25, 75), radius=10), 3),
        (lambda: make_2d_square_mask(start_corner=(10, 10), width=5), 3),
        (lambda: make_3d_sphere_mask(center=(50, 50, 50), radius=20), 4),
        (lambda: make_3d_sphere_mask(center=(25, 75, 30), radius=15), 4),
        (lambda: make_3d_cube_mask(start_corner=(10, 10, 10), width=5), 4),
    ],
)
def test_mask_pixels_roundtrip(mask_func, ndim):
    """Test that mask -> pixels -> mask roundtrip preserves the mask."""
    # Create original mask
    original_mask = mask_func()
    time = 5  # Arbitrary time point

    # Convert mask to pixels
    pixels = td_mask_to_pixels(original_mask, time=time, ndim=ndim)

    # Verify pixel format
    assert len(pixels) == ndim  # Should have ndim arrays
    assert len(pixels[0]) == len(pixels[1])  # All arrays same length
    assert np.all(pixels[0] == time)  # Time should be constant

    # Convert pixels back to mask
    reconstructed_mask, area = pixels_to_td_mask(
        pixels, ndim=ndim, scale=[1 for _ in range(ndim)], include_area=True
    )

    # Verify the reconstructed mask matches the original
    assert np.array_equal(reconstructed_mask.bbox, original_mask.bbox), (
        "Bounding boxes should match"
    )
    assert np.array_equal(reconstructed_mask.mask, original_mask.mask), (
        "Mask arrays should match"
    )
    assert area == np.sum(original_mask.mask), "Area should match pixel count"


@pytest.mark.parametrize("ndim", [3, 4])
def test_mask_pixels_roundtrip_with_scale(ndim):
    """Test mask->pixels->mask roundtrip with scale factors."""
    # Create mask
    if ndim == 3:
        mask = make_2d_disk_mask(center=(40, 60), radius=15)
        scale = [1.0, 2.0, 3.0]  # time, y, x scales
    else:
        mask = make_3d_sphere_mask(center=(40, 60, 30), radius=12)
        scale = [1.0, 2.0, 3.0, 4.0]  # time, z, y, x scales

    time = 3

    # Convert mask to pixels
    pixels = td_mask_to_pixels(mask, time=time, ndim=ndim)

    # Convert back with scale
    reconstructed_mask, scaled_area = pixels_to_td_mask(
        pixels, ndim=ndim, scale=scale, include_area=True
    )

    # Verify mask structure is preserved
    assert np.array_equal(reconstructed_mask.bbox, mask.bbox)
    assert np.array_equal(reconstructed_mask.mask, mask.mask)

    # Verify area is scaled correctly
    expected_area = np.sum(mask.mask) * np.prod(scale[1:])
    assert np.isclose(scaled_area, expected_area), (
        f"Scaled area {scaled_area} should match expected {expected_area}"
    )


def test_td_mask_to_pixels_empty_mask():
    """Test converting an empty mask to pixels."""
    from tracksdata.nodes import Mask

    # Create a truly empty mask (all False)
    empty_mask_array = np.zeros((2, 2), dtype=bool)
    empty_bbox = np.array([10, 10, 12, 12])
    empty_mask = Mask(empty_mask_array, bbox=empty_bbox)

    pixels = td_mask_to_pixels(empty_mask, time=1, ndim=3)

    # Should return empty arrays
    assert len(pixels) == 3
    assert len(pixels[0]) == 0  # No pixels
    assert len(pixels[1]) == 0
    assert len(pixels[2]) == 0


@pytest.mark.parametrize("ndim", [3, 4])
def test_pixels_coordinate_offset(ndim):
    """Test that bbox offset is correctly applied in pixel coordinates."""
    # Create a mask at a non-zero position
    if ndim == 3:
        mask = make_2d_square_mask(start_corner=(20, 30), width=3)
        expected_bbox = np.array([20, 30, 23, 33])
    else:
        mask = make_3d_cube_mask(start_corner=(20, 30, 40), width=3)
        expected_bbox = np.array([20, 30, 40, 23, 33, 43])

    assert np.array_equal(mask.bbox, expected_bbox)

    # Convert to pixels
    pixels = td_mask_to_pixels(mask, time=7, ndim=ndim)

    # Verify pixel coordinates are in global space (not local)
    if ndim == 3:
        assert np.min(pixels[1]) == 20  # min y
        assert np.max(pixels[1]) == 22  # max y
        assert np.min(pixels[2]) == 30  # min x
        assert np.max(pixels[2]) == 32  # max x
    else:
        assert np.min(pixels[1]) == 20  # min z
        assert np.max(pixels[1]) == 22  # max z
        assert np.min(pixels[2]) == 30  # min y
        assert np.max(pixels[2]) == 32  # max y
        assert np.min(pixels[3]) == 40  # min x
        assert np.max(pixels[3]) == 42  # max x


def test_memory_graph_survives_thread_boundary():
    """A base graph created in a worker thread must stay accessible from the main thread.

    Regression test: nodes_from_segmentation previously used database=':memory:',
    which caused 'no such table: Metadata' when the graph crossed a thread boundary
    (SQLite in-memory DBs are connection-scoped; a new thread gets a fresh empty DB).
    Fix: use the default temp-file database instead of ':memory:'.
    """
    result = {}

    def worker():
        graph = create_empty_graph(
            node_attributes=["pos"],
            ndim=3,
        )
        graph.bulk_add_nodes([{"t": 0, "pos": [1.0, 2.0], "solution": True}], indices=[1])
        result["graph"] = graph

    t = threading.Thread(target=worker)
    t.start()
    t.join()

    graph = result["graph"]

    # create_empty_graph now returns the base graph; build a view to exercise
    # detach(), which calls graph.metadata internally via BaseGraph.from_other().
    # With :memory: + default connection pool it opens a new empty DB → crash.
    detached = graph.filter().subgraph().detach()

    assert detached.num_nodes() == 1


def test_create_empty_graph_with_solution_attr():
    """Test that passing solution as a node/edge attribute does not raise.

    Regression test: create_empty_graph unconditionally added the
    solution attribute at the end, even when it was already added via the
    node_attributes / edge_attributes loop, causing a ValueError.
    """
    # Should not raise ValueError even though solution is listed explicitly
    graph = create_empty_graph(
        node_attributes=["solution"],
        edge_attributes=["solution"],
        node_default_values=[True],
        edge_default_values=[True],
    )

    assert graph is not None


def test_create_empty_graphview_graph_warns():
    """create_empty_graphview_graph is a deprecated alias for create_empty_graph,
    kept for downstream code (e.g. motile_tracker) that still expects a view back."""
    with pytest.warns(DeprecationWarning, match="create_empty_graphview_graph"):
        view = create_empty_graphview_graph(node_attributes=["pos"], ndim=3)

    assert isinstance(view, td.graph.GraphView)


def test_tighten_td_mask_shrinks_bounding_box():
    """A box with empty margin is shrunk around the set pixels, keeping them put."""
    array = np.zeros((6, 7), dtype=bool)
    array[2:4, 3:6] = True
    mask = Mask(array, bbox=np.array([10, 20, 16, 27]))

    tightened = tighten_td_mask(mask)

    assert np.array_equal(np.asarray(tightened.bbox), [12, 23, 14, 26])
    assert tightened.mask.shape == (2, 3)
    assert tightened.mask.all()
    # the pixels themselves did not move
    assert np.array_equal(
        np.array(np.nonzero(array)) + np.array([[10], [20]]),
        np.array(np.nonzero(tightened.mask)) + np.array([[12], [23]]),
    )


def test_tighten_td_mask_leaves_tight_box_alone():
    mask = Mask(np.ones((2, 3), dtype=bool), bbox=np.array([1, 2, 3, 5]))
    assert tighten_td_mask(mask) is mask


def test_tighten_td_mask_rejects_empty():
    mask = Mask(np.zeros((2, 3), dtype=bool), bbox=np.array([0, 0, 2, 3]))
    with pytest.raises(ValueError, match="empty mask"):
        tighten_td_mask(mask)


def test_union_td_masks_combines_disjoint_boxes():
    """The union spans every set pixel, with a box tightened around them."""
    left = Mask(np.ones((2, 2), dtype=bool), bbox=np.array([0, 0, 2, 2]))
    right = Mask(np.ones((1, 3), dtype=bool), bbox=np.array([5, 4, 6, 7]))

    combined = union_td_masks([left, right])

    assert np.array_equal(np.asarray(combined.bbox), [0, 0, 6, 7])
    assert combined.mask.sum() == left.mask.sum() + right.mask.sum()
    assert combined.mask[0:2, 0:2].all()
    assert combined.mask[5, 4:7].all()


def test_union_td_masks_tightens_loose_inputs():
    """Callers may pass masks in whatever box they had; the union is still tight."""
    array = np.zeros((5, 5), dtype=bool)
    array[1, 1] = True
    loose = Mask(array, bbox=np.array([0, 0, 5, 5]))

    combined = union_td_masks([loose, loose])

    assert np.array_equal(np.asarray(combined.bbox), [1, 1, 2, 2])
    assert combined.mask.shape == (1, 1)


def test_union_td_masks_matches_pixels_to_td_mask():
    """Unioning parts agrees with building one mask from all their pixels at once."""
    rng = np.random.default_rng(0)
    pixels_per_part = []
    masks = []
    for offset in (0, 7, 13):
        coords = rng.integers(0, 4, size=(2, 6)) + offset
        pixels = (np.full(coords.shape[1], 3), coords[0], coords[1])
        pixels_per_part.append(pixels)
        masks.append(pixels_to_td_mask(pixels, ndim=3))

    combined = union_td_masks(masks)
    expected = pixels_to_td_mask(
        tuple(np.concatenate([p[dim] for p in pixels_per_part]) for dim in range(3)),
        ndim=3,
    )

    assert np.array_equal(np.asarray(combined.bbox), np.asarray(expected.bbox))
    assert np.array_equal(combined.mask, expected.mask)


def test_union_td_masks_rejects_empty_sequence():
    with pytest.raises(ValueError, match="zero masks"):
        union_td_masks([])
