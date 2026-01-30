"""Tests for tracksdata utility functions."""

import numpy as np
import pytest

from funtracks.utils.tracksdata_utils import pixels_to_td_mask, td_mask_to_pixels

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
    reconstructed_mask, area = pixels_to_td_mask(pixels, ndim=ndim, scale=None)

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
    reconstructed_mask, scaled_area = pixels_to_td_mask(pixels, ndim=ndim, scale=scale)

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
    from tracksdata.nodes._mask import Mask

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
