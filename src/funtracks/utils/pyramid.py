"""Create OME-Zarr scale pyramids from dask arrays.

Supports anisotropy-aware downsampling for intensity images (mean) and
nearest-neighbor downsampling for label/segmentation data.  Writes
standard OME-NGFF 0.4 metadata via ``ome-zarr-py``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import dask.array as da
import numpy as np
import zarr
from ome_zarr.writer import write_multiscales_metadata

logger = logging.getLogger(__name__)

# Target ~3M voxels per spatial chunk (benchmark sweet spot: 3-6M).
# 144^3 ≈ 3M voxels.
DEFAULT_SPATIAL_CHUNK_SIZE = 144


def create_pyramid(
    data: da.Array,
    output_path: str | Path,
    voxel_size: tuple[float, ...],
    is_label: bool = False,
    num_levels: int | None = None,
    spatial_chunk_size: int = DEFAULT_SPATIAL_CHUNK_SIZE,
) -> Path:
    """Write a dask array to an on-disk OME-Zarr scale pyramid.

    Downsample factors are computed per-level based on voxel sizes:
    a spatial dimension is only downsampled (2x) when its voxel size
    is within 2x of the finest spatial dimension, so coarser axes
    (e.g. Z in anisotropic data) wait until the finer ones catch up.

    Each level is persisted to zarr before the next is computed,
    preventing deep dask-graph cascading.

    Args:
        data: 4-D dask array with axes ``(T, Z, Y, X)``.
        output_path: Filesystem path for the output ``.zarr`` store.
        voxel_size: Physical voxel sizes ``(t, z, y, x)``.
        is_label: If *True* use nearest-neighbor (stride) downsampling;
            otherwise use mean block-reduction.
        num_levels: Number of pyramid levels (including the base).
            If *None*, auto-computed so the coarsest level fits in a
            single spatial chunk of *spatial_chunk_size*.
        spatial_chunk_size: Target chunk-edge length for spatial dims.

    Returns:
        The *output_path* as a :class:`~pathlib.Path`.
    """
    if data.ndim != 4:
        raise ValueError(f"Expected a 4-D (TZYX) array, got {data.ndim}-D")
    if len(voxel_size) != 4:
        raise ValueError(
            f"voxel_size must have 4 elements (t, z, y, x), got {len(voxel_size)}"
        )

    output_path = Path(output_path)
    spatial_shape = data.shape[1:]  # (Z, Y, X)
    current_voxel = list(voxel_size)

    if num_levels is None:
        num_levels = _compute_num_levels(spatial_shape, current_voxel, spatial_chunk_size)

    logger.info("Creating %d-level pyramid at %s", num_levels, output_path)

    group = zarr.open_group(str(output_path), mode="w")
    level_data = data
    datasets: list[dict] = []

    for level in range(num_levels):
        if level > 0:
            factors = _compute_downsample_factors(current_voxel)
            level_data = _downsample(level_data, factors, is_label)
            for i in range(1, 4):
                current_voxel[i] *= factors[i]

        # Rechunk — clamp to actual array size
        chunks = (
            1,
            min(spatial_chunk_size, level_data.shape[1]),
            min(spatial_chunk_size, level_data.shape[2]),
            min(spatial_chunk_size, level_data.shape[3]),
        )
        level_data = level_data.rechunk(chunks)

        level_path = f"s{level}"
        logger.info(
            "  Level %d: shape=%s  voxel=%s",
            level,
            level_data.shape,
            current_voxel,
        )

        da.to_zarr(
            level_data,
            group.store,
            component=level_path,
            overwrite=True,
        )

        # Build dataset entry for OME-NGFF metadata
        datasets.append(
            {
                "path": level_path,
                "coordinateTransformations": [
                    {"type": "scale", "scale": list(current_voxel)}
                ],
            }
        )

        # Re-open from zarr so next level's graph is shallow
        level_data = da.from_zarr(group[level_path])

    # Write OME-NGFF 0.4 metadata via ome-zarr-py
    write_multiscales_metadata(
        group,
        datasets,
        axes=[
            {"name": "t", "type": "time"},
            {"name": "z", "type": "space", "unit": "micrometer"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"},
        ],
    )

    return output_path


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _compute_downsample_factors(
    current_voxel_size: list[float],
) -> tuple[int, int, int, int]:
    """Anisotropy-aware downsample factors for a TZYX array.

    A spatial dimension is downsampled (factor 2) only when its voxel
    size is within 2x of the finest spatial dimension.  Time is never
    downsampled.
    """
    spatial = current_voxel_size[1:]  # z, y, x
    finest = min(spatial)
    return (
        1,
        2 if spatial[0] < finest * 2 else 1,
        2 if spatial[1] < finest * 2 else 1,
        2 if spatial[2] < finest * 2 else 1,
    )


def _compute_num_levels(
    spatial_shape: tuple[int, ...],
    voxel_size: list[float],
    chunk_size: int,
) -> int:
    """How many levels until every spatial dim fits in one chunk."""
    shape = list(spatial_shape)
    voxel = list(voxel_size)
    levels = 1
    while any(s > chunk_size for s in shape):
        factors = _compute_downsample_factors(voxel)
        for i in range(3):
            shape[i] = (shape[i] + factors[i + 1] - 1) // factors[i + 1]
            voxel[i + 1] *= factors[i + 1]
        levels += 1
    return levels


def _downsample(
    data: da.Array,
    factors: tuple[int, ...],
    is_label: bool,
) -> da.Array:
    """Downsample a dask array by *factors*.

    Labels use stride-slicing (nearest-neighbor); intensity data uses
    ``da.coarsen(np.mean, ...)`` with edge-padding for ceiling-division
    behaviour.
    """
    original_dtype = data.dtype

    if is_label:
        slices = tuple(slice(None, None, f) for f in factors)
        return data[slices]

    # Pad so shape is divisible by factors (ceiling division)
    pad_width = [(0, (f - s % f) % f) for s, f in zip(data.shape, factors, strict=True)]
    if any(p[1] > 0 for p in pad_width):
        data = da.pad(data, pad_width, mode="edge")

    return da.coarsen(
        np.mean,
        data,
        dict(enumerate(factors)),
        trim_excess=False,
    ).astype(original_dtype)
