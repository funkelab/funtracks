"""Adapted implementation of napari.napari_builtins.io._read.magic_imread"""

import re
from glob import glob
from pathlib import Path

import dask.array as da
import numpy as np
import tifffile
import zarr
from dask import delayed
from numpy.typing import ArrayLike


def _alphanumeric_key(s: str):
    return [int(c) if c.isdigit() else c for c in re.split("([0-9]+)", s)]


def _guess_zarr_path(path: Path) -> bool:
    return any(part.endswith(".zarr") for part in path.parts)


def read_zarr_dataset(path: Path) -> tuple[ArrayLike | list[ArrayLike], tuple[int]]:
    """Read a zarr dataset, including an array or a group of arrays."""
    if (path / ".zarray").exists():
        image = da.from_zarr(path)
        shape = image.shape
    elif (path / "zarr.json").exists():
        data = zarr.open(store=path)
        if isinstance(data, zarr.Array):
            image = da.from_zarr(data)
            shape = image.shape
        else:
            raise ValueError(f"Not a valid zarr dataset: {path}")
    else:
        raise ValueError(f"Not a valid zarr dataset: {path}")
    return image, shape


def imread(filename: str) -> np.ndarray:
    """Read an image from a string file path using tifffile.imwrite."""
    return tifffile.imread(filename)


PathOrStr = str | Path


def magic_imread(
    filenames: PathOrStr | list[PathOrStr], *, use_dask=None, stack=True
) -> ArrayLike | list[ArrayLike]:
    """Dispatch the appropriate reader given some files."""

    _filenames: list[str] = (
        [str(x) for x in filenames]
        if isinstance(filenames, list | tuple)
        else [str(filenames)]
    )

    if not _filenames:
        raise ValueError("No files provided")

    filenames_expanded: list[str] = []
    for filename in _filenames:
        path = Path(filename)
        if path.is_dir() and not _guess_zarr_path(path):
            dir_contents = sorted(
                glob(str(path / "*.tif")) + glob(str(path / "*.tiff")),
                key=_alphanumeric_key,
            )
            dir_contents_files = [f for f in dir_contents if not Path(f).is_dir()]
            filenames_expanded.extend(dir_contents_files)
        else:
            filenames_expanded.append(filename)

    if not filenames_expanded:
        raise ValueError(f"No readable TIFF files found in {filenames}")

    if use_dask is None:
        use_dask = len(filenames_expanded) > 1

    images: list[ArrayLike] = []
    shape: tuple[int, ...] | None = None
    dtype = None

    for filename in filenames_expanded:
        path = Path(filename)
        if _guess_zarr_path(path):
            image, zarr_shape = read_zarr_dataset(path)
            if len(zarr_shape) == 1:
                continue
            if shape is None:
                shape = zarr_shape
        else:
            if shape is None:
                image = imread(filename)
                shape = image.shape
                dtype = image.dtype
            if use_dask:
                image = da.from_delayed(
                    delayed(imread)(filename), shape=shape, dtype=dtype
                )
            elif len(images) > 0:
                image = imread(filename)
        images.append(image)

    if not images:
        raise ValueError("No valid images loaded")

    if len(images) == 1:
        return images[0]
    elif stack:
        if use_dask:
            return da.stack(images)
        else:
            try:
                return np.stack(images)
            except ValueError as e:
                raise ValueError(
                    "To stack multiple files with numpy, all input arrays must have the"
                    " same shape. Set `use_dask=True` to allow stacking with different "
                    "shapes."
                ) from e
    else:
        return images
