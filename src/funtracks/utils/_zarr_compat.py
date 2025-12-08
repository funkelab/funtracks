"""Zarr v2/v3 compatibility utilities.

This module provides utility functions for handling zarr version compatibility,
supporting both zarr-python v2 and v3, and both zarr spec v2 and v3.

Adapted from geff.core_io._utils.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import zarr

if TYPE_CHECKING:
    from zarr.storage import StoreLike

# Type for zarr open modes
OpenMode = Literal["r", "r+", "a", "w", "w-"]

__all__ = [
    "detect_zarr_spec_version",
    "get_store_path",
    "is_zarr_v3",
    "open_zarr_store",
    "remove_tilde",
    "setup_zarr_array",
    "setup_zarr_group",
]


def is_zarr_v3() -> bool:
    """Check if the installed zarr-python version is v3.

    Returns:
        True if zarr-python v3 is installed, False otherwise.
    """
    return zarr.__version__.startswith("3")


def remove_tilde(store: StoreLike) -> StoreLike:
    """Remove tilde from a store path/str.

    zarr v3 does not recognize the tilde and may write the zarr
    in the wrong directory.

    Args:
        store: The store path to process

    Returns:
        The store with the tilde expanded
    """
    if isinstance(store, str | Path):
        store_str = str(store)
        if "~" in store_str:
            store = os.path.expanduser(store_str)
    return store


def detect_zarr_spec_version(store: StoreLike) -> int | None:
    """Detect the zarr specification version of an existing zarr store.

    Args:
        store: The zarr store path or object

    Returns:
        The zarr spec version (2 or 3) if detectable, None if unknown
    """
    try:
        if isinstance(store, str | Path):
            store_path = Path(store)
            # Check for zarr v3 indicator: zarr.json instead of .zarray/.zgroup
            if (store_path / "zarr.json").exists():
                return 3
            # Check for zarr v2 indicators
            elif (store_path / ".zgroup").exists() or (store_path / ".zarray").exists():
                return 2
        else:
            # For store objects, try to detect based on metadata
            group = zarr.open_group(store, mode="r")
            if group.metadata.zarr_format == 3:  # type: ignore[union-attr]
                return 3
            elif group.metadata.zarr_format == 2:  # type: ignore[union-attr]
                return 2
    except (OSError, KeyError, ValueError, AttributeError):
        # If we can't detect, return None
        pass

    return None


def setup_zarr_group(
    store: StoreLike, zarr_format: Literal[2, 3] = 2, mode: OpenMode = "a"
) -> zarr.Group:
    """Set up and return a zarr group for writing.

    Args:
        store: The zarr store path or object
        zarr_format: The zarr format version to use (default: 2)
        mode: The mode to open the group with (default: "a" for append)

    Returns:
        The opened zarr group
    """
    store = remove_tilde(store)

    # Check for trying to write zarr spec v3 with zarr python v2 and warn
    if zarr_format == 3 and not is_zarr_v3():
        warnings.warn(
            "Requesting zarr spec v3 with zarr-python v2. "
            "zarr-python v2 does not support spec v3. "
            "Ignoring zarr_format=3 and writing zarr spec v2 instead. "
            "Consider upgrading to zarr-python v3 to write zarr spec v3 files.",
            UserWarning,
            stacklevel=2,
        )

    # open/create zarr container
    if is_zarr_v3():
        return zarr.open_group(store, mode=mode, zarr_format=zarr_format)
    else:
        return zarr.open_group(store, mode=mode)


def setup_zarr_array(
    store: StoreLike,
    zarr_format: Literal[2, 3] = 2,
    mode: OpenMode = "w",
    **kwargs,
) -> zarr.Array:
    """Set up and return a zarr array for writing.

    Args:
        store: The zarr store path or object
        zarr_format: The zarr format version to use (default: 2)
        mode: The mode to open the array with (default: "w")
        **kwargs: Additional arguments passed to zarr.open_array
            (shape, dtype, chunks, etc.)

    Returns:
        The opened zarr array
    """
    store = remove_tilde(store)

    # Check for trying to write zarr spec v3 with zarr python v2 and warn
    if zarr_format == 3 and not is_zarr_v3():
        warnings.warn(
            "Requesting zarr spec v3 with zarr-python v2. "
            "zarr-python v2 does not support spec v3. "
            "Ignoring zarr_format=3 and writing zarr spec v2 instead. "
            "Consider upgrading to zarr-python v3 to write zarr spec v3 files.",
            UserWarning,
            stacklevel=2,
        )

    # open/create zarr array
    if is_zarr_v3():
        return zarr.open_array(store, mode=mode, zarr_format=zarr_format, **kwargs)
    else:
        return zarr.open_array(store, mode=mode, **kwargs)


def open_zarr_store(path: str | Path):
    """Open a zarr store with version-appropriate store class.

    In zarr-python v3, uses LocalStore. In v2, uses FSStore.

    Args:
        path: Path to the zarr store

    Returns:
        Appropriate store object for the installed zarr version

    Raises:
        FileNotFoundError: Path does not exist
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    # Check for zarr spec v3 files being opened with zarr python v2 and warn
    if not is_zarr_v3():
        spec_version = detect_zarr_spec_version(path)
        if spec_version == 3:
            warnings.warn(
                "Attempting to open a zarr spec v3 file with zarr-python v2. "
                "This may cause compatibility issues. Consider upgrading to "
                "zarr-python v3 or recreating the file with zarr spec v2.",
                UserWarning,
                stacklevel=2,
            )

    if is_zarr_v3():
        return zarr.storage.LocalStore(path)
    else:
        # FSStore only exists in zarr v2
        return zarr.storage.FSStore(str(path))  # type: ignore[attr-defined]


def get_store_path(store) -> Path:
    """Get the filesystem path from a zarr store.

    Handles differences between zarr v2 FSStore (.path) and v3 LocalStore (.root).

    Args:
        store: A zarr store object

    Returns:
        The filesystem path of the store

    Raises:
        ValueError: Cannot determine store path
    """
    if hasattr(store, "root"):
        return Path(store.root)
    elif hasattr(store, "path"):
        return Path(store.path)
    else:
        raise ValueError(f"Cannot determine store path from {type(store)}")
