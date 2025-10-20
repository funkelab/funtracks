from __future__ import annotations

from typing import TYPE_CHECKING

from ._feature import Feature

if TYPE_CHECKING:
    from collections.abc import Sequence


def Centroid(axes: Sequence[str]) -> Feature:
    """A regionprops feature for computing the centroid.

    Args:
        axes (Sequence[str]): The names of the axes to use for display names.

    Returns:
        Feature: A feature dict representing centroid
    """
    return {
        "feature_type": "node",
        "value_type": "float",
        "num_values": len(axes),
        "valid_ndim": (3, 4),
        "display_name": list(axes),
        "recompute": True,
        "required": True,
        "default_value": None,
    }


def Area(ndim: int = 3) -> Feature:
    """A regionprops feature for computing area or volume.

    Args:
        ndim (int): The number of dimensions of the tracks. Controls the display
            name.

    Returns:
        Feature: A feature dict representing area/volume
    """
    return {
        "feature_type": "node",
        "value_type": "float",
        "num_values": 1,
        "valid_ndim": (3, 4),
        "display_name": "Area" if ndim == 3 else "Volume",
        "recompute": True,
        "required": True,
        "default_value": None,
    }


def Intensity() -> Feature:
    """A regionprops feature for computing the intensity.

    Returns:
        Feature: A feature dict representing intensity
    """
    return {
        "feature_type": "node",
        "value_type": "float",
        "num_values": 1,
        "valid_ndim": (3, 4),
        "display_name": "Intensity",
        "recompute": True,
        "required": True,
        "default_value": None,
    }


def EllipsoidAxes(ndim: int = 3) -> Feature:
    """A regionprops feature for computing the ellipsoid axis radii.

    Args:
        ndim (int): The number of dimensions of the tracks. Controls the display
            name.

    Returns:
        Feature: A feature dict representing ellipsoid axes
    """
    return {
        "feature_type": "node",
        "value_type": "float",
        "num_values": 1,
        "valid_ndim": (3, 4),
        "display_name": "Ellipse axis radii" if ndim == 3 else "Ellipsoid axis radii",
        "recompute": True,
        "required": True,
        "default_value": None,
    }


def Circularity(ndim: int = 3) -> Feature:
    """A regionprops feature for computing the circularity or sphericity.

    Args:
        ndim (int): The number of dimensions of the tracks. Controls the display
            name.

    Returns:
        Feature: A feature dict representing circularity/sphericity
    """
    return {
        "feature_type": "node",
        "value_type": "float",
        "num_values": 1,
        "valid_ndim": (3, 4),
        "display_name": "Circularity" if ndim == 3 else "Sphericity",
        "recompute": True,
        "required": True,
        "default_value": None,
    }


def Perimeter(ndim: int = 3) -> Feature:
    """A regionprops feature for computing perimeter or surface area.

    Args:
        ndim (int): The number of dimensions of the tracks. Controls the display
            name.

    Returns:
        Feature: A feature dict representing perimeter/surface area
    """
    return {
        "feature_type": "node",
        "value_type": "float",
        "num_values": 1,
        "valid_ndim": (3, 4),
        "display_name": "Perimeter" if ndim == 3 else "Surface Area",
        "recompute": True,
        "required": True,
        "default_value": None,
    }
