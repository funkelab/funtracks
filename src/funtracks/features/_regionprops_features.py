from __future__ import annotations

from typing import TYPE_CHECKING

from ._feature import Feature

if TYPE_CHECKING:
    pass


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
        "display_name": "Area" if ndim == 3 else "Volume",
        "required": True,
        "default_value": None,
        "is_group": False,
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
        "display_name": "Intensity",
        "required": True,
        "default_value": None,
        "is_group": False,
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
        "display_name": "Ellipse axis radii" if ndim == 3 else "Ellipsoid axis radii",
        "required": True,
        "default_value": None,
        "is_group": False,
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
        "display_name": "Circularity" if ndim == 3 else "Sphericity",
        "required": True,
        "default_value": None,
        "is_group": False,
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
        "display_name": "Perimeter" if ndim == 3 else "Surface Area",
        "required": True,
        "default_value": None,
        "is_group": False,
    }
