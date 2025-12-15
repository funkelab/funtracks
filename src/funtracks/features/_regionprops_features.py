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
    }


def EllipsoidAxes(ndim: int | None = 4) -> Feature:
    """A regionprops feature for computing the ellipsoid axis radii.

    Args:
        ndim (int | None): The number of dimensions including time.
            For ndim=3 (2D+time), returns 2 axis values (major, minor).
            For ndim=4 (3D+time), returns 3 axis values (major, semi-minor, minor).
            If None, defaults to 4 to include all possible axis names.

    Returns:
        Feature: A feature dict representing ellipsoid axes
    """
    if ndim is None:
        ndim = 4
    num_spatial_dims = ndim - 1
    if num_spatial_dims == 2:
        display_name = "Ellipse axis radii"
        value_names = ["major_axis", "minor_axis"]
    else:
        display_name = "Ellipsoid axis radii"
        value_names = ["major_axis", "semi_minor_axis", "minor_axis"]
    return {
        "feature_type": "node",
        "value_type": "float",
        "num_values": num_spatial_dims,
        "display_name": display_name,
        "value_names": value_names,
        "required": True,
        "default_value": None,
        "spatial_dims": True,
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
    }
