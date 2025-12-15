from funtracks.features import (
    Area,
    Circularity,
    EllipsoidAxes,
    Perimeter,
    Position,
    Time,
)


def test_time_feature():
    """Test that Time() returns a valid Feature TypedDict"""
    feat = Time()
    assert feat["feature_type"] == "node"
    assert feat["value_type"] == "int"
    assert feat["num_values"] == 1
    assert feat["display_name"] == "Time"
    assert feat["required"] is True
    assert feat["default_value"] is None


def test_position_feature():
    """Test that Position() returns a valid Feature TypedDict"""
    feat = Position(axes=["y", "x"])
    assert feat["feature_type"] == "node"
    assert feat["value_type"] == "float"
    assert feat["num_values"] == 2
    assert feat["display_name"] == "position"
    assert feat["value_names"] == ["y", "x"]
    assert feat["required"] is True
    assert feat["default_value"] is None


def test_area_feature():
    """Test that Area() returns a valid Feature TypedDict"""
    feat = Area(ndim=3)
    assert feat["feature_type"] == "node"
    assert feat["value_type"] == "float"
    assert feat["num_values"] == 1
    assert feat["display_name"] == "Area"
    assert feat["required"] is True
    assert feat["default_value"] is None

    feat = Area(ndim=4)
    assert feat["display_name"] == "Volume"


def test_ellipsoid_axes_feature():
    """Test that EllipsoidAxes() returns a valid Feature TypedDict"""
    # ndim=3 means 2D+time -> 2 spatial dimensions
    feat = EllipsoidAxes(ndim=3)
    assert feat["feature_type"] == "node"
    assert feat["value_type"] == "float"
    assert feat["num_values"] == 2
    assert feat["display_name"] == "Ellipse axis radii"
    assert feat["value_names"] == ["major_axis", "minor_axis"]

    # ndim=4 means 3D+time -> 3 spatial dimensions
    feat = EllipsoidAxes(ndim=4)
    assert feat["num_values"] == 3
    assert feat["display_name"] == "Ellipsoid axis radii"
    assert feat["value_names"] == ["major_axis", "semi_minor_axis", "minor_axis"]


def test_circularity_feature():
    """Test that Circularity() returns a valid Feature TypedDict"""
    feat = Circularity(ndim=3)
    assert feat["feature_type"] == "node"
    assert feat["value_type"] == "float"
    assert feat["num_values"] == 1
    assert feat["display_name"] == "Circularity"

    feat = Circularity(ndim=4)
    assert feat["display_name"] == "Sphericity"


def test_perimeter_feature():
    """Test that Perimeter() returns a valid Feature TypedDict"""
    feat = Perimeter(ndim=3)
    assert feat["feature_type"] == "node"
    assert feat["value_type"] == "float"
    assert feat["num_values"] == 1
    assert feat["display_name"] == "Perimeter"

    feat = Perimeter(ndim=4)
    assert feat["display_name"] == "Surface Area"


def test_feature_as_dict():
    """Test that Features are valid dicts"""
    feat = Time()
    assert isinstance(feat, dict)
    assert "feature_type" in feat
    assert "value_type" in feat

    # Can convert to regular dict
    regular_dict = dict(feat)
    assert regular_dict == feat
