import numpy as np
import pytest
from skimage.draw import disk, ellipsoid
from skimage.measure import label
from funtracks.features.regionprops_extended import regionprops_extended

@pytest.fixture
def sample_data_2d():
    """Create a sample 2D disk image"""

    circ_img = np.zeros((200, 200), dtype=np.uint8)
    rr, cc = disk((100, 100), 50)  # Center the disk at (100, 100) with a radius of 50
    circ_img[rr, cc] = 1
    spacing = (1, 1)

    return circ_img, spacing

@pytest.fixture 
def sample_data_3d():
    """Create a sample 3D ellipsoid image"""

    ellip = ellipsoid(6, 10, 16, levelset=False)
    ellip_img = label(ellip)
    spacing = (1.0, 1.0, 1.0)

    return ellip_img, spacing

def test_axes(sample_data_3d):
    ellip_img, spacing = sample_data_3d
    props = regionprops_extended(ellip_img, spacing)
    expected_axes = (16, 10, 6)
    for prop in props:
        axes = prop.axes
        assert len(axes) == 3
        assert axes == pytest.approx(expected_axes, rel=1.e0)

def test_circularity(sample_data_2d):
    circ_img, spacing = sample_data_2d
    props = regionprops_extended(circ_img, spacing)
    for prop in props:
        circularity = prop.circularity
        assert isinstance(circularity, float)
        assert circularity == pytest.approx(1, rel=1e-1) 

def test_voxel_count(sample_data_3d):
    ellip_img, spacing = sample_data_3d
    props = regionprops_extended(ellip_img, spacing)
    for prop in props:
        voxel_count = prop.voxel_count
        assert isinstance(voxel_count, int)
        assert voxel_count == 3985

def test_surface_area(sample_data_3d):
    ellip_img, spacing = sample_data_3d
    props = regionprops_extended(ellip_img, spacing)
    for prop in props:
        surface_area = prop.surface_area
        assert surface_area.dtype == np.float32
        assert surface_area == pytest.approx(1383.6, abs=160) # measuring surface area with marching cubes (1485) overestimates the surface compared to the theoretical value (1383.6). 
    
    props = regionprops_extended(ellip_img, spacing=(2.0, 1.0, 1.0))
    for prop in props:
        surface_area = prop.surface_area
        assert isinstance(surface_area, float)
        assert surface_area == pytest.approx(1383.6*2**(2/3), abs=150) # theoretical value should be 2**(2/3) times 1383.6 but we need a tolerance because of marching cubes discretization errors
       
def test_sphericity(sample_data_3d):
    ellip_img, spacing = sample_data_3d
    props = regionprops_extended(ellip_img, spacing)
    for prop in props:
        sphericity = prop.sphericity
        assert isinstance(sphericity, float)
        assert sphericity == pytest.approx(0.879, abs=1.e-1) # theoretical value computed based on equivalent sphere with radius 9.84 and volume 3985, having a surface area of 1215.5

def test_volume(sample_data_3d):
    ellip_img, spacing = sample_data_3d
    props = regionprops_extended(ellip_img, spacing)
    for prop in props:
        volume = prop.volume
        assert isinstance(volume, float)
        assert volume == 3985
    
    props = regionprops_extended(ellip_img, spacing=(2.0, 1.0, 1.0))
    for prop in props:
        volume = prop.volume
        assert isinstance(volume, float)
        assert volume == 3985*2