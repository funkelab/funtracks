import math

import numpy as np
from skimage.measure import marching_cubes, mesh_surface_area, regionprops
from skimage.measure._regionprops import RegionProperties


class ExtendedRegionProperties(RegionProperties):
    """Adding additional properties to skimage.measure._regionprops following the logic
    from the porespy package with some modifications to include the spacing information.
    """

    @property
    def axes(self):
        """
        Calculate the three axes radii of the fitted ellipsoid.

        This method calculates the principal axes of inertia for the label region, which
        are used to determine the lengths of the axes radii of an ellipsoid that
        approximates the shape of the region.

        Returns:
            tuple: A tuple containing the lengths of the three principal axes radii
            (longr, midr, shortr).

        """

        if self._label_image.ndim == 3:
            # Extract the coordinates of the region's voxels
            cell = np.where(self._label_image == self.label)
            voxel_count = self.voxel_count

            z, y, x = cell

            # Center the coordinates and apply spacing
            z = (z - np.mean(z)) * self._spacing[0]
            y = (y - np.mean(y)) * self._spacing[1]
            x = (x - np.mean(x)) * self._spacing[2]

            # Calculate the elements of the inertia tensor
            i_xx = np.sum(y**2 + z**2)
            i_yy = np.sum(x**2 + z**2)
            i_zz = np.sum(x**2 + y**2)
            i_xy = np.sum(x * y)
            i_xz = np.sum(x * z)
            i_yz = np.sum(y * z)
            i = np.array(
                [[i_xx, -i_xy, -i_xz], [-i_xy, i_yy, -i_yz], [-i_xz, -i_yz, i_zz]]
            )

            # Compute the eigenvalues and eigenvectors of the inertia tensor. The
            # eigenvalues of the inertia tensor represent the principal moments of
            # inertia, and the eigenvectors represent the directions of the principal
            # axes.
            eig = np.linalg.eig(i)
            eigval = eig[0]

            # Identify the principal axes
            longaxis = np.where(np.min(eigval) == eigval)[0][0]
            shortaxis = np.where(np.max(eigval) == eigval)[0][0]
            midaxis = (
                0
                if shortaxis != 0 and longaxis != 0
                else 1
                if shortaxis != 1 and longaxis != 1
                else 2
            )

            # Calculate the lengths of the principal axes
            longr = math.sqrt(
                5.0
                / 2.0
                * (eigval[midaxis] + eigval[shortaxis] - eigval[longaxis])
                / voxel_count
            )
            midr = math.sqrt(
                5.0
                / 2.0
                * (eigval[shortaxis] + eigval[longaxis] - eigval[midaxis])
                / voxel_count
            )
            shortr = math.sqrt(
                5.0
                / 2.0
                * (eigval[longaxis] + eigval[midaxis] - eigval[shortaxis])
                / voxel_count
            )

            return (longr, midr, shortr)

        elif self._label_image.ndim == 2:
            return (self.axis_major_length, self.axis_minor_length)

    @property
    def circularity(self):
        """
        Calculate the circularity or sphericity of the region.

        For 2D regions, circularity is defined as 4π * (Area / Perimeter^2).
        For 3D regions, sphericity is defined as the ratio of the surface area of
        a sphere with the same volume as the region to the surface area of the region.

        Returns:
            float: The circularity (2D) or sphericity (3D) of the region.
        """
        if self._label_image.ndim == 2:
            return 4 * math.pi * self.area / self.perimeter**2
        else:  # 3D
            vol = self.volume
            r = (3 / 4 / np.pi * vol) ** (1 / 3)
            a_equiv = 4 * np.pi * r**2
            a_region = self.perimeter  # perimeter returns surface_area for 3D
            return a_equiv / a_region

    @property
    def pixel_count(self):
        """
        Get the number of pixels in the region.

        Returns:
            int: The number of pixels in the region.
        """
        return self.voxel_count

    @property
    def perimeter(self):
        """
        Calculate the perimeter or surface area of the region.

        For 2D regions, returns the perimeter.
        For 3D regions, returns the surface area calculated using the marching cubes
        algorithm to extract a 2D surface mesh from the 3D volume. The mesh surface
        area is calculated with skimage.measure.mesh_surface_area.

        Returns:
            float: The perimeter (2D) or surface area (3D) of the region.
        """
        if self._label_image.ndim == 2:
            return super().perimeter
        else:  # 3D
            verts, faces, _, _ = marching_cubes(
                self._label_image == self.label, level=0.5, spacing=self._spacing
            )
            return mesh_surface_area(verts, faces)

    @property
    def volume(self):
        """
        Calculate the volume of the region.

        The volume is calculated as the number of voxels in the region multiplied by the
        product of the spacing in each dimension.

        Returns:
            float: The volume of the region.
        """
        vol = np.sum(self._label_image == self.label) * np.prod(self._spacing)
        return vol

    @property
    def voxel_count(self):
        """
        Get the number of voxels in the region.

        Returns:
            int: The number of voxels in the region.
        """
        voxel_count = int(np.sum(self._label_image == self.label))
        return voxel_count


def regionprops_extended(
    img: np.ndarray,
    spacing: tuple[float, ...] | None,
    intensity_image: np.ndarray | None = None,
) -> list[ExtendedRegionProperties]:
    """
    Create instances of ExtendedRegionProperties that extend
    skimage.measure.RegionProperties.

    Args:
        img (np.ndarray): The labeled image.
        spacing (tuple[float, ...]| None): The spacing between voxels in each dimension.
            If None, each voxel is assumed to be 1 in all dimensions.
        intensity_image (np.ndarray, optional): The intensity image.

    Returns:
        list[ExtendedRegionProperties]: A list of ExtendedRegionProperties instances.
    """
    results = regionprops(img, intensity_image=intensity_image, spacing=spacing)
    for i, _ in enumerate(results):
        a = results[i]
        b = ExtendedRegionProperties(
            slice=a.slice,
            label=a.label,
            label_image=a._label_image,
            intensity_image=a._intensity_image,
            cache_active=a._cache_active,
            spacing=a._spacing,
        )
        results[i] = b

    return results
