from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from tracksdata.nodes import Mask

from funtracks.actions.add_delete_node import AddNode
from funtracks.actions.update_node_attrs import UpdateNodeAttrs
from funtracks.features import Feature, Intensity

from ._graph_annotator import GraphAnnotator
from ._intensity_images import (
    IntensityImagesMixin,
    _FrameCache,
    _intensity_crop,
    _times_in_order,
)
from ._regionprops_annotator import DEFAULT_INTENSITY_KEY

if TYPE_CHECKING:
    from funtracks.actions import BasicAction
    from funtracks.data_model import Tracks

    from ._intensity_images import IntensityImage

DEFAULT_POINT_DIAMETER = 5.0


def _ball_mask(
    position: Sequence[float],
    diameter: float,
    spacing: np.ndarray,
    frame_shape: tuple[int, ...],
) -> Mask | None:
    """The disk (2D) or sphere (3D) of pixels around a point, clipped to the frame.

    A pixel belongs to the ball if its center lies within ``diameter / 2`` of the point,
    measured in world units, so anisotropic spacing gives an ellipse/ellipsoid in pixel
    space. When the ball is too small to contain a pixel center, the pixel nearest to
    the point is used, so that every point inside the frame gets a measurement.

    Args:
        position: The point in world units, one value per spatial axis.
        diameter: The ball diameter in world units.
        spacing: Pixel spacing per spatial axis.
        frame_shape: The spatial shape of the image the ball is clipped to.

    Returns:
        The ball as a Mask, or None if it does not overlap the frame at all.
    """
    center = np.asarray(position, dtype=float) / spacing
    shape = np.asarray(frame_shape)
    radius = diameter / 2
    radii = radius / spacing

    lo = np.maximum(np.ceil(center - radii).astype(int), 0)
    hi = np.minimum(np.floor(center + radii).astype(int) + 1, shape)
    if np.all(hi > lo):
        grids = np.ogrid[
            tuple(slice(start, stop) for start, stop in zip(lo, hi, strict=True))
        ]
        dist2 = sum(
            ((grid - c) * s) ** 2
            for grid, c, s in zip(grids, center, spacing, strict=True)
        )
        inside = dist2 <= radius**2
        if inside.any():
            return Mask(inside, bbox=np.concatenate([lo, hi]))

    nearest = np.round(center).astype(int)
    if np.any(nearest < 0) or np.any(nearest >= shape):
        return None
    return Mask(
        np.ones((1,) * len(nearest), dtype=bool),
        bbox=np.concatenate([nearest, nearest + 1]),
    )


def _mean_intensity(mask: Mask, intensity_image: np.ndarray) -> Any:
    """The mean intensity inside a mask, one value per channel.

    Matches what ``RegionpropsAnnotator`` stores for ``intensity_mean``: a float for a
    single channel, a list of floats (from the trailing channel axis) for several.
    """
    mean = intensity_image[mask.mask].mean(axis=0)
    if np.ndim(mean) == 0:
        return float(mean)
    return [float(v) for v in mean]


class PointIntensityAnnotator(IntensityImagesMixin, GraphAnnotator):
    """Measures mean intensity around each point, for tracks without a segmentation.

    Intensity is measured in a disk (2D) or sphere (3D) of a user-chosen diameter, in
    world units, centered on the node position.
    """

    @classmethod
    def can_annotate(cls, tracks) -> bool:
        """Point-based intensity is only for tracks without a segmentation.

        With a segmentation, ``RegionpropsAnnotator`` measures inside the mask instead.
        """
        return tracks.segmentation is None

    def __init__(
        self,
        tracks: Tracks,
        intensity_images: Sequence[IntensityImage] | None = None,
        channel_names: Sequence[str] | None = None,
        diameter: float = DEFAULT_POINT_DIAMETER,
    ):
        """
        Args:
            tracks: The tracks to compute intensity for.
            intensity_images: Optional raw images to measure intensity in, one per
                channel, each shaped ``(t, [z], y, x)``.
            channel_names: Optional display names, one per intensity image.
            diameter: Diameter of the disk/sphere to measure in, in world units
        """
        self.intensity_key = DEFAULT_INTENSITY_KEY
        self.diameter = self._validate_diameter(diameter)

        self.intensity_images: list[IntensityImage] | None = None
        self.channel_names: list[str] | None = None
        self._validate_intensity_images(tracks, intensity_images, channel_names)

        super().__init__(tracks, {self.intensity_key: Intensity(self.channel_names)})

    @classmethod
    def get_available_features(
        cls, ndim: int = 3, channel_names: Sequence[str] | None = None
    ) -> dict[str, Feature]:
        """Get all features that can be computed by this annotator.

        Args:
            ndim: Total number of dimensions including time (3 or 4). Unused, for
                consistency with the other annotators.
            channel_names: Display names for the intensity channels.

        Returns:
            Dictionary mapping feature keys to Feature definitions.
        """
        return {DEFAULT_INTENSITY_KEY: Intensity(channel_names)}

    @staticmethod
    def _validate_diameter(diameter: float) -> float:
        diameter = float(diameter)
        if not np.isfinite(diameter) or diameter <= 0:
            raise ValueError(f"Diameter must be a positive number, got {diameter}")
        return diameter

    def set_diameter(self, diameter: float) -> None:
        """Change the disk/sphere diameter, recomputing intensity if it is enabled.

        Args:
            diameter: The new diameter, in world units.
        """
        diameter = self._validate_diameter(diameter)
        if diameter == self.diameter:
            return
        self.diameter = diameter
        if self.intensity_key in self.features:
            self.compute([self.intensity_key])

    @property
    def _spacing(self) -> np.ndarray:
        """Pixel spacing per spatial axis, to turn world positions into pixels."""
        if self.tracks.scale is None:
            return np.ones(self.tracks.ndim - 1)
        return np.asarray(self.tracks.scale[1:], dtype=float)

    def _images_to_measure(self) -> list[IntensityImage] | None:
        """The images to measure in, or None (with a warning) if there are none."""
        if self.intensity_images is not None:
            return self.intensity_images
        warnings.warn(
            f"Cannot compute {self.intensity_key!r}: no intensity image is set on "
            "the PointIntensityAnnotator. Call set_intensity_images() first.",
            stacklevel=3,
        )
        return None

    def compute(self, feature_keys: list[str] | None = None) -> None:
        """Compute mean intensity around every node and add it to the tracks.

        Args:
            feature_keys: Optional list of specific feature keys to compute.
                If None, computes all currently active features. Keys not in
                self.features (not enabled) are ignored.
        """
        if not self._filter_feature_keys(feature_keys):
            return
        images = self._images_to_measure()
        if images is None:
            return

        node_ids = [
            node_id for node_id in self.graph.node_ids() if self.graph.has_node(node_id)
        ]
        if not node_ids:
            return
        # Walk the nodes in time order so that each frame is read once and serves
        # every node in it.
        node_ids, times = _times_in_order(self.tracks, node_ids)
        positions = self.tracks.get_positions(node_ids)
        spacing = self._spacing
        frame_shape = tuple(images[0].shape[1:])
        frames = _FrameCache(images)

        values: list[Any] = []
        for position, time in zip(positions, times, strict=True):
            mask = _ball_mask(position, self.diameter, spacing, frame_shape)
            if mask is None:
                values.append(None)
                continue
            values.append(_mean_intensity(mask, frames.crop(mask, time)))

        self.tracks._set_nodes_attrs(node_ids, {self.intensity_key: values})

    def _compute_node(self, node: int, images: list[IntensityImage]) -> None:
        """Measure the intensity around a single node."""
        mask = _ball_mask(
            self.tracks.get_position(node),
            self.diameter,
            self._spacing,
            tuple(images[0].shape[1:]),
        )
        value = None
        if mask is not None:
            time = self.tracks.get_time(node)
            value = _mean_intensity(mask, _intensity_crop(images, mask, time))
        self.tracks._set_node_attr(node, self.intensity_key, value)

    def update(self, action: BasicAction) -> None:
        """Remeasure a node when it is added or moved.

        Args:
            action (BasicAction): The action that triggered this update
        """
        moved = isinstance(action, UpdateNodeAttrs) and self._moves_node(action)
        if not (isinstance(action, AddNode) or moved):
            return
        if not self._filter_feature_keys(None):
            return
        images = self._images_to_measure()
        if images is not None:
            self._compute_node(action.node, images)

    def _moves_node(self, action: UpdateNodeAttrs) -> bool:
        """Whether an attribute update changes the node position."""
        position_key = self.tracks.features.position_key
        pos_keys = [position_key] if isinstance(position_key, str) else position_key
        return any(key in action.new_attrs for key in pos_keys or [])

    def change_key(self, old_key: str, new_key: str) -> None:
        """Rename a feature key, keeping the intensity key in sync.

        Args:
            old_key: Existing key to rename.
            new_key: New key to replace it with.
        """
        super().change_key(old_key, new_key)
        if old_key == self.intensity_key:
            self.intensity_key = new_key
