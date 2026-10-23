"""Intensity image handling shared by the annotators that measure mean intensity.

Mean intensity is measured in a region per node: the segmentation mask when there is
one (``RegionpropsAnnotator``), or a disk/sphere around the point otherwise
(``PointIntensityAnnotator``). In both cases the region is converted to a Mask. All
functions related to validating, reading, and stacking the intensity images lives here.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, TypeAlias

import numpy as np
from tracksdata.nodes import Mask

from funtracks.features import Intensity

if TYPE_CHECKING:
    import dask.array as da

    from funtracks.data_model import Tracks
    from funtracks.features import Feature

    IntensityImage: TypeAlias = np.ndarray | da.Array


def _bbox_slicing(mask: Mask) -> tuple[slice, ...]:
    """The spatial slices covering a mask's bounding box, one per spatial axis."""
    ndim = mask.mask.ndim
    bbox = mask.bbox
    return tuple(slice(bbox[i], bbox[i + ndim]) for i in range(ndim))


def _as_intensity_image(crops: list[np.ndarray]) -> np.ndarray:
    """Combine one crop per channel into the intensity image skimage expects.

    Several channels are stacked on a trailing axis, which skimage reads as a
    multichannel intensity image and answers with one mean per channel.
    """
    return crops[0] if len(crops) == 1 else np.stack(crops, axis=-1)


class _FrameCache:
    """Serves bounding box crops for many nodes out of one materialized time point.

    ``compute`` walks nodes in time order, so holding the current frame lets every node
    in it be cropped from memory.

    Only the current time point is held (one frame per channel), and the cache is local
    to a single ``compute`` call, so nothing is retained afterwards.
    """

    def __init__(self, intensity_images: list[IntensityImage] | None):
        self._images = intensity_images
        self._time: int | None = None
        self._frames: list[np.ndarray] = []

    def crop(self, mask: Mask, time: int | None) -> np.ndarray | None:
        """The intensity image for one mask, read from the cached time point."""
        if self._images is None or time is None:
            return None
        if time != self._time:
            self._frames = [np.asarray(image[time]) for image in self._images]
            self._time = time
        slicing = _bbox_slicing(mask)
        return _as_intensity_image([frame[slicing] for frame in self._frames])


def _intensity_crop(
    intensity_images: list[IntensityImage] | None, mask: Mask, time: int | None
) -> np.ndarray | None:
    """Crop the intensity image(s) of one time point to a mask's bounding box.

    For updating a single node, where caching a whole frame would be wasted.

    Args:
        intensity_images: The images to crop, one per channel.
        mask: The mask defining the bounding box to crop to.
        time: The time point to take the intensity frame from.

    Returns:
        The cropped intensity image, shaped like the mask with a trailing channel
        axis when there is more than one channel, or None if no image is set.
    """
    if intensity_images is None or time is None:
        return None
    # Slice the time point and the bounding box in a single indexing operation, so
    # a store that supports it fetches only the box instead of the whole frame.
    slicing = (time, *_bbox_slicing(mask))
    return _as_intensity_image([np.asarray(image[slicing]) for image in intensity_images])


def _times_in_order(tracks: Tracks, node_ids: list[int]) -> tuple[list[int], list[int]]:
    """The nodes sorted by time, with their times, so each frame is read only once.

    The times are fetched in one bulk query rather than one graph lookup per node.
    """
    in_time_order = sorted(
        zip(node_ids, tracks.get_times(node_ids), strict=True),
        key=lambda pair: pair[1],
    )
    return (
        [node_id for node_id, _ in in_time_order],
        [time for _, time in in_time_order],
    )


class IntensityImagesMixin:
    """Holds the raw images an annotator measures mean intensity in.

    Expects the host annotator to provide ``tracks``, ``all_features``,
    ``intensity_key`` and ``compute``, as ``GraphAnnotator`` subclasses do.
    """

    tracks: Tracks
    all_features: dict[str, tuple[Feature, bool]]
    intensity_key: str
    intensity_images: list[IntensityImage] | None
    channel_names: list[str] | None

    if TYPE_CHECKING:
        # Provided by the annotator; declared only for type checking, so that it
        # cannot shadow the annotator's own compute in the method resolution order.
        def compute(self, feature_keys: list[str] | None = None) -> None: ...

    def _validate_intensity_images(
        self,
        tracks: Tracks,
        intensity_images: Sequence[IntensityImage] | None,
        channel_names: Sequence[str] | None,
    ) -> None:
        """Validate and store the intensity images and channel names.

        With a segmentation, every image must match its shape. Without one, every image
        must have the dimensions of the tracks (time first), and all must share a shape.

        Args:
            tracks: The tracks, used to validate the image shapes.
            intensity_images: raw images to measure intensity on.
            channel_names: Optional display names.

        Raises:
            TypeError: If a single array is passed instead of a sequence of them.
            ValueError: If the image shapes do not fit the tracks or each other, or if
                the number of channel names does not match the number of channels.
        """
        if hasattr(intensity_images, "shape"):
            raise TypeError(
                "intensity_images takes one image per channel: pass [image], not a "
                "bare array"
            )
        if not intensity_images:
            self.intensity_images = None
            self.channel_names = None
            return

        images = list(intensity_images)
        if tracks.segmentation is not None:
            seg_shape = tuple(tracks.segmentation.shape)
            for image in images:
                if tuple(image.shape) != seg_shape:
                    raise ValueError(
                        f"Intensity image shape {tuple(image.shape)} does not match "
                        f"the segmentation shape {seg_shape}"
                    )
        else:
            first_shape = tuple(images[0].shape)
            for image in images:
                if len(image.shape) != tracks.ndim:
                    raise ValueError(
                        f"Intensity image shape {tuple(image.shape)} does not have "
                        f"the {tracks.ndim} dimensions of the tracks (t, [z], y, x)"
                    )
                if tuple(image.shape) != first_shape:
                    raise ValueError(
                        f"Intensity images have different shapes: "
                        f"{tuple(image.shape)} and {first_shape}"
                    )

        num_channels = len(images)
        if channel_names is None:
            names = (
                None
                if num_channels == 1
                else [f"channel_{i}" for i in range(num_channels)]
            )
        else:
            if len(channel_names) != num_channels:
                raise ValueError(
                    f"Got {len(channel_names)} channel names for {num_channels} "
                    "intensity channels"
                )
            names = list(channel_names)

        self.intensity_images = images
        self.channel_names = names

    def _intensity_feature(self) -> Feature:
        """The intensity Feature for the current channels."""
        return Intensity(self.channel_names)

    def set_intensity_images(
        self,
        intensity_images: Sequence[IntensityImage] | None,
        channel_names: Sequence[str] | None = None,
    ) -> None:
        """Attach (or clear) the intensity images used to compute the intensity feature.

        ``Tracks`` builds this annotator before any raw image is known, so this is the
        normal way to supply them. If the intensity feature is already enabled, it is
        brought up to date here: re-registered when the number of channels changed
        (the column holds one value per channel), recomputed when the images changed,
        and left alone when only the channel names differ.

        Args:
            intensity_images: Raw images, one per channel, each shaped like the
                segmentation, or ``(t, [z], y, x)`` without one. Pass None (or an empty
                list) to clear.
            channel_names: Display names, one per intensity image.
        """
        previous_images = self.intensity_images
        previous_feature, included = self.all_features[self.intensity_key]

        self._validate_intensity_images(self.tracks, intensity_images, channel_names)

        # Rebuild the intensity Feature: its num_values follows the channel count.
        feature = self._intensity_feature()
        self.all_features[self.intensity_key] = (feature, included)

        if not included or self.intensity_key not in self.tracks.features:
            return

        if feature["num_values"] != previous_feature["num_values"]:
            # The column shape changed, so it has to be dropped and rebuilt
            self.tracks.disable_features([self.intensity_key])
            self.tracks.enable_features([self.intensity_key])
            return

        self.tracks.features[self.intensity_key] = feature
        if not self._is_same_image(previous_images, self.intensity_images):
            self.compute([self.intensity_key])

    @staticmethod
    def _is_same_image(
        previous: list[IntensityImage] | None, current: list[IntensityImage] | None
    ) -> bool:
        """Whether two intensity inputs are the very same images, channel for channel.

        Compared by identity: renaming a channel should not trigger a recompute, but
        swapping in a different image (even an equal-looking one) should.
        """
        if previous is None or current is None:
            return previous is current
        return len(previous) == len(current) and all(
            before is after for before, after in zip(previous, current, strict=True)
        )
