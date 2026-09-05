from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, NamedTuple, TypeAlias

import numpy as np
from tracksdata.nodes import Mask

from funtracks.actions.add_delete_node import AddNode
from funtracks.actions.update_segmentation import UpdateNodeSeg
from funtracks.features import (
    Area,
    Circularity,
    EllipsoidAxes,
    Feature,
    Intensity,
    Perimeter,
    Position,
)

from ._graph_annotator import GraphAnnotator
from ._regionprops_extended import regionprops_extended

if TYPE_CHECKING:
    import dask.array as da

    from funtracks.actions import BasicAction
    from funtracks.data_model import Tracks

    IntensityImage: TypeAlias = np.ndarray | da.Array

DEFAULT_POS_KEY = "pos"
DEFAULT_AREA_KEY = "area"
DEFAULT_ELLIPSE_AXIS_KEY = "ellipse_axis_radii"
DEFAULT_CIRCULARITY_KEY = "circularity"
DEFAULT_PERIMETER_KEY = "perimeter"
DEFAULT_INTENSITY_KEY = "intensity"


def _centroid(mask: Mask, spacing: tuple[float, ...] | None) -> list[float]:
    """Centroid in world units, read directly from the mask array.

    Equivalent to ``ExtendedRegionProperties.centroid`` (``(local_centroid + bbox_min)
    * spacing``) but skips the skimage regionprops machinery (find_objects, region
    caching), which is wasted overhead when only the centroid is needed.

    Args:
        mask: A Mask object representing one detection.
        spacing: Voxel spacing per spatial dimension, or None for unit spacing.

    Returns:
        The centroid coordinates, one float per spatial dimension.
    """
    arr = mask.mask
    bbox_min = mask.bbox[: arr.ndim]
    local = np.array([idx.mean() for idx in np.nonzero(arr)])
    world = local + bbox_min
    if spacing is not None:
        world = world * np.asarray(spacing)
    return [float(v) for v in world]


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


def _to_attr_value(value: Any) -> Any:
    """Convert a regionprops value into something the graph backend can store.

    Multi-valued properties (centroid, axes, per-channel intensity means) become
    plain lists of floats; numpy scalars become floats. Anything else is passed
    through unchanged.
    """
    if isinstance(value, np.ndarray):
        return [float(v) for v in value.ravel()]
    if isinstance(value, tuple):
        # cannot be a list of np.arrays with single values
        return [float(v) for v in value]
    if isinstance(value, np.floating | np.integer):
        return float(value)
    return value


class FeatureSpec(NamedTuple):
    """Specification for a regionprops feature.

    Attributes:
        key: The key to use in the graph attributes and feature dict
        feature: The Feature TypedDict definition
        regionprops_attr: The name of the corresponding regionprops attribute
    """

    key: str
    feature: Feature
    regionprops_attr: str


class RegionpropsAnnotator(GraphAnnotator):
    """A graph annotator using regionprops to extract node features from segmentations.

    The possible features include:
    - centroid (to use as node position)
    - area/volume
    - ellipsoid major/minor/semi-minor axes
    - circularity/sphericity
    - perimeter/surface area
    - mean intensity (requires an intensity image, see ``set_intensity_images``)

    Defaults to computing all features, but individual ones can be turned off by changing
    the self.include value at the corresponding index to the feature in self.features.
    """

    @classmethod
    def can_annotate(cls, tracks) -> bool:
        """Check if this annotator can annotate the given tracks.

        Requires segmentation data to be present.

        Args:
            tracks: The tracks to check compatibility with

        Returns:
            True if tracks have segmentation, False otherwise
        """
        return tracks.segmentation is not None

    def __init__(
        self,
        tracks: Tracks,
        pos_key: str | None = DEFAULT_POS_KEY,
        intensity_images: Sequence[IntensityImage] | None = None,
        channel_names: Sequence[str] | None = None,
    ):
        """
        Args:
            tracks: The tracks to compute features for.
            pos_key: Graph attribute to write the centroid to.
            intensity_images: Optional raw images to measure intensity in, one per
                channel, each shaped like the segmentation ``(t, [z], y, x)``. The
                intensity feature holds one mean per channel. The images are held as
                given (lazy arrays stay lazy) and read one node bounding box at a
                time. If None or empty, the intensity feature is advertised but
                skipped (with a warning) when requested.
            channel_names: Optional display names, one per intensity image. Defaults
                to ``channel_0``, ``channel_1``, ... for multichannel input.
        """
        self.pos_key: str = pos_key if pos_key is not None else DEFAULT_POS_KEY
        self.area_key = DEFAULT_AREA_KEY
        self.ellipse_axis_radii_key = DEFAULT_ELLIPSE_AXIS_KEY
        self.circularity_key = DEFAULT_CIRCULARITY_KEY
        self.perimeter_key = DEFAULT_PERIMETER_KEY
        self.intensity_key = DEFAULT_INTENSITY_KEY

        # One image per intensity channel; stacked per node, not up front
        self.intensity_images: list[IntensityImage] | None = None
        self.channel_names: list[str] | None = None
        self._validate_intensity_images(tracks, intensity_images, channel_names)

        specs = RegionpropsAnnotator._define_features(
            tracks.ndim,
            self.channel_names,
        )
        # update position key in spec
        if self.pos_key != DEFAULT_POS_KEY:
            for feat in specs:
                if feat.key == DEFAULT_POS_KEY:
                    specs.remove(feat)
                    new_feat = FeatureSpec(
                        self.pos_key, feat.feature, feat.regionprops_attr
                    )
                    specs.append(new_feat)
                    break

        feats = {spec.key: spec.feature for spec in specs}
        super().__init__(tracks, feats)
        # Build regionprops name mapping from specs
        self.regionprops_names = {spec.key: spec.regionprops_attr for spec in specs}

    def _validate_intensity_images(
        self,
        tracks: Tracks,
        intensity_images: Sequence[IntensityImage] | None,
        channel_names: Sequence[str] | None,
    ) -> None:
        """Validate and store the intensity images and channel names.

        Args:
            tracks: The tracks, used to validate the image shapes against the
                segmentation.
            intensity_images: raw images to measure intensity on.
            channel_names: Optional display names.

        Raises:
            TypeError: If a single array is passed instead of a sequence of them.
            ValueError: If an image does not match the segmentation shape, or if the
                number of channel names does not match the number of channels.
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
        seg_shape = tracks.segmentation.shape if tracks.segmentation is not None else None
        if seg_shape is not None:
            for image in images:
                if tuple(image.shape) != tuple(seg_shape):
                    raise ValueError(
                        f"Intensity image shape {tuple(image.shape)} does not match "
                        f"the segmentation shape {tuple(seg_shape)}"
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
            intensity_images: See ``__init__``. Pass None (or an empty list) to clear.
            channel_names: See ``__init__``.
        """
        previous_images = self.intensity_images
        previous_feature, included = self.all_features[self.intensity_key]

        self._validate_intensity_images(self.tracks, intensity_images, channel_names)

        # Rebuild the intensity Feature: its num_values follows the channel count.
        feature = Intensity(self.channel_names)
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

    @classmethod
    def _define_features(
        cls,
        ndim: int,
        channel_names: Sequence[str] | None = None,
    ) -> list[FeatureSpec]:
        """Define all supported regionprops features along with keys and function names.

        Single source of truth for feature definitions. Returns FeatureSpec objects
        that include the regionprops attribute mapping needed for computation.

        Args:
            ndim: Total number of dimensions including time (3 or 4)
            channel_names: Display names for the intensity channels, one per channel.
                Controls how many values the intensity feature holds.

        Returns:
            list[FeatureSpec]: List of feature specifications with key, feature,
                and regionprops attribute name.
        """
        # Derive axis names from ndim (spatial dimensions only, no time)
        # Default to 3D when ndim is None to enable matching all position columns
        axis_names = ["z", "y", "x"] if ndim is None or ndim == 4 else ["y", "x"]

        return [
            FeatureSpec(DEFAULT_POS_KEY, Position(axes=axis_names), "centroid"),
            FeatureSpec(DEFAULT_AREA_KEY, Area(ndim=ndim), "area"),
            FeatureSpec(
                DEFAULT_INTENSITY_KEY, Intensity(channel_names), "intensity_mean"
            ),
            FeatureSpec(DEFAULT_ELLIPSE_AXIS_KEY, EllipsoidAxes(ndim=ndim), "axes"),
            FeatureSpec(DEFAULT_CIRCULARITY_KEY, Circularity(ndim=ndim), "circularity"),
            FeatureSpec(DEFAULT_PERIMETER_KEY, Perimeter(ndim=ndim), "perimeter"),
        ]

    @classmethod
    def get_available_features(
        cls, ndim: int = 3, channel_names: Sequence[str] | None = None
    ) -> dict[str, Feature]:
        """Get all features that can be computed by this annotator.

        Returns features with default keys. Custom keys can be specified at
        initialization time.

        Args:
            ndim: Total number of dimensions including time (3 or 4). Defaults to 3.
            channel_names: Display names for the intensity channels. Defaults to a
                single-valued intensity feature.

        Returns:
            Dictionary mapping feature keys to Feature definitions.
        """
        specs = RegionpropsAnnotator._define_features(ndim, channel_names)
        return {spec.key: spec.feature for spec in specs}

    def _intensity_crop(self, mask: Mask, time: int | None) -> np.ndarray | None:
        """Crop the intensity image(s) of one time point to a mask's bounding box.

        skimage requires the intensity image to match the shape of the label image,
        which in this case is the bbox-sized mask array rather than the full frame.

        Args:
            mask: The mask defining the bounding box to crop to.
            time: The time point to take the intensity frame from.

        Returns:
            The cropped intensity image, shaped like the mask with a trailing channel
            axis when there is more than one channel, or None if no image is set.
        """
        if self.intensity_images is None or time is None:
            return None
        # Slice the time point and the bounding box in a single indexing operation, so
        # a store that supports it fetches only the box instead of the whole frame.
        slicing = (time, *_bbox_slicing(mask))
        return _as_intensity_image(
            [np.asarray(image[slicing]) for image in self.intensity_images]
        )

    def compute(self, feature_keys: list[str] | None = None) -> None:
        """Compute the currently included features and add them to the tracks.

        Args:
            feature_keys: Optional list of specific feature keys to compute.
                If None, computes all currently active features. Keys not in
                self.features (not enabled) are ignored.
        """
        # Can only compute features if segmentation is present
        if self.tracks.segmentation is None:
            return

        keys_to_compute = self._filter_feature_keys(feature_keys)
        if not keys_to_compute:
            return
        if self.intensity_key in keys_to_compute and self.intensity_images is None:
            warnings.warn(
                f"Cannot compute {self.intensity_key!r}: no intensity image is set on "
                "the RegionpropsAnnotator. Call set_intensity_images() first.",
                stacklevel=2,
            )
            keys_to_compute = [
                key for key in keys_to_compute if key != self.intensity_key
            ]

        spacing = None if self.tracks.scale is None else tuple(self.tracks.scale[1:])
        all_node_ids = []
        all_values: dict[str, list] = {key: [] for key in keys_to_compute}

        # Position (centroid) is the only feature computed at construction. When it is
        # the sole requested feature, reading it via skimage regionprops pays for a
        # find_objects + RegionProperties build per mask that is pure overhead, so take
        # it straight from the mask array. If any other feature is requested we run the
        # regionprops pass anyway and its centroid comes for free, so pos goes through
        # the normal path with everything else.
        fast_pos = keys_to_compute == [self.pos_key]

        node_ids = [
            node_id for node_id in self.graph.node_ids() if self.graph.has_node(node_id)
        ]
        # Times are only needed for intensity, and stay None otherwise so that the
        # frame cache has nothing to read.
        times: list[int | None] = [None] * len(node_ids)
        if self.intensity_key in keys_to_compute:
            # Fetch the times in one bulk query rather than one graph lookup per node,
            # and walk the nodes in time order so that each frame is read once and
            # serves every node in it.
            in_time_order = sorted(
                zip(node_ids, self.tracks.get_times(node_ids), strict=True),
                key=lambda pair: pair[1],
            )
            node_ids = [node_id for node_id, _ in in_time_order]
            times = [time for _, time in in_time_order]
        frames = _FrameCache(self.intensity_images)

        for node_id, time in zip(node_ids, times, strict=True):
            mask = self.graph.nodes[node_id]["mask"]
            all_node_ids.append(node_id)
            if fast_pos:
                all_values[self.pos_key].append(_centroid(mask, spacing))
                continue
            (region,) = regionprops_extended(
                mask,
                spacing=spacing,
                intensity_image=frames.crop(mask, time),
            )
            for key in keys_to_compute:
                value = _to_attr_value(getattr(region, self.regionprops_names[key]))
                all_values[key].append(value)

        for key in keys_to_compute:
            self.tracks._set_nodes_attr(all_node_ids, key, all_values[key])

    def _regionprops_update(
        self, node_id: int, mask: Mask, feature_keys: list[str], time: int | None = None
    ) -> None:
        """Perform the regionprops computation and update all feature values for a
        single mask.

        Args:
            node_id (int): The node ID to update features for.
            mask (Mask): A Mask object representing one time point
                of segmentation data.
            feature_keys (list): List of feature keys to compute
                (already filtered to enabled).
            time (int | None): The time point of the mask, used to select the intensity
                frame. Looked up from the graph when not provided.
        """
        spacing = None if self.tracks.scale is None else tuple(self.tracks.scale[1:])
        if time is None and self.intensity_key in feature_keys:
            time = self.tracks.get_time(node_id)
        for region in regionprops_extended(
            mask, spacing=spacing, intensity_image=self._intensity_crop(mask, time)
        ):
            # Skip labels that aren't nodes in the graph (e.g., unselected detections)
            if not self.graph.has_node(node_id):
                continue
            for key in feature_keys:
                value = _to_attr_value(getattr(region, self.regionprops_names[key]))
                self.tracks._set_node_attr(node_id, key, value)

    def update(self, action: BasicAction):
        """Update the regionprops features based on the action.

        Only responds to AddNode and UpdateNodeSeg actions that affect segmentation.

        Args:
            action (BasicAction): The action that triggered this update
        """
        # Only update for actions that change segmentation
        if not isinstance(action, (AddNode, UpdateNodeSeg)):
            return

        # Can only compute features if segmentation is present
        if self.tracks.segmentation is None:
            return

        # Get the node from the action
        node = action.node

        keys_to_compute = self._filter_feature_keys(None)
        if not keys_to_compute:
            return
        if self.intensity_key in keys_to_compute and self.intensity_images is None:
            warnings.warn(
                f"Cannot compute {self.intensity_key!r}: no intensity image is set on "
                "the RegionpropsAnnotator. Call set_intensity_images() first.",
                stacklevel=2,
            )
            keys_to_compute = [
                key for key in keys_to_compute if key != self.intensity_key
            ]

        time = self.tracks.get_time(node)

        if self.graph.nodes[node]["mask"].mask.sum() == 0:
            warnings.warn(
                f"Cannot find label {node} in frame {time}: "
                "updating regionprops values to None",
                stacklevel=2,
            )
            for key in keys_to_compute:
                value = None
                self.tracks._set_node_attr(node, key, value)
        else:
            mask = self.graph.nodes[node]["mask"]
            self._regionprops_update(node, mask, keys_to_compute, time=time)

    def change_key(self, old_key: str, new_key: str) -> None:
        """Rename a feature key in this annotator, and related mappings.

        Overrides base implementation to also update the regionprops name mapping.

        Args:
            old_key: Existing key to rename.
            new_key: New key to replace it with.

        Raises:
            KeyError: If old_key does not exist.
        """
        # Call base implementation to update all_features
        super().change_key(old_key, new_key)

        # Update regionprops-specific name mapping
        if old_key in self.regionprops_names:
            rp_name = self.regionprops_names.pop(old_key)
            self.regionprops_names[new_key] = rp_name

        # Keep the intensity key in sync: it gates intensity-image handling
        if old_key == self.intensity_key:
            self.intensity_key = new_key
