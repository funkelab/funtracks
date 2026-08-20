import numpy as np
import pytest
from tracksdata.nodes import Mask

from funtracks.actions import UpdateNodeSeg, UpdateTrackIDs
from funtracks.annotators import RegionpropsAnnotator
from funtracks.data_model import Tracks

track_attrs = {"time_attr": "t", "tracklet_attr": "track_id"}


def _seg_shape(ndim: int) -> tuple[int, ...]:
    """Segmentation shape used by the get_graph fixture."""
    return (5, 100, 100) if ndim == 3 else (5, 100, 100, 100)


def _frame_index_image(ndim: int) -> np.ndarray:
    """Intensity image whose value is the time index everywhere in the frame.

    The mean intensity of any mask at time t is then exactly t, whatever its shape.
    Broadcast rather than allocated, to keep the 3D case cheap.
    """
    shape = _seg_shape(ndim)
    frames = np.arange(shape[0], dtype=np.float32).reshape(
        (-1,) + (1,) * (len(shape) - 1)
    )
    return np.broadcast_to(frames, shape)


class _RecordingImage:
    """Array-like that records the region requested on each read.

    Stands in for a chunked backend (zarr, h5py), where reading more than the
    bounding box would mean fetching and decompressing chunks for nothing.
    """

    def __init__(self, array: np.ndarray):
        self._array = array
        self.shape = array.shape
        self.ndim = array.ndim
        self.reads: list = []

    def __getitem__(self, key):
        self.reads.append(key)
        return self._array[key]


def _x_gradient_image(ndim: int) -> np.ndarray:
    """Intensity image whose value is the x index, so intensity follows mask shape."""
    shape = _seg_shape(ndim)
    return np.broadcast_to(np.arange(shape[-1], dtype=np.float32), shape)


@pytest.mark.parametrize("ndim", [3, 4])
class TestRegionpropsAnnotator:
    def test_init(self, get_graph, ndim):
        graph = get_graph(ndim, with_seg=True)
        tracks = Tracks(
            graph,
            ndim=ndim,
            **track_attrs,
        )
        rp_ann = RegionpropsAnnotator(tracks)
        # Features start disabled by default
        assert len(rp_ann.all_features) == 6
        assert len(rp_ann.features) == 0
        # Enable features
        rp_ann.activate_features(list(rp_ann.all_features.keys()))
        # pos, area, intensity, ellipse_axis_radii, circularity, perimeter
        assert len(rp_ann.features) == 6

    def test_compute_all(self, get_graph, ndim):
        graph = get_graph(ndim, with_seg=True)
        tracks = Tracks(
            graph,
            ndim=ndim,
            **track_attrs,
        )
        rp_ann = RegionpropsAnnotator(tracks)
        tracks.enable_features(list(rp_ann.all_features.keys()))

        for key in rp_ann.all_features:
            assert key in tracks.graph_solution.node_attr_keys()
            for node_id in tracks.graph_solution.node_ids():
                value = tracks.graph_solution.nodes[node_id][key]
                assert value is not None

    def test_update_all(self, get_graph, ndim):
        graph = get_graph(ndim, with_seg=True)
        tracks = Tracks(
            graph,
            ndim=ndim,
            **track_attrs,
        )
        node_id = 3

        # Get the RegionpropsAnnotator from the registry
        rp_ann = next(
            ann for ann in tracks.annotators if isinstance(ann, RegionpropsAnnotator)
        )
        # Enable features through tracks
        tracks.enable_features(list(rp_ann.all_features.keys()))

        node_mask = tracks.get_mask(node_id)
        removal = Mask(node_mask.mask.copy(), node_mask.bbox)
        removal.mask.flat[np.argmax(removal.mask.flat)] = False
        expected_area = 1

        # Use UpdateNodeSeg action to modify segmentation and update features
        UpdateNodeSeg(tracks, node_id, removal, added=False)
        assert tracks.get_node_attr(node_id, "area") == expected_area
        for key in rp_ann.features:
            assert key in tracks.graph_solution.node_attr_keys()

        # segmentation is fully erased and you try to update
        node_id = 1
        mask = tracks.get_mask(node_id)
        with pytest.warns(
            match="Cannot find label 1 in frame .*: updating regionprops values to None"
        ):
            UpdateNodeSeg(tracks, node_id, mask, added=False)
        # all regionprops features should be the defaults, because seg doesn't exist
        for key in rp_ann.features:
            actual = tracks.graph_solution.nodes[node_id][key]
            expected = tracks.graph_solution._node_attr_schemas()[key].default_value
            # Convert to numpy arrays for comparison (handles both scalar and array types)
            actual_np = np.asarray(actual)
            expected_np = np.asarray(expected)
            assert np.array_equal(actual_np, expected_np)

    def test_add_remove_feature(self, get_graph, ndim):
        graph = get_graph(ndim, with_seg=True)
        tracks = Tracks(
            graph,
            ndim=ndim,
            **track_attrs,
        )
        # Get the RegionpropsAnnotator from the registry
        rp_ann = next(
            ann for ann in tracks.annotators if isinstance(ann, RegionpropsAnnotator)
        )
        to_remove_key = "area"
        # area is not auto-enabled, so enable it first before testing disable
        tracks.enable_features([to_remove_key])
        tracks.disable_features([to_remove_key])

        rp_ann.compute()
        assert to_remove_key not in tracks.graph_solution.node_attr_keys()

        # add it back in
        tracks.enable_features([to_remove_key])
        # but remove a different one
        second_remove_key = "ellipse_axis_radii"
        tracks.disable_features([second_remove_key])

        # remove all but one pixel
        node_id = 3
        node_mask = tracks.get_mask(node_id)
        assert node_mask is not None
        removal = Mask(node_mask.mask.copy(), node_mask.bbox)
        removal.mask.flat[np.argmax(removal.mask.flat)] = False
        # Use UpdateNodeSeg action to modify segmentation and update features
        UpdateNodeSeg(tracks, node_id, removal, added=False)
        # the one we added back in is now present
        assert tracks.get_node_attr(node_id, to_remove_key) is not None

    def test_intensity_single_channel(self, get_graph, ndim):
        """One intensity image yields one mean per node."""
        graph = get_graph(ndim, with_seg=True)
        tracks = Tracks(graph, ndim=ndim, **track_attrs)
        tracks.set_intensity_images([_frame_index_image(ndim)], channel_names=["raw"])
        tracks.enable_features(["intensity"])

        assert tracks.features["intensity"]["num_values"] == 1
        assert tracks.features["intensity"]["display_name"] == "Mean intensity (raw)"
        for node_id in tracks.graph_solution.node_ids():
            assert tracks.get_node_attr(node_id, "intensity") == pytest.approx(
                tracks.get_time(node_id)
            )

    def test_intensity_from_constructor(self, get_graph, ndim):
        """Intensity images can be passed straight to Tracks."""
        graph = get_graph(ndim, with_seg=True)
        tracks = Tracks(
            graph, ndim=ndim, intensity_images=[_frame_index_image(ndim)], **track_attrs
        )
        tracks.enable_features(["intensity"])
        for node_id in tracks.graph_solution.node_ids():
            assert tracks.get_node_attr(node_id, "intensity") == pytest.approx(
                tracks.get_time(node_id)
            )

    def test_intensity_multichannel(self, get_graph, ndim):
        """Several intensity images yield one mean per channel."""
        graph = get_graph(ndim, with_seg=True)
        tracks = Tracks(graph, ndim=ndim, **track_attrs)
        raw = _frame_index_image(ndim)
        tracks.set_intensity_images([raw, raw * 10], channel_names=["gfp", "rfp"])
        tracks.enable_features(["intensity"])

        # The channels are held as given — no full-size stacked copy is made
        assert tracks.regionprops_annotator.intensity_images[0] is raw

        feature = tracks.features["intensity"]
        assert feature["num_values"] == 2
        # each column is named after the image it measures
        assert list(feature["value_names"]) == [
            "Mean intensity (gfp)",
            "Mean intensity (rfp)",
        ]
        for node_id in tracks.graph_solution.node_ids():
            time = tracks.get_time(node_id)
            value = list(tracks.get_node_attr(node_id, "intensity"))
            assert value == pytest.approx([time, 10 * time])

    def test_intensity_without_image_warns_and_skips(self, get_graph, ndim):
        """Intensity is advertised even with no image, but computing it warns."""
        graph = get_graph(ndim, with_seg=True)
        tracks = Tracks(graph, ndim=ndim, **track_attrs)
        assert "intensity" in tracks.get_available_features()

        with pytest.warns(UserWarning, match="no intensity image is set"):
            tracks.enable_features(["intensity"])

    def test_set_intensity_images_recomputes(self, get_graph, ndim):
        """Replacing the images (and their channel count) refreshes stored values."""
        graph = get_graph(ndim, with_seg=True)
        tracks = Tracks(graph, ndim=ndim, **track_attrs)
        raw = _frame_index_image(ndim)
        tracks.set_intensity_images([raw])
        tracks.enable_features(["intensity"])

        # Same channel count: values follow the new image
        tracks.set_intensity_images([raw + 100])
        for node_id in tracks.graph_solution.node_ids():
            assert tracks.get_node_attr(node_id, "intensity") == pytest.approx(
                tracks.get_time(node_id) + 100
            )

        # Channel count changes: the feature is re-registered with two values
        tracks.set_intensity_images([raw, raw * 10])
        assert tracks.features["intensity"]["num_values"] == 2
        for node_id in tracks.graph_solution.node_ids():
            time = tracks.get_time(node_id)
            value = list(tracks.get_node_attr(node_id, "intensity"))
            assert value == pytest.approx([time, 10 * time])

    def test_compute_reads_each_frame_once(self, get_graph, ndim):
        """Bulk compute reads one frame per time point, not one per node.

        A read on a lazily loaded image (e.g. a delayed imread per time point) costs a
        whole frame however small the crop, so nodes sharing a frame must share a read.
        """
        graph = get_graph(ndim, with_seg=True)
        tracks = Tracks(graph, ndim=ndim, **track_attrs)
        image = _RecordingImage(_x_gradient_image(ndim))
        tracks.set_intensity_images([image])
        tracks.enable_features(["intensity"])

        node_ids = list(tracks.graph_solution.node_ids())
        times = set(tracks.get_times(node_ids))
        assert len(times) < len(node_ids)  # otherwise the test proves nothing
        assert len(image.reads) == len(times)
        # each read is a whole frame: a bare time index, no per-axis slicing
        assert sorted(image.reads) == sorted(times)

        # and the values are still right
        assert tracks.get_node_attr(4, "intensity") == pytest.approx(1.5)

    def test_single_node_update_reads_only_its_bounding_box(self, get_graph, ndim):
        """Editing one mask reads that box alone, not the frame around it."""
        graph = get_graph(ndim, with_seg=True)
        tracks = Tracks(graph, ndim=ndim, **track_attrs)
        image = _RecordingImage(_x_gradient_image(ndim))
        tracks.set_intensity_images([image])
        tracks.enable_features(["intensity"])

        node_id = 4
        image.reads.clear()
        node_mask = tracks.get_mask(node_id)
        removal = Mask(node_mask.mask.copy(), node_mask.bbox)
        removal.mask[..., 3] = False
        UpdateNodeSeg(tracks, node_id, removal, added=False)

        frame_shape = _seg_shape(ndim)[1:]
        assert len(image.reads) == 1
        read = image.reads[0]
        # one indexing op: a time point followed by one slice per spatial axis
        assert isinstance(read, tuple)
        assert len(read) == 1 + len(frame_shape)
        assert not isinstance(read[0], slice)
        spans = [sl.stop - sl.start for sl in read[1:]]
        assert all(span < extent for span, extent in zip(spans, frame_shape, strict=True))

        assert tracks.get_node_attr(node_id, "intensity") == pytest.approx(3.0)

    def test_renaming_channels_does_not_recompute(self, get_graph, ndim):
        """Same images under new names only refreshes the feature metadata."""
        graph = get_graph(ndim, with_seg=True)
        tracks = Tracks(graph, ndim=ndim, **track_attrs)
        raw = _frame_index_image(ndim)
        tracks.set_intensity_images([raw, raw], channel_names=["a", "b"])
        tracks.enable_features(["intensity"])

        # Poke a sentinel into the column: a recompute would overwrite it
        node_id = next(iter(tracks.graph_solution.node_ids()))
        tracks.graph_solution.update_node_attrs(
            attrs={"intensity": [[-1.0, -1.0]]}, node_ids=[node_id]
        )

        tracks.set_intensity_images([raw, raw], channel_names=["gfp", "rfp"])

        assert list(tracks.features["intensity"]["value_names"]) == [
            "Mean intensity (gfp)",
            "Mean intensity (rfp)",
        ]
        assert list(tracks.get_node_attr(node_id, "intensity")) == [-1.0, -1.0]

        # A different image does recompute
        tracks.set_intensity_images([raw, raw + 1], channel_names=["gfp", "rfp"])
        time = tracks.get_time(node_id)
        assert list(tracks.get_node_attr(node_id, "intensity")) == pytest.approx(
            [time, time + 1]
        )

    def test_intensity_updated_on_seg_edit(self, get_graph, ndim):
        """Editing a mask recomputes its intensity over the new pixels."""
        graph = get_graph(ndim, with_seg=True)
        tracks = Tracks(graph, ndim=ndim, **track_attrs)
        tracks.set_intensity_images([_x_gradient_image(ndim)])
        tracks.enable_features(["intensity"])

        # Node 4 is a square/cube at the origin with width 4, so mean x is 1.5
        node_id = 4
        assert tracks.get_node_attr(node_id, "intensity") == pytest.approx(1.5)

        # Subtract everything but the x == 3 slab, so only intensity 3.0 is left
        node_mask = tracks.get_mask(node_id)
        removal = Mask(node_mask.mask.copy(), node_mask.bbox)
        removal.mask[..., 3] = False
        UpdateNodeSeg(tracks, node_id, removal, added=False)

        assert tracks.get_node_attr(node_id, "intensity") == pytest.approx(3.0)

    def test_intensity_image_shape_mismatch(self, get_graph, ndim):
        graph = get_graph(ndim, with_seg=True)
        tracks = Tracks(graph, ndim=ndim, **track_attrs)
        with pytest.raises(ValueError, match="does not match the segmentation shape"):
            tracks.set_intensity_images([np.zeros((5, 10, 10), dtype=np.float32)])

    def test_bare_array_is_rejected(self, get_graph, ndim):
        """One image per channel: a bare array is a mistake worth naming."""
        graph = get_graph(ndim, with_seg=True)
        tracks = Tracks(graph, ndim=ndim, **track_attrs)
        with pytest.raises(TypeError, match=r"pass \[image\], not a bare array"):
            tracks.set_intensity_images(_frame_index_image(ndim))

    def test_empty_intensity_images_clears(self, get_graph, ndim):
        """An empty list means the same as None."""
        graph = get_graph(ndim, with_seg=True)
        tracks = Tracks(graph, ndim=ndim, **track_attrs)
        tracks.set_intensity_images([_frame_index_image(ndim)])
        tracks.enable_features(["intensity"])

        tracks.disable_features(["intensity"])
        tracks.set_intensity_images([])
        assert tracks.regionprops_annotator.intensity_images is None

    def test_intensity_channel_name_mismatch(self, get_graph, ndim):
        graph = get_graph(ndim, with_seg=True)
        tracks = Tracks(graph, ndim=ndim, **track_attrs)
        raw = _frame_index_image(ndim)
        with pytest.raises(ValueError, match="channel names"):
            tracks.set_intensity_images([raw, raw], channel_names=["only_one"])

    def test_set_intensity_images_without_seg(self, get_graph, ndim):
        graph = get_graph(ndim, with_seg=False)
        tracks = Tracks(graph, ndim=ndim, **track_attrs)
        assert tracks.regionprops_annotator is None
        with pytest.raises(ValueError, match="no segmentation"):
            tracks.set_intensity_images([_frame_index_image(ndim)])

    def test_missing_seg(self, get_graph, ndim):
        """Test that RegionpropsAnnotator gracefully handles missing segmentation."""
        graph = get_graph(ndim, with_seg=False)
        tracks = Tracks(graph, ndim=ndim, **track_attrs)
        rp_ann = RegionpropsAnnotator(tracks)
        assert len(rp_ann.features) == 0
        # Should not raise an error, just return silently
        rp_ann.compute()  # No error expected

    def test_centroid_world_coords_with_scale(self, get_graph, ndim):
        """Centroid in 'pos' must be pixel_centroid * scale (world units).

        Without the fix, skimage returns local_centroid * spacing + bbox_min_pixel
        (mixed units). The correct formula is (local_centroid + bbox_min_pixel) *
        spacing = pixel_centroid * spacing.

        Node 6 has a cube/square at corner (96, 96, ...) with width 4,
        so pixel centroid = 97.5 in each spatial axis.
        """
        graph = get_graph(ndim, with_seg=True)
        if ndim == 3:
            scale = [1.0, 2.0, 3.0]
            pixel_centroid = np.array([97.5, 97.5])
        else:
            scale = [1.0, 2.0, 3.0, 4.0]
            pixel_centroid = np.array([97.5, 97.5, 97.5])

        tracks = Tracks(graph, ndim=ndim, scale=scale, **track_attrs)
        # Force recomputation so regionprops runs with the given scale as spacing
        tracks.enable_features(["pos"])

        pos = np.array(tracks.graph_solution.nodes[6]["pos"])
        expected = pixel_centroid * np.array(scale[1:])

        bug_value = np.array([1.5] * len(pixel_centroid)) * np.array(
            scale[1:]
        ) + np.array([96.0] * len(pixel_centroid))
        np.testing.assert_allclose(
            pos,
            expected,
            atol=0.1,
            err_msg=(
                f"World centroid must be pixel_centroid * scale. "
                f"Got {pos}, expected {expected}. "
                f"Bug value would be local_centroid * scale + bbox_min = {bug_value}"
            ),
        )

    def test_ignores_irrelevant_actions(self, get_graph, ndim):
        """Test that RegionpropsAnnotator ignores actions that don't affect
        segmentation.
        """
        graph = get_graph(ndim, prefill_track_ids=True, with_seg=True)
        tracks = Tracks(
            graph,
            ndim=ndim,
            **track_attrs,
        )
        tracks.enable_features(["area", "track_id"])

        node_id = 1
        initial_area = tracks.get_node_attr(node_id, "area")

        # Make the stored area stale by writing a fake value directly to the graph,
        # bypassing the action system. If UpdateTrackIDs incorrectly triggers
        # RegionpropsAnnotator, it would recompute area back to initial_area and
        # the assertion below would fail.
        fake_area = initial_area + 999
        tracks.graph_solution.update_node_attrs(
            attrs={"area": [fake_area]}, node_ids=[node_id]
        )

        original_track_id = tracks.get_track_id(node_id)
        new_track_id = original_track_id + 100

        UpdateTrackIDs(tracks, node_id, new_track_id)

        assert tracks.get_node_attr(node_id, "area") == fake_area
        assert tracks.get_track_id(node_id) == new_track_id
