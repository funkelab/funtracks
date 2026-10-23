import numpy as np
import pytest

from funtracks.annotators import PointIntensityAnnotator
from funtracks.annotators._point_intensity_annotator import _ball_mask
from funtracks.data_model import Tracks
from funtracks.user_actions import UserAddNode, UserUpdateNodeAttrs

track_attrs = {"time_attr": "t", "tracklet_attr": "track_id"}


def _image_shape(ndim: int) -> tuple[int, ...]:
    """Image shape matching the extent of the get_graph fixture positions."""
    return (5, 100, 100) if ndim == 3 else (5, 100, 100, 100)


def _frame_index_image(ndim: int) -> np.ndarray:
    """Intensity image whose value is the time index everywhere in the frame."""
    shape = _image_shape(ndim)
    frames = np.arange(shape[0], dtype=np.float32).reshape(
        (-1,) + (1,) * (len(shape) - 1)
    )
    return np.broadcast_to(frames, shape)


def _x_gradient_image(ndim: int) -> np.ndarray:
    """Intensity image whose value is the x index, so a centered ball averages to x."""
    shape = _image_shape(ndim)
    return np.broadcast_to(np.arange(shape[-1], dtype=np.float32), shape)


class TestBallMask:
    def test_disk_pixels(self):
        # Pixel centers within 2.5 of (50, 50): 1 + 4 + 4 + 4 + 8
        mask = _ball_mask([50, 50], 5, np.ones(2), (100, 100))
        assert mask.mask.sum() == 21
        assert list(mask.bbox) == [48, 48, 53, 53]

    def test_sphere_pixels(self):
        mask = _ball_mask([50, 50, 50], 3, np.ones(3), (100, 100, 100))
        # Radius 1.5: center, 6 face neighbours and 12 edge neighbours
        assert mask.mask.sum() == 19

    def test_anisotropic_spacing(self):
        """The diameter is in world units, so coarser axes span fewer pixels."""
        mask = _ball_mask([50, 50], 9, np.array([4.0, 1.0]), (100, 100))
        rows, cols = np.nonzero(mask.mask)
        assert np.ptp(rows) + 1 == 2  # rows 12 and 13 in 12.5 +- 1.125
        assert np.ptp(cols) + 1 == 9  # columns 46 to 54 in 50 +- 4.5

    def test_clipped_to_frame(self):
        mask = _ball_mask([0, 0], 5, np.ones(2), (100, 100))
        assert list(mask.bbox[:2]) == [0, 0]
        assert mask.mask.sum() == 8  # the quarter disk inside the frame

    def test_tiny_diameter_uses_nearest_pixel(self):
        mask = _ball_mask([10.4, 20.6], 0.1, np.ones(2), (100, 100))
        assert mask.mask.sum() == 1
        assert list(mask.bbox) == [10, 21, 11, 22]

    def test_outside_frame(self):
        assert _ball_mask([-10, 50], 5, np.ones(2), (100, 100)) is None
        assert _ball_mask([50, 200], 5, np.ones(2), (100, 100)) is None


@pytest.mark.parametrize("ndim", [3, 4])
class TestPointIntensityAnnotator:
    def test_registered_without_seg(self, get_graph, ndim):
        tracks = Tracks(get_graph(ndim), ndim=ndim, **track_attrs)
        assert tracks.regionprops_annotator is None
        assert isinstance(tracks.point_intensity_annotator, PointIntensityAnnotator)
        assert tracks.intensity_annotator is tracks.point_intensity_annotator
        assert "intensity" in tracks.get_available_features()

    def test_not_registered_with_seg(self, get_graph, ndim):
        tracks = Tracks(get_graph(ndim, with_seg=True), ndim=ndim, **track_attrs)
        assert tracks.point_intensity_annotator is None
        assert tracks.intensity_diameter is None
        with pytest.raises(ValueError, match="has a segmentation"):
            tracks.set_intensity_diameter(3)

    def test_compute(self, get_graph, ndim):
        """Each node measures in its own frame."""
        tracks = Tracks(get_graph(ndim), ndim=ndim, **track_attrs)
        tracks.set_intensity_images([_frame_index_image(ndim)])
        tracks.enable_features(["intensity"])

        for node in tracks.graph_full.node_ids():
            assert tracks.get_node_attr(node, "intensity") == pytest.approx(
                tracks.get_time(node)
            )

    def test_centered_on_position(self, get_graph, ndim):
        tracks = Tracks(get_graph(ndim), ndim=ndim, **track_attrs)
        tracks.set_intensity_images([_x_gradient_image(ndim)])
        tracks.set_intensity_diameter(7)
        tracks.enable_features(["intensity"])
        # Node 1 sits at x=50, far from the frame edge, so the ball is symmetric
        assert tracks.get_node_attr(1, "intensity") == pytest.approx(50)

    def test_scale(self, get_graph, ndim):
        """Positions are in world units, so they are divided by the scale to find the
        pixels to measure."""
        scale = [1.0] + [2.0] * (ndim - 1)
        tracks = Tracks(get_graph(ndim), ndim=ndim, scale=scale, **track_attrs)
        tracks.set_intensity_images([_x_gradient_image(ndim)])
        tracks.enable_features(["intensity"])
        # Node 1 at x=50 (world) is pixel 25
        assert tracks.get_node_attr(1, "intensity") == pytest.approx(25)

    def test_multichannel(self, get_graph, ndim):
        tracks = Tracks(get_graph(ndim), ndim=ndim, **track_attrs)
        tracks.set_intensity_images(
            [_frame_index_image(ndim), _x_gradient_image(ndim)],
            channel_names=["time", "x"],
        )
        tracks.enable_features(["intensity"])
        assert tracks.features["intensity"]["num_values"] == 2
        assert list(tracks.get_node_attr(1, "intensity")) == pytest.approx([0, 50])

    def test_set_diameter_recomputes(self, get_graph, ndim):
        tracks = Tracks(get_graph(ndim), ndim=ndim, **track_attrs)
        tracks.set_intensity_images([_x_gradient_image(ndim)])
        tracks.set_intensity_diameter(1)
        tracks.enable_features(["intensity"])
        # Node 4 sits at x=1.5, so a larger ball is clipped by the frame edge and
        # leans towards higher x
        small = tracks.get_node_attr(4, "intensity")
        tracks.set_intensity_diameter(9)
        assert tracks.intensity_diameter == 9
        assert tracks.get_node_attr(4, "intensity") > small

    def test_invalid_diameter(self, get_graph, ndim):
        tracks = Tracks(get_graph(ndim), ndim=ndim, **track_attrs)
        for diameter in [0, -1, np.nan]:
            with pytest.raises(ValueError, match="positive"):
                tracks.set_intensity_diameter(diameter)

    def test_update_on_add_and_move(self, get_graph, ndim):
        tracks = Tracks(get_graph(ndim), ndim=ndim, **track_attrs)
        tracks.set_intensity_images([_x_gradient_image(ndim)])
        tracks.enable_features(["intensity"])

        pos = [50.0] * (ndim - 3) + [50.0, 30.0]
        UserAddNode(tracks, node=10, attributes={"pos": pos, "t": 3, "track_id": 10})
        assert tracks.get_node_attr(10, "intensity") == pytest.approx(30)

        new_pos = [50.0] * (ndim - 3) + [50.0, 70.0]
        UserUpdateNodeAttrs(tracks, node=10, attrs={"pos": new_pos})
        assert tracks.get_node_attr(10, "intensity") == pytest.approx(70)

    def test_wrong_image_ndim(self, get_graph, ndim):
        tracks = Tracks(get_graph(ndim), ndim=ndim, **track_attrs)
        with pytest.raises(ValueError, match="dimensions of the tracks"):
            tracks.set_intensity_images([np.zeros((5,) * (ndim + 1))])

    def test_mismatched_channel_shapes(self, get_graph, ndim):
        tracks = Tracks(get_graph(ndim), ndim=ndim, **track_attrs)
        with pytest.raises(ValueError, match="different shapes"):
            tracks.set_intensity_images([np.zeros((5,) * ndim), np.zeros((6,) * ndim)])

    def test_no_image_warns(self, get_graph, ndim):
        tracks = Tracks(get_graph(ndim), ndim=ndim, **track_attrs)
        with pytest.warns(UserWarning, match="no intensity image"):
            tracks.enable_features(["intensity"])
