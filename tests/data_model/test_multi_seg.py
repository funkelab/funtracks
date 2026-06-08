import numpy as np
import pytest

from funtracks.annotators import EdgeAnnotator, RegionpropsAnnotator
from funtracks.data_model import SolutionTracks, Tracks
from funtracks.features import FeatureDict, Position, SegBbox, SegMask, Time
from funtracks.utils.tracksdata_utils import td_mask_to_pixels


def _make_multi_seg_features():
    """Build the FeatureDict entries for dual nuclear/membrane masks."""
    return {
        "t": Time(),
        "pos": Position(axes=["y", "x"]),
        "nuclear_mask": SegMask(ndim=3, bbox_key="nuclear_bbox"),
        "nuclear_bbox": SegBbox(ndim=3),
        "membrane_mask": SegMask(ndim=3, bbox_key="membrane_bbox"),
        "membrane_bbox": SegBbox(ndim=3),
    }


def _make_multi_seg_tracks(trackclass, graph):
    """Create a Tracks/SolutionTracks with dual mask FeatureDict."""
    features_dict = _make_multi_seg_features()
    is_solution = trackclass is SolutionTracks
    features = FeatureDict(
        features=features_dict,
        time_key="t",
        position_key="pos",
        tracklet_key="track_id" if is_solution else None,
        lineage_key="lineage_id" if is_solution else None,
    )
    return trackclass(graph, ndim=3, features=features), features_dict


@pytest.mark.parametrize("trackclass", [Tracks, SolutionTracks])
def test_multi_seg_tracks(trackclass, graph_2d_with_nuclear_and_membrane):
    tracks, features_dict = _make_multi_seg_tracks(
        trackclass, graph_2d_with_nuclear_and_membrane
    )
    assert len(tracks.segmentations) == 2

    # Assert that each mask feature has an associated annotator
    mask_features = [k for k, v in features_dict.items() if v.get("value_type") == "mask"]
    regionprops_annotators = [
        ann for ann in tracks.annotators if isinstance(ann, RegionpropsAnnotator)
    ]
    edge_annotators = [ann for ann in tracks.annotators if isinstance(ann, EdgeAnnotator)]

    assert len(regionprops_annotators) == len(mask_features), (
        f"Expected {len(mask_features)} RegionpropsAnnotator(s) "
        f"for mask features {mask_features}, "
        f"but found {len(regionprops_annotators)}"
    )

    assert len(edge_annotators) == len(mask_features), (
        f"Expected {len(mask_features)} EdgeAnnotator(s) "
        f"for mask features {mask_features}, "
        f"but found {len(edge_annotators)}"
    )


@pytest.mark.parametrize("trackclass", [Tracks, SolutionTracks])
@pytest.mark.parametrize(
    "updated_mask_key,other_mask_prefix",
    [
        ("nuclear_mask", "membrane_"),
        ("membrane_mask", "nuclear_"),
    ],
)
def test_update_seg_isolation(
    trackclass,
    graph_2d_with_nuclear_and_membrane,
    updated_mask_key,
    other_mask_prefix,
):
    """Updating one mask should only change that mask's features,
    leaving the other mask's features untouched."""
    from funtracks.actions import UpdateNodeSeg

    from ..conftest import make_2d_disk_mask

    tracks, _ = _make_multi_seg_tracks(trackclass, graph_2d_with_nuclear_and_membrane)

    # Enable area and IoU features for both masks
    tracks.enable_features(
        [
            "nuclear_area",
            "membrane_area",
            "nuclear_iou",
            "membrane_iou",
        ]
    )

    # Use node 1 and edge 1→3 (overlapping masks → non-zero IoU)
    test_node = 1
    test_edge = (1, 3)

    # Derive key names from the mask being updated
    from funtracks.annotators._graph_annotator import (
        _derive_mask_prefix,
    )

    updated_prefix = _derive_mask_prefix(updated_mask_key)

    init_updated_area = tracks.get_node_attr(test_node, f"{updated_prefix}area")
    init_other_area = tracks.get_node_attr(test_node, f"{other_mask_prefix}area")
    init_updated_iou = tracks.get_edge_attr(test_edge, f"{updated_prefix}iou")
    init_other_iou = tracks.get_edge_attr(test_edge, f"{other_mask_prefix}iou")

    assert init_updated_area is not None
    assert init_other_area is not None
    # Edge 1→3 has overlapping masks → non-zero IoU
    assert init_updated_iou > 0
    assert init_other_iou > 0

    # Create a new, larger mask and apply it
    pos = tracks.get_position(test_node)
    new_mask = make_2d_disk_mask(center=(pos[0], pos[1]), radius=30)
    UpdateNodeSeg(
        tracks,
        node=test_node,
        mask=new_mask,
        mask_key=updated_mask_key,
    )

    # Updated mask's features should have changed
    new_area = tracks.get_node_attr(test_node, f"{updated_prefix}area")
    assert new_area != init_updated_area, (
        f"{updated_prefix}area should change after {updated_mask_key} update"
    )
    new_iou = tracks.get_edge_attr(test_edge, f"{updated_prefix}iou")
    assert new_iou != init_updated_iou, (
        f"{updated_prefix}iou should change after {updated_mask_key} update"
    )

    # Other mask's features should be untouched
    assert (
        tracks.get_node_attr(test_node, f"{other_mask_prefix}area") == init_other_area
    ), f"{other_mask_prefix}area should NOT change after {updated_mask_key} update"
    assert tracks.get_edge_attr(test_edge, f"{other_mask_prefix}iou") == init_other_iou, (
        f"{other_mask_prefix}iou should NOT change after {updated_mask_key} update"
    )


@pytest.mark.parametrize(
    "updated_mask_key,other_mask_prefix",
    [
        ("nuclear_mask", "membrane_"),
        ("membrane_mask", "nuclear_"),
    ],
)
def test_user_update_seg_multi_mask_shrink(
    graph_2d_with_nuclear_and_membrane,
    updated_mask_key,
    other_mask_prefix,
):
    """UserUpdateSegmentation shrinking one mask should only affect
    that mask's computed features."""
    from funtracks.annotators._graph_annotator import (
        _derive_mask_prefix,
    )
    from funtracks.user_actions import UserUpdateSegmentation

    tracks, _ = _make_multi_seg_tracks(SolutionTracks, graph_2d_with_nuclear_and_membrane)
    tracks.enable_features(
        [
            "nuclear_area",
            "membrane_area",
            "nuclear_iou",
            "membrane_iou",
        ]
    )

    updated_prefix = _derive_mask_prefix(updated_mask_key)
    test_node = 3
    test_edge = (1, 3)

    init_updated_area = tracks.get_node_attr(test_node, f"{updated_prefix}area")
    init_other_area = tracks.get_node_attr(test_node, f"{other_mask_prefix}area")
    init_updated_iou = tracks.get_edge_attr(test_edge, f"{updated_prefix}iou")
    init_other_iou = tracks.get_edge_attr(test_edge, f"{other_mask_prefix}iou")

    # Get the mask's pixels and remove all but one
    mask = tracks.graph.nodes[test_node][updated_mask_key]
    time = tracks.get_time(test_node)
    pixels = td_mask_to_pixels(mask, time, ndim=tracks.ndim)
    pixels_to_remove = tuple(p[1:] for p in pixels)

    action = UserUpdateSegmentation(
        tracks,
        new_value=0,
        updated_pixels=[(pixels_to_remove, test_node)],
        current_track_id=1,
        mask_key=updated_mask_key,
    )

    # Updated mask's area should have shrunk
    new_area = tracks.get_node_attr(test_node, f"{updated_prefix}area")
    assert new_area < init_updated_area, (
        f"{updated_prefix}area should decrease after shrinking {updated_mask_key}"
    )

    # Other mask's features should be untouched
    assert (
        tracks.get_node_attr(test_node, f"{other_mask_prefix}area") == init_other_area
    ), f"{other_mask_prefix}area should NOT change"
    assert tracks.get_edge_attr(test_edge, f"{other_mask_prefix}iou") == init_other_iou, (
        f"{other_mask_prefix}iou should NOT change"
    )

    # Undo should restore
    action.inverse()
    assert tracks.get_node_attr(test_node, f"{updated_prefix}area") == init_updated_area
    assert tracks.get_edge_attr(test_edge, f"{updated_prefix}iou") == pytest.approx(
        init_updated_iou, abs=0.01
    )


@pytest.mark.parametrize(
    "updated_mask_key,other_mask_prefix",
    [
        ("nuclear_mask", "membrane_"),
        ("membrane_mask", "nuclear_"),
    ],
)
def test_user_update_seg_multi_mask_grow(
    graph_2d_with_nuclear_and_membrane,
    updated_mask_key,
    other_mask_prefix,
):
    """UserUpdateSegmentation growing one mask should only affect
    that mask's computed features."""
    from funtracks.annotators._graph_annotator import (
        _derive_mask_prefix,
    )
    from funtracks.user_actions import UserUpdateSegmentation

    tracks, _ = _make_multi_seg_tracks(SolutionTracks, graph_2d_with_nuclear_and_membrane)
    tracks.enable_features(
        [
            "nuclear_area",
            "membrane_area",
            "nuclear_iou",
            "membrane_iou",
        ]
    )

    updated_prefix = _derive_mask_prefix(updated_mask_key)
    test_node = 3
    test_edge = (1, 3)

    init_updated_area = tracks.get_node_attr(test_node, f"{updated_prefix}area")
    init_other_area = tracks.get_node_attr(test_node, f"{other_mask_prefix}area")
    init_other_iou = tracks.get_edge_attr(test_edge, f"{other_mask_prefix}iou")

    # Add one new pixel adjacent to the existing mask
    mask = tracks.graph.nodes[test_node][updated_mask_key]
    time = tracks.get_time(test_node)
    pixels = td_mask_to_pixels(mask, time, ndim=tracks.ndim)
    # Take the first pixel's position and shift x by 1
    new_pixel = (
        np.array([pixels[0][0]]),
        np.array([pixels[1][0]]),
        np.array([10]),  # arbitrary x inside the 100x100 frame
    )

    action = UserUpdateSegmentation(
        tracks,
        new_value=test_node,
        updated_pixels=[(new_pixel, 0)],
        current_track_id=1,
        mask_key=updated_mask_key,
    )

    # Updated mask's area should have grown
    new_area = tracks.get_node_attr(test_node, f"{updated_prefix}area")
    assert new_area > init_updated_area, (
        f"{updated_prefix}area should increase after growing {updated_mask_key}"
    )

    # Other mask's features should be untouched
    assert (
        tracks.get_node_attr(test_node, f"{other_mask_prefix}area") == init_other_area
    ), f"{other_mask_prefix}area should NOT change"
    assert tracks.get_edge_attr(test_edge, f"{other_mask_prefix}iou") == init_other_iou, (
        f"{other_mask_prefix}iou should NOT change"
    )

    # Undo should restore
    action.inverse()
    assert tracks.get_node_attr(test_node, f"{updated_prefix}area") == init_updated_area
