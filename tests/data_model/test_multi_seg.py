import pytest

from funtracks.data_model import SolutionTracks, Tracks
from funtracks.features import FeatureDict, Position, SegBbox, SegMask, Time


@pytest.mark.parametrize("trackclass", [Tracks, SolutionTracks])
def test_multi_seg_tracks(trackclass, graph_2d_with_nuclear_and_membrane):
    features_dict = {
        "t": Time(),
        "pos": Position(axes=["y", "x"]),
        "nuclear_mask": SegMask(ndim=3, bbox_key="nuclear_bbox"),
        "nuclear_bbox": SegBbox(ndim=3),
        "membrane_mask": SegMask(ndim=3, bbox_key="membrane_bbox"),
        "membrane_bbox": SegBbox(ndim=3),
    }

    features = FeatureDict(
        features=features_dict,
        time_key="t",
        position_key="pos",
        tracklet_key=None,
        lineage_key=None,
    )

    tracks = trackclass(graph_2d_with_nuclear_and_membrane, ndim=3, features=features)
    assert len(tracks.segmentations) == 2
