import pytest

from funtracks.actions import BasicAction
from funtracks.annotators import GraphAnnotator
from funtracks.data_model import Tracks
from funtracks.features import Time

track_attrs = {"time_attr": "t", "tracklet_attr": "track_id"}


def test_base_graph_annotator(graph_2d_with_segmentation):
    tracks = Tracks(graph_2d_with_segmentation, **track_attrs)
    ann = GraphAnnotator(tracks, {})
    assert len(ann.features) == 0

    feat = Time()
    ann = GraphAnnotator(tracks, {"time": feat})
    # Features start disabled by default
    assert len(ann.all_features) == 1
    assert len(ann.features) == 0
    # Enable to test
    ann.activate_features(["time"])
    assert len(ann.features) == 1

    with pytest.raises(NotImplementedError):
        ann.compute()

    with pytest.raises(NotImplementedError):
        action = BasicAction(tracks)
        ann.update(action)
