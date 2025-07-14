import pytest

from funtracks.data_model import Tracks
from funtracks.features import GraphAnnotator, Time


def test_base_graph_annotator(graph_2d, segmentation_2d):
    tracks = Tracks(graph_2d, segmentation=segmentation_2d)
    ann = GraphAnnotator(tracks, [])
    assert len(ann.features) == 0

    feat = Time()
    ann = GraphAnnotator(tracks, [feat])
    assert len(ann.features) == 1

    with pytest.raises(NotImplementedError):
        ann.compute()

    with pytest.raises(NotImplementedError):
        ann.update(1)
