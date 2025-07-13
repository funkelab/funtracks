import pytest

from funtracks.data_model import Tracks
from funtracks.features import GraphAnnotator, Time


def test_base_graph_annotator(graph_2d, segmentation_2d):
    ann = GraphAnnotator([])
    assert len(ann.features) == 0

    feat = Time()
    ann = GraphAnnotator([feat])
    assert len(ann.features) == 1

    tracks = Tracks(graph_2d, segmentation=segmentation_2d)
    with pytest.raises(NotImplementedError):
        ann.compute(tracks, feat)

    with pytest.raises(NotImplementedError):
        ann.update(tracks, feat, 1)
