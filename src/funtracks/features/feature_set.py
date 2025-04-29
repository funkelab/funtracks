from typing import Any

from ._base import Feature, FeatureType
from .edge_features import EdgeSelected, EdgeSelectionPin, IoU
from .node_features import Area, NodeSelected, NodeSelectionPin, Position, Time, TrackID


def extra_features(ndim: int, seg: bool) -> list[Feature]:
    feats: list[Feature] = []
    if seg:
        feats.extend([Area(ndim=ndim), IoU()])
    return feats


class FeatureSet:
    def __init__(
        self,
        ndim: int,
        seg: bool,
        pos_attr: str | None = None,
        time_attr: str | None = None,
    ):
        self.time = Time(time_attr=time_attr)
        axes = ("z", "y", "x") if ndim == 4 else ("y", "x")
        self.position = Position(axes=axes, attr_name=pos_attr)
        self.track_id = TrackID()
        self.node_selected = NodeSelected()
        self.edge_selected = EdgeSelected()
        self.node_pin = NodeSelectionPin()
        self.edge_pin = EdgeSelectionPin()

        self.extra_features = extra_features(ndim, seg)
        self._features = [
            self.time,
            self.position,
            self.track_id,
            self.node_selected,
            self.edge_selected,
            self.node_pin,
            self.edge_pin,
        ]
        self._features.extend(self.extra_features)

    def __getitem__(self, i) -> Feature:
        return self._features[i]

    def validate_new_node_features(self, features: dict[Feature, Any]):
        assert set(features).issubset(set(self._features))
        for feature in self._features:
            if feature.feature_type == FeatureType.NODE and not feature.computed:
                assert feature in features
