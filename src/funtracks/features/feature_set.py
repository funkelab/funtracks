from typing import Any

from ._base import Feature, FeatureType
from .edge_features import Distance, EdgeSelected, EdgeSelectionPin, FrameSpan, IoU
from .node_features import (
    Area,
    ComputedPosition,
    NodeSelected,
    NodeSelectionPin,
    Position,
    Time,
    TrackID,
)


def extra_features(ndim: int, seg: bool) -> dict[str, Feature]:
    feats = {}
    if seg:
        feats["area"] = Area(ndim=ndim)
        feats["iou"] = IoU()
    return feats


class FeatureSet:
    """Mapping from unique strings to Feature metadata
    Ideally we have keyword access to semantically meaningful tracking features
    and string based access to all feature metadata
    Want to be able to:
    - list all available (node/edge) features
    - access the string for semantically meaningful features as attributes of something
    """

    def __init__(
        self,
        ndim: int,
        seg: bool,
        pos_attr: str | None = None,
        time_attr: str | None = None,
    ):
        self.node_features = {}
        self.edge_features = {}

        self.time = Time(attr_name=time_attr)
        axes = ("z", "y", "x") if ndim == 4 else ("y", "x")
        self.position = (
            Position(axes=axes, attr_name=pos_attr)
            if not seg
            else ComputedPosition(axes=axes, attr_name=pos_attr)
        )
        self.track_id = TrackID()
        self.distance = Distance()
        self.frame_span = FrameSpan()
        self.node_selected = NodeSelected()
        self.edge_selected = EdgeSelected()
        self.node_selection_pin = NodeSelectionPin()
        self.edge_selection_pin = EdgeSelectionPin()

        for name, feat in extra_features(ndim, seg).items():
            self.add_feature(name, feat)

    def add_feature(self, name, feature):
        if name in self.__dict__:
            raise KeyError(f"Name {name} already in feature set")
        self.__setattr__(name, feature)

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()

    def validate_new_features(self, features: dict[str, Any], feature_type: FeatureType):
        assert set(features).issubset(set(self.keys()))
        for name, feature in self.items():
            if feature.feature_type == feature_type and not feature.computed:
                assert name in features
