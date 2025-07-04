from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._base import Feature, FeatureType
from .edge_features import Distance, EdgeSelected, EdgeSelectionPin, FrameSpan, IoU
from .node_features import (
    ComputedPosition,
    NodeSelected,
    NodeSelectionPin,
    Position,
    Time,
    TrackID,
)

if TYPE_CHECKING:
    from ..actions._base import TracksAction


# def extra_features(ndim: int, seg: bool) -> list[Feature]:
#     feats = []
#     if seg:
#         feats.extend([Area(ndim=ndim), IoU()])
#     return feats


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

        self._features: list[Feature] = [
            self.time,
            self.position,
            self.track_id,
            self.distance,
            self.frame_span,
            self.node_selected,
            self.edge_selected,
            self.node_selection_pin,
            self.edge_selection_pin,
        ]

        # for feat in extra_features(ndim, seg):
        #     self.add_feature(feat)

    @property
    def node_features(self):
        return [f for f in self._features if f.feature_type == FeatureType.NODE]

    @property
    def edge_features(self):
        return [f for f in self._features if f.feature_type == FeatureType.EDGE]

    def get_features_to_compute(self, action: TracksAction):
        return [f for f in self._features if f.computed]  # and
        # any(map(partial(isinstance, action), f.compute_on))]

    def add_feature(self, feature: Feature):
        if feature.feature_type == FeatureType.NODE:
            existing_features = self.node_features
        elif feature.feature_type == FeatureType.EDGE:
            existing_features = self.edge_features

        if feature.attr_name in [f.attr_name for f in existing_features]:
            return
            # raise KeyError(f"Name {feature.attr_name} already in feature set")
        self._features.append(feature)

    def validate_new_node_features(self, features: dict[Feature, Any]):
        assert set(features).issubset(set(self.node_features)), (
            f"Feature in {features} not in {self.node_features}"
        )
        for feature in self.node_features:
            if not feature.computed and feature.required:
                assert feature in features, f"Required feature {feature} not provided"

    def validate_new_edge_features(self, features: dict[Feature, Any]):
        assert set(features).issubset(set(self.edge_features)), (
            f"Feature in {features} not in {self.edge_features}"
        )
        for feature in self.edge_features:
            if not feature.computed and feature.required:
                assert feature in features, f"Required feature {feature} not provided"

    def dump_json(self) -> dict:
        return {"features": [feat.model_dump(mode="json") for feat in self._features]}

    @classmethod
    def from_json(cls, json_dict) -> FeatureSet:
        features_list = [Feature(**feat) for feat in json_dict["features"]]
        (
            time,
            position,
            track_id,
            distance,
            frame_span,
            node_selected,
            edge_selected,
            node_selection_pin,
            edge_selection_pin,
        ) = features_list[0:9]
        features = FeatureSet.__new__(FeatureSet)
        features.time = time
        features.position = position
        features.track_id = track_id
        features.distance = distance
        features.frame_span = frame_span
        features.node_selected = node_selected
        features.edge_selected = edge_selected
        features.node_selection_pin = node_selection_pin
        features.edge_selection_pin = edge_selection_pin
        features._features = features_list
        return features
