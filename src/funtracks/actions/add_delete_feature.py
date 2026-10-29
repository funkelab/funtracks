from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
import tracksdata as td

from ._base import BasicAction

if TYPE_CHECKING:
    from funtracks.data_model import Tracks
    from funtracks.features import Feature

# Sparse column values: the node ids whose value differs from the feature's default,
# and their values, in the same order.
FeatureValues = tuple[np.ndarray, np.ndarray]


def _core_keys(tracks: Tracks) -> set[str]:
    """Keys that the Tracks itself depends on, which no feature action may touch."""
    features = tracks.features
    keys = {
        features.time_key,
        td.DEFAULT_ATTR_KEYS.MASK,
        td.DEFAULT_ATTR_KEYS.BBOX,
        "solution",
    }
    position_key = features.position_key
    if isinstance(position_key, list):
        keys.update(position_key)
    elif position_key is not None:
        keys.add(position_key)
    for key in (features.tracklet_key, features.lineage_key):
        if key is not None:
            keys.add(key)
    return keys


def _check_node_feature(key: str, feature: Feature) -> None:
    if feature["feature_type"] != "node":
        raise NotImplementedError(
            f"Feature actions only support node features, but {key!r} has "
            f"feature_type {feature['feature_type']!r}"
        )


def _differs(value: Any, default: Any) -> bool:
    if value is None or default is None:
        return value is not default
    if isinstance(value, np.ndarray | list | tuple) or isinstance(
        default, np.ndarray | list | tuple
    ):
        return not np.array_equal(np.asarray(value), np.asarray(default))
    return bool(value != default)


def _read_sparse_values(tracks: Tracks, key: str, default: Any) -> FeatureValues:
    """Read the values of a node column that differ from its default.

    Reads graph_full, so soft-deleted candidates keep their value too.
    """
    df = tracks.graph_full.node_attrs(attr_keys=[td.DEFAULT_ATTR_KEYS.NODE_ID, key])
    if df[key].dtype.is_numeric() or df[key].dtype == pl.Boolean:
        # Fast path for scalar columns. ne_missing keeps nulls when the default is
        # not null (and drops them when it is).
        differs = (
            pl.col(key).is_not_null()
            if default is None
            else pl.col(key).ne_missing(default)
        )
        df = df.filter(differs)
        node_ids = df[td.DEFAULT_ATTR_KEYS.NODE_ID].to_numpy().astype(np.int64)
        return node_ids, df[key].to_numpy()

    node_ids_list = []
    values_list = []
    for node_id, value in zip(
        df[td.DEFAULT_ATTR_KEYS.NODE_ID].to_list(), df[key].to_list(), strict=True
    ):
        if _differs(value, default):
            node_ids_list.append(node_id)
            values_list.append(value)
    values = np.empty(len(values_list), dtype=object)
    values[:] = values_list
    return np.asarray(node_ids_list, dtype=np.int64), values


class AddFeature(BasicAction):
    """Action for adding a new node feature (an attribute column) to the Tracks.

    Meant for user-created features such as groups. Features computed by an
    annotator are switched on with ``Tracks.enable_features`` instead, which is not
    an action and is not undoable.
    """

    def __init__(
        self,
        tracks: Tracks,
        key: str,
        feature: Feature,
        values: FeatureValues | None = None,
    ):
        """Create an action to add a new node feature.

        Args:
            tracks (Tracks): The Tracks to add the feature to.
            key (str): The key of the new feature.
            feature (Feature): The feature metadata. Only node features are supported.
            values (tuple[np.ndarray, np.ndarray] | None): Optional node ids and the
                values to set for them. Nodes not listed get the feature's default
                value. Used by the inverse of DeleteFeature to restore a column.

        Raises:
            NotImplementedError: If the feature is not a node feature.
            ValueError: If the key is already a feature, or is owned by an annotator.
        """
        super().__init__(tracks)
        _check_node_feature(key, feature)
        if key in tracks.features:
            raise ValueError(f"Feature {key!r} already exists")
        if key in tracks.annotators.all_features:
            raise ValueError(
                f"Feature {key!r} is computed by an annotator; use enable_features"
            )
        # A column that exists without being a registered feature (e.g. loaded from
        # a geff) would keep its values here, and the inverse would then drop them.
        if key in tracks.graph_full.node_attr_keys():
            raise ValueError(f"Node attribute {key!r} already exists in the graph")
        self.key = key
        # Copy, so that a later change to the caller's dict (which add_feature
        # stores as is) does not change the feature this action restores.
        self.feature = feature.copy()
        self.values = values
        self._apply()

    def inverse(self) -> BasicAction:
        """Invert the action to delete the feature again."""
        return DeleteFeature(self.tracks, self.key)

    def _apply(self) -> None:
        self.tracks.add_feature(self.key, self.feature)
        if self.values is not None and len(self.values[0]) > 0:
            node_ids, values = self.values
            self.tracks.graph_full.update_node_attrs(
                attrs={self.key: list(values)},
                node_ids=[int(node) for node in node_ids],
            )
        self.tracks.notify_annotators(self)


class DeleteFeature(BasicAction):
    """Action for deleting a user-created node feature (an attribute column).

    Saves the values that differ from the feature's default, for every node in
    graph_full (soft-deleted candidates included), so the action can be inverted.
    """

    def __init__(self, tracks: Tracks, key: str):
        """Create an action to delete a node feature.

        Args:
            tracks (Tracks): The Tracks to delete the feature from.
            key (str): The key of the feature to delete.

        Raises:
            NotImplementedError: If the feature is not a node feature.
            ValueError: If the key is not a feature, is a core feature of the Tracks,
                is owned by an annotator, or has derived features.
        """
        super().__init__(tracks)
        if key not in tracks.features:
            raise ValueError(f"Feature {key!r} does not exist")
        if key in _core_keys(tracks):
            raise ValueError(f"Cannot delete core feature {key!r}")
        if key in tracks.annotators.all_features:
            raise ValueError(
                f"Feature {key!r} is computed by an annotator; use disable_features"
            )
        feature = tracks.features[key]
        _check_node_feature(key, feature)
        if feature.get("derived_features"):
            raise ValueError(f"Cannot delete feature {key!r}, it has derived features")
        self.key = key
        self.feature = feature
        self.values = _read_sparse_values(tracks, key, feature.get("default_value"))
        self._apply()

    def inverse(self) -> BasicAction:
        """Invert the action to re-add the feature with its saved values."""
        return AddFeature(self.tracks, self.key, self.feature, values=self.values)

    def _apply(self) -> None:
        self.tracks.delete_feature(self.key)
        self.tracks.notify_annotators(self)
