from __future__ import annotations

from typing import TYPE_CHECKING, Any

import tracksdata as td

from ._base import BasicAction

if TYPE_CHECKING:
    from funtracks.data_model import Tracks
    from funtracks.features import Feature

# Column values: the node ids and their values, in the same order. Stored densely
# (every node). If undo memory becomes a problem on large graphs, store only the
# values that differ from the feature's default.
FeatureValues = tuple[list[int], list[Any]]


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
            values (tuple[list[int], list] | None): Optional node ids and their
                values. Used by the inverse of DeleteFeature to restore a column.

        Raises:
            NotImplementedError: If the feature is not a node feature.
            ValueError: If the key is already a feature, or is owned by an annotator.
        """
        super().__init__(tracks)
        if feature["feature_type"] != "node":
            raise NotImplementedError(f"Only node features are supported, not {key!r}")
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
        # Copy: add_feature stores the dict by reference
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
                attrs={self.key: values}, node_ids=node_ids
            )
        self.tracks.notify_annotators(self)


class DeleteFeature(BasicAction):
    """Action for deleting a user-created node feature (an attribute column).

    Saves the column's values for every node in graph_full (soft-deleted candidates
    included), so the action can be inverted.
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
        # Track ids are always annotator-owned, so caught by the next check
        pos_key = tracks.features.position_key
        core_keys = {
            tracks.features.time_key,
            td.DEFAULT_ATTR_KEYS.MASK,
            td.DEFAULT_ATTR_KEYS.BBOX,
            "solution",
        }
        core_keys.update(pos_key if isinstance(pos_key, list) else [pos_key])
        if key in core_keys:
            raise ValueError(f"Cannot delete core feature {key!r}")
        if key in tracks.annotators.all_features:
            raise ValueError(
                f"Feature {key!r} is computed by an annotator; use disable_features"
            )
        feature = tracks.features[key]
        if feature["feature_type"] != "node":
            raise NotImplementedError(f"Only node features are supported, not {key!r}")
        if feature.get("derived_features"):
            raise ValueError(f"Cannot delete feature {key!r}, it has derived features")
        self.key = key
        self.feature = feature
        node_id = td.DEFAULT_ATTR_KEYS.NODE_ID
        df = tracks.graph_full.node_attrs(attr_keys=[node_id, key])
        self.values = (df[node_id].to_list(), df[key].to_list())
        self._apply()

    def inverse(self) -> BasicAction:
        """Invert the action to re-add the feature with its saved values."""
        return AddFeature(self.tracks, self.key, self.feature, values=self.values)

    def _apply(self) -> None:
        self.tracks.delete_feature(self.key)
        self.tracks.notify_annotators(self)
