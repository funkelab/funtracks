from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from funtracks.data_model import Tracks
    from funtracks.features.feature import Feature


class GraphAnnotator:
    """A base class for adding and updating graph features.

    This class holds a set of features that it is responsible for. The annotator will
    compute these features and add them to the Tracks initially, and update them when
    necessary. The set of features will all be computed and updated together, although
    individual ones can be removed for efficiency.

    Args:
        tracks (Tracks): The tracks to manage features for.
        features (list[Feature]): A list of all features that this annotator is
            capable of computing and updating.

    Attributes:
        all_features: (list[Feature]): The list of features that the annotator knows
            how to manage (the same as passed in to the constructor).
        features (list[Feature]): The list of features that the annotator is currently
            managing. Does not necessarily include all of the features.
    """

    def __init__(self, tracks: Tracks, features: list[Feature]):
        self.tracks = tracks
        self.all_features = features

        # whether to include each of the features or not. Can update later to exclude
        # features to be more efficient.
        self._include: list[bool] = [True] * len(self.all_features)

    def remove_feature(self, feature: Feature, update_set: bool = True) -> None:
        """Stop computing the given feature in the edge annotation process.
        This will not actually remove the feature from the tracks edges. It will just
        remove it from the list of features that this annotator is updating/computing.

        Args:
            feature (Feature): The feature to remove. Must be in _features list.
            update_set (bool, optional): Whether to update the tracks FeatureSet or not.
                Defaults to True. Will error if the feature is not already in the
                FeatureSet and the value is True.
        """
        if feature in self.all_features:
            self._include[self.all_features.index(feature)] = False
        if update_set:
            self.tracks.features._features.remove(feature)

    def add_feature(self, feature: Feature, update_set: bool = True) -> None:
        """Start computing the given feature in the edges annotation process.
        This will not actually add the feature to the tracks. It will just add it to
        the list of features that this annotator is updating/computing.

        Args:
            feature (Feature): The feature to add. Must be in _features list.
            update_set (bool, optional): Whether to update the tracks FeatureSet or not.
                Defaults to True. Will error if the feature is already in the
                FeatureSet and the value is True.
        """
        if feature in self.all_features:
            self._include[self.all_features.index(feature)] = True
        else:
            raise ValueError(
                f"Cannot add {feature} - this annotator does not know how to manage this "
                "feature."
            )
        if update_set:
            self.tracks.features.add_feature(feature)

    @property
    def features(self) -> list[Feature]:
        """The list of features that this annotator currently manages.

        The list of all features filtered by the include flags.
        """
        return [
            feat
            for feat, include in zip(self.all_features, self._include, strict=True)
            if include
        ]

    def add_features_to_set(self) -> None:
        """Add the currently included features to the tracks FeatureSet.

        Usually performed during initial computation.
        """
        for feature in self.features:
            self.tracks.features.add_feature(feature)

    def compute(self) -> None:
        """Compute a set of features and add them to the tracks.

        This involves both updating the node/edge attributes on the tracks.graph
        and adding the features to the FeatureSet, if necessary. This is distinct
        from `update` to allow more efficient bulk computation of features.

        Raises:
            NotImplementedError: If not implemented in subclass and you attempt to call
                it.
        """
        raise NotImplementedError("Must implement compute in the annotator subclass")

    def update(
        self,
        element: int | tuple[int, int],
    ) -> None:
        """Update a set of features for a given node or edge.

        This involves both updating the node or edge attributes on the tracks.graph
        and adding the features to the FeatureSet, if necessary. This is distinct
        from `compute` to allow more efficient computation of features for single
        elements.

        Args:
            element (int | tuple[int, int]): The node or edge to update the features for

        Raises:
            NotImplementedError: If not implemented in subclass and you attempt to call
                it.
        """
        raise NotImplementedError("Must implement update in the annotator subclass")
