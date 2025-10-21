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
        features (dict[str, Feature]): A dict mapping keys to features that this
            annotator is capable of computing and updating.

    Attributes:
        all_features (dict[str, tuple[Feature, bool]]): Maps feature keys to
            (feature, is_included) tuples. Tracks both what can be computed and
            what is currently being computed.
    """

    def __init__(self, tracks: Tracks, features: dict[str, Feature]):
        self.tracks = tracks
        # Store (feature, is_included) for each key
        self.all_features: dict[str, tuple[Feature, bool]] = {
            key: (feat, True) for key, feat in features.items()
        }

    def remove_feature(self, key: str) -> None:
        """Stop computing the given feature in the annotation process.

        This only affects whether the annotator computes values for this feature.
        The feature remains in tracks.features (FeatureDict modifications should be
        done through AnnotatorManager).

        Args:
            key (str): The key of the feature to remove. Must be in all_features.
        """
        if key in self.all_features:
            feat, _ = self.all_features[key]
            self.all_features[key] = (feat, False)

    def add_feature(self, key: str) -> None:
        """Start computing the given feature in the annotation process.

        This only affects whether the annotator computes values for this feature.
        The feature should already be in tracks.features (added during initialization).

        Args:
            key (str): The key of the feature to add. Must be in all_features.
        """
        if key in self.all_features:
            feat, _ = self.all_features[key]
            self.all_features[key] = (feat, True)
        else:
            raise ValueError(
                f"Cannot add feature '{key}' - annotator cannot manage this feature."
            )

    @property
    def features(self) -> dict[str, Feature]:
        """The dict of features that this annotator currently manages.

        Filtered from all_features based on inclusion flags.
        """
        return {k: feat for k, (feat, included) in self.all_features.items() if included}

    def add_features_to_set(self) -> None:
        """Add the currently included features to the tracks FeatureDict.

        Usually performed during initial computation.

        Raises:
            KeyError: If any key already exists in the feature dict.
        """
        for key, feature in self.features.items():
            if key in self.tracks.features:
                raise KeyError(f"Key '{key}' already in feature set")
            self.tracks.features[key] = feature

    def compute(self) -> None:
        """Compute a set of features and add them to the tracks.

        This involves both updating the node/edge attributes on the tracks.graph
        and adding the features to the FeatureDict, if necessary. This is distinct
        from `update` to allow more efficient bulk computation of features.

        Raises:
            NotImplementedError: If not implemented in subclass.
        """
        raise NotImplementedError("Must implement compute in the annotator subclass")

    def update(
        self,
        element: int | tuple[int, int],
    ) -> None:
        """Update a set of features for a given node or edge.

        This involves both updating the node or edge attributes on the tracks.graph
        and adding the features to the FeatureDict, if necessary. This is distinct
        from `compute` to allow more efficient computation of features for single
        elements.

        Args:
            element (int | tuple[int, int]): The node or edge to update

        Raises:
            NotImplementedError: If not implemented in subclass.
        """
        raise NotImplementedError("Must implement update in the annotator subclass")
