from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from funtracks.actions import BasicAction
    from funtracks.data_model import Tracks
    from funtracks.features.feature import Feature

logger = logging.getLogger(__name__)


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
            what is currently being computed. Defaults to computing nothing.
    """

    @classmethod
    def can_annotate(cls, tracks: Tracks) -> bool:
        """Check if this annotator can annotate the given tracks.

        Subclasses should override this method to specify their requirements
        (e.g., segmentation, SolutionTracks, etc.).

        Args:
            tracks: The tracks to check compatibility with

        Returns:
            True if the annotator can annotate these tracks, False otherwise
        """
        return True

    def __init__(self, tracks: Tracks, features: dict[str, Feature]):
        self.tracks = tracks
        # Store (feature, is_included) for each key - default to False (disabled)
        self.all_features: dict[str, tuple[Feature, bool]] = {
            key: (feat, False) for key, feat in features.items()
        }

    def activate_features(self, keys: list[str]) -> None:
        """Activate computation of the given features in the annotation process.

        Filters the list to only features this annotator owns, ignoring others.

        Args:
            keys: List of feature keys to activate. Only keys in all_features are
                activated.
        """
        for key in keys:
            if key in self.all_features:
                feat, _ = self.all_features[key]
                self.all_features[key] = (feat, True)

    def deactivate_features(self, keys: list[str]) -> None:
        """Deactivate computation of the given features in the annotation process.

        Filters the list to only features this annotator owns, ignoring others.

        Args:
            keys: List of feature keys to deactivate. Only keys in all_features
                are deactivated.
        """
        for key in keys:
            if key in self.all_features:
                feat, _ = self.all_features[key]
                self.all_features[key] = (feat, False)

    @property
    def features(self) -> dict[str, Feature]:
        """The dict of features that this annotator currently manages.

        Filtered from all_features based on inclusion flags.
        """
        return {k: feat for k, (feat, included) in self.all_features.items() if included}

    def _filter_feature_keys(self, feature_keys: list[str] | None) -> list[str]:
        """Filter feature keys to only those that are active.

        Args:
            feature_keys: Optional list of feature keys to filter. If None, returns
                all currently active features.

        Returns:
            List of feature keys that are active.
        """
        if feature_keys is None:
            return list(self.features.keys())

        return [k for k in feature_keys if k in self.features]

    def compute(self, feature_keys: list[str] | None = None) -> None:
        """Compute a set of features and add them to the tracks.

        This involves both updating the node/edge attributes on the tracks.graph
        and adding the features to the FeatureDict, if necessary. This is distinct
        from `update` to allow more efficient bulk computation of features.

        Args:
            feature_keys: Optional list of specific feature keys to compute.
                If None, computes all currently active features. Any provided
                keys not in the currently active set are ignored.

        Raises:
            NotImplementedError: If not implemented in subclass.
        """
        raise NotImplementedError("Must implement compute in the annotator subclass")

    def update(self, action: BasicAction) -> None:
        """Update a set of features based on the given action.

        This involves both updating the node or edge attributes on the tracks.graph
        and adding the features to the FeatureDict, if necessary. This is distinct
        from `compute` to allow more efficient computation of features for single
        elements.

        The action contains all necessary information about which elements to update
        (e.g., AddNode.node, AddEdge.edge, UpdateNodeSeg.node).

        Args:
            action (BasicAction): The action that triggered this update

        Raises:
            NotImplementedError: If not implemented in subclass.
        """
        raise NotImplementedError("Must implement update in the annotator subclass")

    def change_key(self, old_key: str, new_key: str) -> None:
        """Rename a feature key in this annotator.

        Base implementation updates the all_features dictionary. Subclasses should
        override this method to update any additional internal mappings they maintain.

        Args:
            old_key: Existing key to rename.
            new_key: New key to replace it with.

        Raises:
            KeyError: If old_key does not exist in all_features.
        """
        if old_key not in self.all_features:
            raise KeyError(f"Cannot rename missing feature key: {old_key!r}")

        # Update all_features dictionary
        value = self.all_features.pop(old_key)
        self.all_features[new_key] = value
