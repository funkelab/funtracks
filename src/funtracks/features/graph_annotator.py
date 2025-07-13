from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from funtracks.data_model import Tracks

    from .feature import Feature


class GraphAnnotator:
    """A base class for adding and updating graph features.

    This class holds a set of features that is responsible for adding and updating
    to Tracks objects.
    """

    def __init__(self, features: list[Feature]):
        """
        Args:
            features (list[Feature]): A list of features that this annotator is
                responsible for computing and updating.
        """
        self.features = features

    def compute(self, tracks: Tracks, features: Feature | list[Feature]) -> None:
        """Compute a set of features and add them to the given tracks.

        This involves both updating the node/edge attributes for the tracks.graph
        and adding the features to the FeatureSet, if necessary. This is distinct
        from `update` to allow more efficient bulk computation of features.

        Args:
            tracks (Tracks): The tracks to add the features to
            features (Feature | list[Feature]): The feature or features to compute

        Raises:
            NotImplementedError: If not implemented in subclass and you attempt to call
                it.
        """
        raise NotImplementedError("Must implement compute in the annotator subclass")

    def update(
        self,
        tracks: Tracks,
        features: Feature | list[Feature],
        element: int | tuple[int, int],
    ) -> None:
        """Update a set of features for a given node or edge.

        This involves both updating the node or edge attributes for the tracks.graph
        and adding the features to the FeatureSet, if necessary. This is distinct
        from `compute` to allow more efficient computation of features for single
        elements.

        Args:
            tracks (Tracks): The tracks to update
            features (Feature | list[Feature]): The feature or features to update
            element (int | tuple[int, int]): The node or edge to update the features for

        Raises:
            NotImplementedError: If not implemented in subclass and you attempt to call
                it.
        """
        raise NotImplementedError("Must implement update in the annotator subclass")
