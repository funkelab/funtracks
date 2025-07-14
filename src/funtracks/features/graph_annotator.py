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

    def __init__(self, tracks: Tracks, features: list[Feature]):
        """
        Args:
            tracks (Tracks): The tracks to manage features for.
            features (list[Feature]): A list of features that this annotator is
                responsible for computing and updating.
        """
        self.tracks = tracks
        self.features = features

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
