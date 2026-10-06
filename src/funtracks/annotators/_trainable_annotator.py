from ..actions import AddEdge, BasicAction, DeleteEdge
from ..features._feature import Feature
from ._graph_annotator import GraphAnnotator

DEFAULT_SCORE_KEY = "trainable_score"


def TrainableScore() -> Feature:
    """A feature representing the predicted score (link vs. not link) of a
    trainable annotator.

    Values will be between 0 (do not link) and 1 (do link).

    Returns:
        Feature: A feature dict representing the trainable score.
    """
    return {
        "feature_type": "edge",
        "value_type": "float",
        "num_values": 1,
        "display_name": "Trainable Score",
        "default_value": 0.5,
    }


class TrainableAnnotator(GraphAnnotator):
    @classmethod
    def can_annotate(cls, tracks) -> bool:
        # we assume that all tracks can be annotated, subclasses for specific
        # methods can disagree here
        return True

    @classmethod
    def get_available_features(cls, ndim: int = 3) -> dict[str, Feature]:
        return {DEFAULT_SCORE_KEY: TrainableScore()}

    def __init__(self, tracks) -> None:
        self.score_key = DEFAULT_SCORE_KEY
        feats = {DEFAULT_SCORE_KEY: TrainableScore()}
        self.training_data = {}
        super().__init__(tracks, feats)

    def update(self, action: BasicAction):
        """Record an action if suitable for training.

        Args:
            action (BasicAction): The action that triggered this update
        """
        # only record adding / removing
        if not isinstance(action, (AddEdge, DeleteEdge)):
            return

        self.add_training_sample(action.edge, isinstance(action, AddEdge))

    def change_key(self, old_key: str, new_key: str) -> None:
        # Call base implementation to update all_features
        super().change_key(old_key, new_key)

        if self.score_key == old_key:
            self.score_key = new_key

    def add_training_sample(self, edge, selected: bool) -> None:
        """Add an edge and its GT selection (true or false) as a training sample.

        Args:
            edge: The edge.
            selected: Whether the edge is a positive or negative example.
        """
        raise NotImplementedError()

    def compute(self, feature_keys: list[str] | None = None) -> None:
        """Predict edge scores and set them as a feature for each edge.

        Args:
            feature_keys: Optional list of specific feature keys to compute.
            Not used here.
        """
        raise NotImplementedError()
