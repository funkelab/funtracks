import logging

from ..actions import BasicAction
from ._trainable_annotator import TrainableAnnotator

logger = logging.getLogger(__name__)


class DummyTrainableAnnotator(TrainableAnnotator):
    """Assign a score of 0.5 for each edge."""

    def compute(self, feature_keys: list[str] | None = None) -> None:
        for edge in self.graph.edge_list():
            self.tracks._set_edge_attr(edge, self.score_key, 0.5)

    def add_training_sample(self, edge, selected):
        logger.info(
            "[DummyTrainableAnnotator] got new training sample: edge %s should be %s",
            edge,
            selected,
        )

        # record a training sample
        self.training_data[edge] = selected
