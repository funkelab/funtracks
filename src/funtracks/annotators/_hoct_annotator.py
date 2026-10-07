from __future__ import annotations

from ._trainable_annotator import TrainableAnnotator


class HoctAnnotator(TrainableAnnotator):
    """Score edges with HOCT and learn a probe from edge corrections.

    HOCT weights are downloaded on first use, or loaded from a local
    checkpoint.

    Args:
        tracks: Tracks whose full candidate graph will be annotated in place.
        model: HOCT model name, checkpoint path, or an already loaded torch module.
        retrain_every: Trigger model update after that many corrections.
        device: Device to use for the model.
        window_size: Minimum number of frames in an input window.
        n_steps: Maximum L-BFGS iterations for each correction update.
    """

    label_mask_key = "hoct_is_labeled"
    label_key = "hoct_label"

    def __init__(
        self,
        tracks,
        *,
        model=None,
        retrain_every: int = 10,
        device: str = "cpu",
        window_size: int = 5,
        n_steps: int = 500,
    ):
        if (
            not isinstance(retrain_every, int)
            or isinstance(retrain_every, bool)
            or retrain_every < 1
        ):
            raise ValueError("retrain_every must be a positive integer")
        if window_size < 2:
            raise ValueError("window_size must be at least 2")
        if n_steps < 1:
            raise ValueError("n_steps must be positive")
        super().__init__(tracks)
        self.model = model if hasattr(model, "forward") else None
        self._model_spec = model
        self.device = device
        self.retrain_every = retrain_every
        self.window_size = window_size
        self.n_steps = n_steps
        self._pending_corrections = 0

    def _load_model(self):
        if self.model is None:
            from hoct import load_model

            self.model = load_model(self._model_spec, device=self.device)
        return self.model

    def _dataset(self):
        from ._hoct_dataset import HoctDataset

        return HoctDataset(self.tracks, self.window_size)

    def compute(self, feature_keys: list[str] | None = None) -> None:
        """Write HOCT link probabilities for every edge in the full graph.

        Args:
            feature_keys: Optional subset of enabled feature keys to compute.
        """
        if self.score_key not in self._filter_feature_keys(feature_keys):
            return
        if self.graph.num_edges() == 0:
            return
        from hoct.inference import predict_edge_scores

        if self.score_key not in self.tracks.features:
            self.tracks.add_feature(self.score_key, self.features[self.score_key])
        scores = predict_edge_scores(self._load_model(), self._dataset())
        self.graph.update_edge_attrs(
            edge_ids=scores["edge_id"].to_list(),
            attrs={self.score_key: scores["similarity"].to_list()},
        )

    def add_training_sample(self, edge, selected: bool) -> None:
        """Record a correction.

        Args:
            edge: Source and target node IDs of an edge in the full graph.
            selected: True for a link, False for a rejected link.
        """
        import polars as pl

        edge = tuple(edge)
        edge_id = self.graph.edge_id(*edge)
        for key in (self.label_mask_key, self.label_key):
            if key not in self.graph.edge_attr_keys():
                self.graph.add_edge_attr_key(key, pl.Boolean, False)
        self.graph.update_edge_attrs(
            edge_ids=[edge_id],
            attrs={self.label_mask_key: [True], self.label_key: [bool(selected)]},
        )
        self.training_data[edge] = bool(selected)
        self._pending_corrections += 1
        if self._pending_corrections >= self.retrain_every:
            from hoct.correction import fit_from_labels

            self.model = fit_from_labels(
                self.graph,
                self._load_model(),
                self.label_mask_key,
                self.label_key,
                dataset=self._dataset(),
                n_steps=self.n_steps,
                consistency_weight=0.0,
            )
            self._pending_corrections = 0
            self.compute()
