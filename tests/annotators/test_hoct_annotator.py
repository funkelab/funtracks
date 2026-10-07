# ruff: noqa: E402
import sys

import numpy as np
import polars as pl
import pytest

# HOCT is an optional dependency of funtracks.
torch = pytest.importorskip("torch")
pytest.importorskip("hoct")
from hoct.correction import ProbedModel
from hoct.inference import ModelPrediction
from torch import nn
from tracksdata.nodes import Mask

from funtracks.actions import AddEdge, DeleteEdge
from funtracks.annotators import HoctAnnotator, TrainableAnnotator
from funtracks.annotators._hoct_dataset import HoctDataset
from funtracks.data_model import Tracks
from funtracks.utils.tracksdata_utils import create_empty_graph


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(2, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, inputs, positions, edge_pos, edges, node_mask, edge_mask):
        features = edge_pos[..., -2:] / 10
        return ModelPrediction(
            self.head(features),
            inputs,
            features,
            torch.zeros(*inputs.shape[:2], 1, device=inputs.device),
        )


@pytest.fixture(params=["memory", "sql"])
def tracks(request, tmp_path):
    graph = create_empty_graph(
        node_attributes=["pos"],
        backend=request.param,
        database=str(tmp_path / "graph.db"),
    )
    graph.bulk_add_nodes(
        nodes=[
            {"t": t, "pos": pos}
            for t, pos in zip(
                [0, 0, 1, 2],
                [[1.0, 2.0], [10.0, 20.0], [3.0, 4.0], [5.0, 6.0]],
                strict=True,
            )
        ],
        indices=[1, 2, 3, 4],
    )
    for source, target in [(1, 3), (2, 3), (3, 4)]:
        graph.add_edge(source, target, attrs={}, validate_keys=False)
    return Tracks(graph, time_attr="t", ndim=3)


def attach(tracks, n=2, **kwargs):
    ann = HoctAnnotator(tracks, model=FakeModel(), n=n, n_steps=15, **kwargs)
    tracks.annotators[:] = [
        a for a in tracks.annotators if not isinstance(a, TrainableAnnotator)
    ]
    tracks.annotators.append(ann)
    tracks.enable_features([ann.score_key])
    return ann


def test_scores_full_graph_and_no_solver(tracks):
    graph = tracks.graph_full
    graph.update_edge_attrs(edge_ids=[graph.edge_id(2, 3)], attrs={"solution": [False]})
    solution_before = graph.edge_attrs(attr_keys=["solution"])
    ann = attach(tracks)
    assert ann.graph is graph
    assert HoctDataset(tracks, 5).graph is graph
    assert tracks.get_edge_attr((1, 3), ann.score_key) == pytest.approx(1 / 3)
    assert tracks.get_edge_attr((2, 3), ann.score_key) == pytest.approx(1 / 3)
    assert tracks.get_edge_attr((3, 4), ann.score_key) == pytest.approx(1 / 2)
    assert graph.edge_attrs(attr_keys=["solution"]).equals(solution_before)
    assert "hoct.tracking" not in sys.modules
    assert "gurobipy" not in sys.modules


def test_corrections_refit_frozen_model_and_refresh(tracks):
    ann = attach(tracks)
    backbone = ann.model
    weights = backbone.head.weight.detach().clone()
    ann.add_training_sample((1, 3), True)
    assert ann.model is backbone
    ann.add_training_sample((2, 3), False)
    assert isinstance(ann.model, ProbedModel)
    assert ann.model._edge_model is backbone
    assert torch.equal(weights, backbone.head.weight)
    assert ann.training_data == {(1, 3): True, (2, 3): False}
    assert ann._pending_corrections == 0
    assert tracks.get_edge_attr((1, 3), ann.score_key) != pytest.approx(1 / 3)
    ann.add_training_sample((1, 3), False)
    ann.add_training_sample((1, 3), True)
    assert ann.model._edge_model is backbone
    assert ann.training_data[(1, 3)] is True


def test_actions_label_deleted_edge(tracks):
    ann = attach(tracks, n=10)
    DeleteEdge(tracks, (1, 3))
    assert ann.training_data[(1, 3)] is False
    assert tracks.graph_full.edges[tracks.graph_full.edge_id(1, 3)][ann.label_mask_key]
    AddEdge(tracks, (1, 3))
    assert ann.training_data[(1, 3)] is True
    assert ann._pending_corrections == 2


def test_disabled_and_renamed_feature(tracks):
    ann = attach(tracks)
    ann.change_key(ann.score_key, "hoct_score")
    ann.compute(["trainable_score"])
    assert "hoct_score" not in tracks.features
    ann.compute()
    assert tracks.get_edge_attr((1, 3), "hoct_score") == pytest.approx(1 / 3)
    ann.deactivate_features(["hoct_score"])
    ann.model = None
    ann.compute()
    assert ann.model is None


def test_skip_edge_single_edge_window(tracks):
    graph = tracks.graph_full
    graph.bulk_remove_edges(edge_ids=graph.edge_ids())
    graph.add_edge(1, 4, attrs={}, validate_keys=False)
    ann = attach(tracks, window_size=2, n=1)
    assert tracks.get_edge_attr((1, 4), ann.score_key) == pytest.approx(0.5)
    ann.add_training_sample((1, 4), True)
    assert isinstance(ann.model, ProbedModel)


def test_mask_features_and_input_layout(tracks):
    graph = tracks.graph_full
    graph.add_node_attr_key("mask", pl.Object)
    mask = Mask(np.ones((3, 4), dtype=bool), np.array([0, 0, 3, 4]))
    graph.update_node_attrs(
        node_ids=graph.node_ids(), attrs={"mask": [mask] * graph.num_nodes()}
    )
    tracks.scale = [1.0, 2.0, 3.0]
    item = HoctDataset(tracks, 5)[0]
    assert item["node_feats"].shape == (4, 19)
    assert torch.isfinite(item["node_feats"]).all()
    assert item["node_pos"][0].tolist() == [0.0, 1.0, 2.0]
    assert graph.nodes[1]["mask"].mask.ndim == 2


def test_empty_graph(tracks):
    tracks.graph_full.bulk_remove_edges(edge_ids=tracks.graph_full.edge_ids())
    ann = HoctAnnotator(tracks)
    ann.activate_features([ann.score_key])
    ann.compute()
    assert ann.model is None


@pytest.mark.parametrize("n", [0, -1, True, 1.5])
def test_invalid_interval(tracks, n):
    with pytest.raises(ValueError, match="positive integer"):
        HoctAnnotator(tracks, n=n)


def test_pretrained_loader_is_lazy(tracks, monkeypatch):
    import hoct

    calls = []
    model = FakeModel()

    def load(spec, *, device):
        calls.append((spec, device))
        return model

    monkeypatch.setattr(hoct, "load_model", load)
    ann = HoctAnnotator(tracks, model="general_v1")
    ann.compute()
    assert not calls
    ann.activate_features([ann.score_key])
    ann.compute()
    ann.compute()
    assert calls == [("general_v1", "cpu")]


def test_custom_time_and_axis_position_keys(tracks):
    graph = tracks.graph_full
    graph.add_node_attr_key("frame", pl.Int64, 0)
    for key in ["height", "width"]:
        graph.add_node_attr_key(key, pl.Float32, 0.0)
    ids = graph.node_ids()
    graph.update_node_attrs(
        node_ids=ids,
        attrs={
            "frame": [0, 0, 1, 2],
            "height": [1.0, 10.0, 3.0, 5.0],
            "width": [2.0, 20.0, 4.0, 6.0],
        },
    )
    custom = Tracks(graph, time_attr="frame", pos_attr=["height", "width"], ndim=3)
    ann = attach(custom)
    assert custom.get_edge_attr((1, 3), ann.score_key) == pytest.approx(1 / 3)
    assert HoctDataset(custom, 5)[0]["node_pos"][0].tolist() == [0.0, 1.0, 2.0]


def test_3d_positions(tracks):
    graph = create_empty_graph(node_attributes=["pos"], ndim=4)
    graph.bulk_add_nodes(
        nodes=[{"t": 0, "pos": [1.0, 2.0, 3.0]}, {"t": 1, "pos": [4.0, 5.0, 6.0]}],
        indices=[1, 2],
    )
    graph.add_edge(1, 2, attrs={}, validate_keys=False)
    tracks3d = Tracks(graph, time_attr="t", ndim=4)
    attach(tracks3d)
    item = HoctDataset(tracks3d, 5)[0]
    assert item["node_feats"].shape == (2, 19)
    assert item["node_pos"][0].tolist() == [1.0, 2.0, 3.0]
