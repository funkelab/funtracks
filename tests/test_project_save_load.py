import funlib.persistence as fp
import geff.utils
import numpy as np
import pytest
import zarr

from funtracks import CandGraph, NxGraph, Project, TrackingGraph
from funtracks.features import FeatureSet
from funtracks.params import CandGraphParams, ProjectParams


@pytest.mark.parametrize("ndim", [3, 4])
@pytest.mark.parametrize("use_seg", [True, False])
@pytest.mark.parametrize("use_raw", [True, False])
@pytest.mark.parametrize("use_cand_graph", [True, False])
class TestSaveLoadInternal:
    def get_project(self, request, ndim, use_seg, use_raw, use_cand_graph):
        params = ProjectParams()
        seg_name = "segmentation_2d" if ndim == 3 else "segmentation_3d"

        segmentation = request.getfixturevalue(seg_name)
        axis_names = ["t", "z", "y", "x"] if ndim == 4 else ["t", "y", "x"]
        if use_seg:
            seg = fp.Array(segmentation, axis_names=axis_names, voxel_size=None)
        else:
            seg = None

        if use_raw:
            raw = np.zeros_like(segmentation)
            raw = [fp.Array(raw, axis_names=axis_names)]
        else:
            raw = None

        gt_graph = self.get_gt_graph(request, ndim)
        features = FeatureSet(ndim=ndim, seg=use_seg)
        if use_cand_graph:
            graph = CandGraph(NxGraph, gt_graph, features, CandGraphParams())
        else:
            graph = TrackingGraph(NxGraph, gt_graph, features)
        return Project("test", params, segmentation=seg, raw=raw, graph=graph)

    def get_gt_graph(self, request, ndim):
        graph_name = "graph_2d" if ndim == 3 else "graph_3d"
        gt_graph = request.getfixturevalue(graph_name)
        return gt_graph

    def test_save_project(
        self, tmp_path, request, ndim, use_seg, use_raw, use_cand_graph
    ):
        project = self.get_project(
            request, ndim, use_seg=use_seg, use_raw=use_raw, use_cand_graph=use_cand_graph
        )
        zarr_name = f"{project.name}.zarr"
        zarr_path = tmp_path / zarr_name
        project.save(zarr_path)

        zroot = zarr.open(zarr_path)
        if use_seg:
            assert "seg" in zroot.array_keys()
            # more robust testing is the responsibility of funlib.persistence
            np.testing.assert_array_almost_equal(project.segmentation.data, zroot["seg"])
        else:
            assert "seg" not in zroot.array_keys()
        if use_raw:
            print(list(zroot.keys()))
            assert "raw" in zroot.group_keys()
            raw_group = zroot["raw"]
            print(list(raw_group.array_keys()))
            assert "0" in raw_group.array_keys()
            # more robust testing is the responsibility of funlib.persistence
            np.testing.assert_array_almost_equal(project.raw[0].data, zroot["raw/0"])
        else:
            assert "raw" not in zroot.array_keys()

        attrs = zroot.attrs["project_params"]
        assert attrs["divisions"] == project.params.divisions
        assert attrs["merges"] == project.params.merges
        assert attrs["appearances"] == project.params.appearances
        assert attrs["disappearances"] == project.params.disappearances
        assert "max_move_distance" not in attrs

        solver_attrs = zroot.attrs["solver_params"]
        solver_params = project.solver_params
        assert solver_attrs["max_children"] == solver_params.max_children
        assert solver_attrs["edge_selection_cost"] == solver_params.edge_selection_cost
        assert solver_attrs["appear_cost"] == solver_params.appear_cost
        assert solver_attrs["distance_cost"] == solver_params.distance_cost
        assert solver_attrs["iou_cost"] == solver_params.iou_cost
        assert solver_attrs["division_cost"] == solver_params.division_cost
        if use_cand_graph:
            assert "graph" in zroot.group_keys()
            geff.utils.validate(zarr_path / "graph")
            # more robust testing of content is the responsibility of geff
            cand_graph_group = zroot["graph"]
            g_attrs = cand_graph_group.attrs["cand_graph_params"]
            g_params = project.graph.params
            assert g_attrs["max_move_distance"] == g_params.max_move_distance
            assert g_attrs["max_neighbors"] == g_params.max_neighbors
            assert g_attrs["max_frame_span"] == g_params.max_frame_span

    def test_load_project(
        self, tmp_path, request, ndim, use_seg, use_raw, use_cand_graph
    ):
        project = self.get_project(
            request, ndim, use_seg=use_seg, use_raw=use_raw, use_cand_graph=use_cand_graph
        )
        zarr_name = f"{project.name}.zarr"
        zarr_path = tmp_path / zarr_name
        project.save(zarr_path)

        loaded = Project.load(zarr_path)
        assert project.name == loaded.name
        assert project.params == loaded.params
        if use_cand_graph:
            assert isinstance(project.graph, CandGraph)
            assert project.graph.params == loaded.graph.params
        else:
            assert not isinstance(project.graph, CandGraph)
        assert project.solver_params == loaded.solver_params

        if use_seg:
            assert loaded.segmentation is not None
            np.testing.assert_array_almost_equal(
                project.segmentation.data.compute(), loaded.segmentation.data.compute()
            )
        else:
            assert loaded.segmentation is None
        if use_raw:
            assert loaded.raw is not None
            np.testing.assert_array_almost_equal(
                project.raw[0].data.compute(), loaded.raw[0][:]
            )
        else:
            assert loaded.raw is None

        from collections import Counter

        assert Counter(project.graph.nodes) == Counter(loaded.graph.nodes)
        assert Counter(project.graph.edges) == Counter(loaded.graph.edges)
