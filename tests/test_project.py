from collections import Counter

import funlib.persistence as fp
import networkx as nx
import numpy as np
import pytest

from funtracks import NxGraph, Project, TrackingGraph
from funtracks.features import FeatureSet
from funtracks.params import ProjectParams


def test_invalid_init():
    # neither seg nor graph raises error
    with pytest.raises(
        ValueError, match="At least one of segmentation or cand graph must be provided"
    ):
        Project("test", ProjectParams())


@pytest.mark.parametrize("ndim", [3, 4])
@pytest.mark.parametrize(
    ("use_seg", "use_graph"),
    [
        (True, False),
        (False, True),
        (True, True),
    ],
)
class TestProjectInit:
    def get_seg(self, request, ndim, use_seg):
        seg_name = "segmentation_2d" if ndim == 3 else "segmentation_3d"
        if use_seg:
            segmentation = request.getfixturevalue(seg_name)
            axis_names = ["t", "z", "y", "x"] if ndim == 4 else ["t", "y", "x"]
            seg = fp.Array(segmentation, axis_names=axis_names)
        else:
            seg = None
        return seg

    def get_gt_graph(self, request, ndim):
        graph_name = "graph_2d" if ndim == 3 else "graph_3d"
        gt_graph = request.getfixturevalue(graph_name)
        return gt_graph

    def get_cand_graph(self, request, ndim, use_seg, use_graph):
        gt_graph = self.get_gt_graph(request, ndim)
        if use_graph:
            features = FeatureSet(ndim=ndim, seg=use_seg)
            cand_graph = TrackingGraph(
                NxGraph, nx.create_empty_copy(gt_graph, with_data=True), features
            )
        else:
            cand_graph = None
        return cand_graph

    def test_init(self, request, ndim, use_seg, use_graph):
        seg = self.get_seg(request, ndim, use_seg)
        gt_graph = self.get_gt_graph(request, ndim)
        cand_graph = self.get_cand_graph(request, ndim, use_seg, use_graph)

        project = Project(
            name="test_project",
            project_params=ProjectParams(),
            segmentation=seg,
            cand_graph=cand_graph,
        )
        assert project.name == "test_project"
        assert project.params == ProjectParams()
        assert project.raw is None
        if use_seg:
            assert isinstance(project.segmentation, fp.Array)
        else:
            assert project.segmentation is None
        nodes_only = project.cand_graph._graph
        assert Counter(nodes_only.nodes.keys()) == Counter(gt_graph.nodes.keys())
        for node, data in nodes_only.nodes(data=True):
            if "area" in gt_graph.nodes[node]:
                assert data["area"] == gt_graph.nodes[node]["area"]
            assert data["pos"] == gt_graph.nodes[node]["pos"]
            assert data["time"] == gt_graph.nodes[node]["time"]


@pytest.mark.parametrize("ndim", [3, 4])
class TestProjectUpdateSeg:
    def get_seg(self, request, ndim):
        seg_name = "segmentation_2d" if ndim == 3 else "segmentation_3d"
        segmentation = request.getfixturevalue(seg_name)
        axis_names = ["t", "z", "y", "x"] if ndim == 4 else ["t", "y", "x"]
        return fp.Array(segmentation, axis_names=axis_names)

    def get_project(self, request, ndim):
        return Project(
            "test_seg", ProjectParams(), segmentation=self.get_seg(request, ndim)
        )

    def test_get_set_pixels(self, request, ndim):
        project = self.get_project(request, ndim)

        # set pixel [0, 0, 0, <0>] and [0, 1, 0, <0>] to 10
        pixels = tuple(np.array([0, 0]) for _ in range(ndim))
        pixels[1][1] = 1
        value = 10

        # check that there are no 10s before we change
        assert np.count_nonzero(project.segmentation.data == value).compute() == 0
        project.set_pixels(pixels, value)

        # check that no other pixels were changed
        assert np.count_nonzero(project.segmentation.data == value).compute() == 2

        first_pixel = project.segmentation[0][0][0]
        second_pixel = project.segmentation[0][1][0]
        if ndim == 4:
            first_pixel = first_pixel[0]
            second_pixel = second_pixel[0]
        assert first_pixel == value
        assert second_pixel == value

        features = project.cand_graph.features
        project.cand_graph.add_node(10, {features.time: 0})

        retrieved = project.get_pixels(10)
        np.testing.assert_array_equal(retrieved, pixels)
