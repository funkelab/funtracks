"""Benchmarks for actions (add/delete/update + inverse).

To diagnose regressions, we recommend using line-profiler locally on the
individual action classes in funtracks.actions.
"""

import networkx as nx
import numpy as np
import pytest
from skimage.draw import disk

from funtracks.actions import (
    AddEdge,
    AddNode,
    DeleteEdge,
    UpdateNodeAttrs,
    UpdateNodeSeg,
    UpdateTrackIDs,
)
from funtracks.candidate_graph.iou import add_iou
from funtracks.candidate_graph.utils import nodes_from_segmentation
from funtracks.data_model import SolutionTracks

NUM_FRAMES = 20
FRAME_SHAPE = (256, 256)
CELLS_PER_FRAME = 50
NUM_DIVISIONS = 5


def _generate_segmentation(seed=42):
    """Generate a synthetic segmentation for action benchmarks."""
    rng = np.random.default_rng(seed)
    seg = np.zeros((NUM_FRAMES, *FRAME_SHAPE), dtype=np.uint16)
    label = 1
    for t in range(NUM_FRAMES):
        for _ in range(CELLS_PER_FRAME):
            cy = rng.integers(20, FRAME_SHAPE[0] - 20)
            cx = rng.integers(20, FRAME_SHAPE[1] - 20)
            radius = rng.integers(10, 20)
            rr, cc = disk((cy, cx), radius, shape=FRAME_SHAPE)
            seg[t, rr, cc] = label
            label += 1
    return seg


def _build_solution_graph(seg, seed=42):
    """Build a tree-structured solution graph with 1-to-1 edges and divisions.

    Each cell in frame t connects to the same-index cell in frame t+1.
    A few cells divide: one parent connects to two children.
    """
    rng = np.random.default_rng(seed)
    cand_graph, node_frame_dict = nodes_from_segmentation(seg)
    frames = sorted(node_frame_dict.keys())

    # Build solution graph: 1-to-1 matching between frames
    solution = nx.DiGraph()
    solution.add_nodes_from(cand_graph.nodes(data=True))

    for i in range(len(frames) - 1):
        cur_nodes = node_frame_dict[frames[i]]
        next_nodes = node_frame_dict[frames[i + 1]]
        n = min(len(cur_nodes), len(next_nodes))
        for j in range(n):
            solution.add_edge(cur_nodes[j], next_nodes[j])

    # Add divisions: pick a parent, connect it to a second child
    division_frames = rng.choice(
        range(len(frames) - 1), size=NUM_DIVISIONS, replace=False
    )
    for frame_idx in division_frames:
        cur_nodes = node_frame_dict[frames[frame_idx]]
        next_nodes = node_frame_dict[frames[frame_idx + 1]]
        if len(cur_nodes) < 1 or len(next_nodes) < 2:
            continue
        # Pick a parent that already has one child
        parent = cur_nodes[0]
        # Find an unconnected child in the next frame
        connected = {v for _, v in solution.out_edges(parent)}
        for child in next_nodes:
            if child not in connected:
                solution.add_edge(parent, child)
                break

    add_iou(solution, seg, node_frame_dict)
    return solution


@pytest.fixture(scope="module")
def tracks():
    seg = _generate_segmentation()
    graph = _build_solution_graph(seg)
    tracks = SolutionTracks(graph, segmentation=seg, ndim=3)
    return tracks


def _add_delete_node(tracks):
    next_id = tracks.get_next_track_id()
    action = AddNode(
        tracks,
        node=max(tracks.graph.nodes) + 1,
        attributes={
            "time": 0,
            "pos": [128.0, 128.0],
            "track_id": next_id,
        },
    )
    action.inverse()


def test_add_delete_node(benchmark, tracks):
    benchmark.pedantic(
        _add_delete_node,
        args=(tracks,),
        rounds=50,
        iterations=1,
    )


def _add_delete_edge(tracks, edge):
    action = AddEdge(tracks, edge)
    action.inverse()


def test_add_delete_edge(benchmark, tracks):
    # Pick an edge, remove it so we can benchmark add+delete cycles
    edge = next(iter(tracks.graph.edges))
    DeleteEdge(tracks, edge)

    benchmark.pedantic(
        _add_delete_edge,
        args=(tracks, edge),
        rounds=50,
        iterations=1,
    )

    # Restore the edge
    AddEdge(tracks, edge)


def _update_node_attrs(tracks, node):
    action = UpdateNodeAttrs(tracks, node, {"score": 1.0})
    action.inverse()


def test_update_node_attrs(benchmark, tracks):
    node = next(iter(tracks.graph.nodes))
    benchmark.pedantic(
        _update_node_attrs,
        args=(tracks, node),
        rounds=50,
        iterations=1,
    )


def _update_node_seg(tracks, node, pixels):
    action = UpdateNodeSeg(tracks, node, pixels=pixels, added=True)
    action.inverse()


def test_update_node_seg(benchmark, tracks):
    node = next(iter(tracks.graph.nodes))
    # Create a small patch of pixels to add/remove
    time = tracks.get_time(node)
    rr, cc = disk((128, 128), 5, shape=FRAME_SHAPE)
    pixels = (np.full_like(rr, time), rr, cc)

    benchmark.pedantic(
        _update_node_seg,
        args=(tracks, node, pixels),
        rounds=50,
        iterations=1,
    )


def _update_track_ids(tracks, node, old_id, new_id):
    UpdateTrackIDs(tracks, node, new_id)
    UpdateTrackIDs(tracks, node, old_id)


def test_update_track_ids(benchmark, tracks):
    node = next(iter(tracks.graph.nodes))
    old_id = tracks.get_track_id(node)
    new_id = old_id + 1000

    benchmark.pedantic(
        _update_track_ids,
        args=(tracks, node, old_id, new_id),
        rounds=50,
        iterations=1,
    )
