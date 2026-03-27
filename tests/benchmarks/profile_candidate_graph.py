"""Profile candidate graph generation using the same synthetic data as the benchmark.

Usage:
    uv run --with line-profiler python tests/benchmarks/profile_candidate_graph.py
"""

from bench_candidate_graph import MAX_EDGE_DISTANCE, _generate_segmentation
from line_profiler import LineProfiler

from funtracks.candidate_graph.compute_graph import compute_graph_from_seg
from funtracks.candidate_graph.iou import _compute_ious, _get_iou_dict, add_iou
from funtracks.candidate_graph.utils import (
    add_cand_edges,
    create_kdtree,
    nodes_from_segmentation,
)

seg_data = _generate_segmentation()

lp = LineProfiler()
lp.add_function(compute_graph_from_seg)
lp.add_function(nodes_from_segmentation)
lp.add_function(add_cand_edges)
lp.add_function(add_iou)
lp.add_function(_get_iou_dict)
lp.add_function(_compute_ious)
lp.add_function(create_kdtree)

lp_wrapper = lp(compute_graph_from_seg)
lp_wrapper(seg_data, MAX_EDGE_DISTANCE, iou=True)

lp.print_stats()
