from __future__ import annotations

import logging

from motile import Solver, TrackGraph
from motile.constraints import MaxChildren, MaxParents
from motile.costs import Appear, EdgeDistance, EdgeSelection, Split

from funtracks.data_model import Tracks

from .params.solver_params import SolverParams

logger = logging.getLogger(__name__)


def construct_solver(cand_graph: Tracks, solver_params: SolverParams) -> Solver:
    """Construct a motile solver with the parameters specified in the solver
    params object.

    Args:
        cand_graph (nx.DiGraph): The candidate graph to use in the solver
        solver_params (SolverParams): The costs and constraints to use in
            the solver

    Returns:
        Solver: A motile solver with the specified graph, costs, and
            constraints.
    """
    solver = Solver(TrackGraph(cand_graph.graph, frame_attribute=cand_graph.time_attr))
    solver.add_constraint(MaxChildren(solver_params.max_children))
    solver.add_constraint(MaxParents(1))

    # Using EdgeDistance instead of EdgeSelection for the constant cost because
    # the attribute is not optional for EdgeSelection (yet)
    if solver_params.edge_selection_cost is not None:
        solver.add_cost(
            EdgeDistance(
                weight=0,
                position_attribute=cand_graph.pos_attr,
                constant=solver_params.edge_selection_cost,
            ),
            name="edge_const",
        )
    if solver_params.appear_cost is not None:
        solver.add_cost(Appear(solver_params.appear_cost))
    if solver_params.division_cost is not None:
        solver.add_cost(Split(constant=solver_params.division_cost))

    if solver_params.distance_cost is not None:
        solver.add_cost(
            EdgeDistance(
                position_attribute=cand_graph.time_attr,
                weight=solver_params.distance_cost,
            ),
            name="distance",
        )
    if solver_params.iou_cost is not None:
        solver.add_cost(
            EdgeSelection(
                weight=solver_params.iou_cost,
                attribute="iou",
            ),
            name="iou",
        )
    return solver
