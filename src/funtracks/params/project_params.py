from pydantic import Field

from ._base import Params
from .solver_params import SolverParams


class ProjectParams(Params):
    max_move_distance: float = Field(
        50.0,
        title="Max Move Distance",
        description=r"""The maximum distance an object center can move between time frames.
Objects further than this cannot be matched, but making this value larger will increase solving time.""",
    )
    divisions: bool = Field(
        True,
        title="Divisions",
        description=r"""Check this feature if objects divide (e.g. cells)""",
    )
    merges: bool = Field(
        False,
        title="Merges",
        description=r"""Check this feature if objects merge""",
    )
    appearances: bool = Field(
        True,
        title="Appearances",
        description=r"""Check this feature if objects appear (e.g. enter the field of view)""",
    )
    disappearances: bool = Field(
        True,
        title="Disappearances",
        description=r"""Check this feature if objects disappear (e.g. exit the field of view, die)""",
    )

    solver_params: SolverParams
