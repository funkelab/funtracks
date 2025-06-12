from pydantic import Field

from ._base import Params


class ProjectParams(Params):
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
