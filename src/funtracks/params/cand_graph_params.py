from pydantic import Field

from ._base import Params


class CandGraphParams(Params):
    max_move_distance: float = Field(
        50.0,
        title="Max Move Distance",
        description=r"""The maximum distance an object center can move between time frames.
Objects further than this cannot be matched, but making this value larger will increase solving time.""",
    )
    max_neighbors: int | None = Field(None)
    max_frame_span: int = Field(1, title="Max Frame Edge Span")
