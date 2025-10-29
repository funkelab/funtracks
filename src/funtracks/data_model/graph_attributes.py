from enum import Enum


class NodeType(Enum):
    """Types of nodes in the track graph. Currently used for standardizing
    visualization. All nodes are exactly one type.
    TODO: Move to motile_tracker, since it's a visualization problem
    """

    SPLIT = "SPLIT"
    END = "END"
    CONTINUE = "CONTINUE"
