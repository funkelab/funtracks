from enum import Enum


class NodeAttr(Enum):
    """Node attributes that can be added to candidate graph using the toolbox.
    Note: Motile can flexibly support any custom attributes. The toolbox provides
    implementations of commonly used ones, listed here.
    """

    TRACK_ID = "track_id"


class NodeType(Enum):
    """Types of nodes in the track graph. Currently used for standardizing
    visualization. All nodes are exactly one type.
    """

    SPLIT = "SPLIT"
    END = "END"
    CONTINUE = "CONTINUE"
