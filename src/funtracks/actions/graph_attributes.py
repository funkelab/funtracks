from enum import Enum


class NodeAttr(Enum):
    """Node attributes that can be added to candidate graph using the toolbox.
    Note: Motile can flexibly support any custom attributes. The toolbox provides
    implementations of commonly used ones, listed here.
    """

    POS = "pos"
    TIME = "time"
    SEG_ID = "seg_id"
    SEG_HYPO = "seg_hypo"
    AREA = "area"
    TRACK_ID = "track_id"


class EdgeAttr(Enum):
    """Edge attributes that can be added to candidate graph using the toolbox.
    Note: Motile can flexibly support any custom attributes. The toolbox provides
    implementations of commonly used ones, listed here.
    """

    IOU = "iou"


class NodeType(Enum):
    """Types of nodes in the track graph. Currently used for standardizing
    visualization. All nodes are exactly one type.
    """

    SPLIT = "SPLIT"
    END = "END"
    CONTINUE = "CONTINUE"
