from ._base import Action, ActionGroup, BasicAction
from .add_delete_edge import AddEdge, DeleteEdge
from .add_delete_node import AddNode, DeleteNode
from .update_node_attrs import UpdateNodeAttrs
from .update_segmentation import UpdateNodeSeg
from .update_track_id import UpdateTrackIDs

__all__ = [
    "Action",
    "ActionGroup",
    "AddEdge",
    "AddNode",
    "BasicAction",
    "DeleteEdge",
    "DeleteNode",
    "UpdateNodeAttrs",
    "UpdateNodeSeg",
    "UpdateTrackIDs",
]


def __getattr__(name: str):
    """Provide backwards compatibility for deprecated names."""
    if name == "UpdateTrackID":
        import warnings

        warnings.warn(
            "UpdateTrackID is deprecated and will be removed in funtracks v2.0. "
            "Use UpdateTrackIDs instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return UpdateTrackIDs
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
