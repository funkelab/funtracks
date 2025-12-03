import warnings
from enum import Enum, EnumMeta


class DeprecatedEnumMeta(EnumMeta):
    """Metaclass for deprecated enums that issues warnings on member access."""

    def __getattribute__(cls, name):
        """Issue deprecation warning when accessing enum members."""
        # Get the attribute first to avoid blocking access
        value = super().__getattribute__(name)

        # Issue warning only for actual enum members (not special attributes)
        if (
            not name.startswith("_")
            and name not in ("name", "value")
            and isinstance(value, cls)
        ):
            enum_name = cls.__name__

            # Customize message based on enum type
            if enum_name == "NodeType":
                message = (
                    f"NodeType.{name} is deprecated and will be removed in "
                    "funtracks v2.0. This is a visualization concern and "
                    "should be moved to motile_tracker."
                )
            elif enum_name == "NodeAttr":
                message = (
                    f"NodeAttr.{name} is deprecated and will be removed in "
                    "funtracks v2.0. Use string keys from tracks.features "
                    "instead (e.g., tracks.features.position_key)."
                )
            elif enum_name == "EdgeAttr":
                message = (
                    f"EdgeAttr.{name} is deprecated and will be removed in "
                    "funtracks v2.0. Use string keys directly (e.g., 'iou')."
                )
            else:
                message = (
                    f"{enum_name}.{name} is deprecated and will be removed "
                    "in funtracks v2.0."
                )

            warnings.warn(message, DeprecationWarning, stacklevel=2)

        return value


class NodeAttr(Enum, metaclass=DeprecatedEnumMeta):
    """Node attributes that can be added to candidate graph.

    .. deprecated:: 2.0
        NodeAttr enum will be removed in funtracks v2.0. Use string keys from
        tracks.features instead (e.g., tracks.features.position_key, "area", etc.).
    """

    POS = "pos"
    TIME = "time"
    AREA = "area"
    TRACK_ID = "track_id"
    SEG_ID = "seg_id"


class EdgeAttr(Enum, metaclass=DeprecatedEnumMeta):
    """Edge attributes that can be added to candidate graph.

    .. deprecated:: 2.0
        EdgeAttr enum will be removed in funtracks v2.0. Use string keys directly
        (e.g., "iou").
    """

    IOU = "iou"


class NodeType(Enum, metaclass=DeprecatedEnumMeta):
    """Types of nodes in the track graph. Currently used for standardizing
    visualization. All nodes are exactly one type.

    .. deprecated:: 2.0
        NodeType will be removed in funtracks v2.0. This is a visualization
        concern and should be moved to motile_tracker.
    """

    SPLIT = "SPLIT"
    END = "END"
    CONTINUE = "CONTINUE"
