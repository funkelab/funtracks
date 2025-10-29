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
            warnings.warn(
                f"NodeType.{name} is deprecated and will be removed in funtracks v2.0. "
                "This is a visualization concern and should be moved to motile_tracker.",
                DeprecationWarning,
                stacklevel=2,
            )

        return value


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
