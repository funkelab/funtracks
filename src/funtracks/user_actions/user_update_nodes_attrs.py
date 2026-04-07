from __future__ import annotations

from typing import TYPE_CHECKING

from ..actions._base import ActionGroup
from .user_update_node_attrs import UserUpdateNodeAttrs

if TYPE_CHECKING:
    from typing import Any

    from funtracks.data_model import SolutionTracks


class UserUpdateNodesAttrs(ActionGroup):
    """Update attributes on multiple nodes in a single action.

    Wraps individual UserUpdateNodeAttrs calls so that history and refresh
    are handled once for the entire batch, rather than per-node.

    Args:
        tracks: The tracks to update node attributes for.
        nodes: The node ids to update.
        attrs: A dict mapping attribute names to a list of values, one per node.
            For scalar attributes: ``{"score": [0.1, 0.9]}``.
            For array attributes: ``{"pos": [[1, 2], [3, 4]]}``.
            All lists must have the same length as nodes.
    """

    def __init__(
        self,
        tracks: SolutionTracks,
        nodes: list[int],
        attrs: dict[str, list[Any]],
    ):
        super().__init__(tracks, actions=[])
        self.tracks: SolutionTracks  # Narrow type from base class
        for key, values in attrs.items():
            if not isinstance(values, list):
                raise ValueError(
                    f"Values for attribute '{key}' must be a list, "
                    f"got {type(values).__name__}"
                )
            if len(values) != len(nodes):
                raise ValueError(
                    f"Values list for attribute '{key}' has length {len(values)}, "
                    f"expected {len(nodes)}"
                )
        for i, node in enumerate(nodes):
            node_attrs = {key: values[i] for key, values in attrs.items()}
            self.actions.append(
                UserUpdateNodeAttrs(tracks, node, node_attrs, _top_level=False)
            )

        self.tracks.action_history.add_new_action(self)
        self.tracks.refresh.emit()
