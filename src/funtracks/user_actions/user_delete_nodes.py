from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..actions._base import ActionGroup
from .user_delete_node import UserDeleteNode

if TYPE_CHECKING:
    from funtracks.data_model import SolutionTracks


class UserDeleteNodes(ActionGroup):
    """Delete multiple nodes in a single action.

    Wraps individual UserDeleteNode calls so that history and refresh
    are handled once for the entire batch, rather than per-node.

    Args:
        tracks: The tracks to delete nodes from.
        nodes: The node ids to delete.
        pixels: Optional list of pixel masks for each node, matching the order
            of nodes. Defaults to None.
    """

    def __init__(
        self,
        tracks: SolutionTracks,
        nodes: list[int],
        pixels: None | list[tuple[np.ndarray, ...]] = None,
    ):
        super().__init__(tracks, actions=[])
        self.tracks: SolutionTracks  # Narrow type from base class
        for i, node in enumerate(nodes):
            self.actions.append(
                UserDeleteNode(
                    tracks,
                    node,
                    pixels=pixels[i] if pixels is not None else None,
                    _top_level=False,
                )
            )

        self.tracks.action_history.add_new_action(self)
        self.tracks.refresh.emit()
