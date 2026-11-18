from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from funtracks.exceptions import InvalidActionError

from ..actions._base import ActionGroup
from ..actions.add_delete_edge import AddEdge
from ..actions.update_track_id import UpdateTrackIDs
from .user_delete_edge import UserDeleteEdge

if TYPE_CHECKING:
    from funtracks.data_model import Tracks


class UserAddEdge(ActionGroup):
    """Assumes that the endpoints already exist and have track ids.

    Args:
        tracks (Tracks): the tracks to add the edge to
        edge (tuple[int, int]): The edge to add
        force (bool, optional): Whether to force the action by removing any conflicting
            edges. Defaults to False.
        _top_level (bool): If True, add this action to the history and emit refresh.
            Set to False when used as a sub-action inside a compound action.
            Defaults to True.
    """

    def __init__(
        self,
        tracks: Tracks,
        edge: tuple[int, int],
        force: bool = False,
        _top_level: bool = True,
    ):
        super().__init__(tracks, actions=[])
        self.tracks: Tracks  # Narrow type from base class
        source, target = edge
        if not tracks.graph_solution.has_node(source):
            raise InvalidActionError(
                f"Source node {source} not in solution yet - must be added before edge"
            )
        if not tracks.graph_solution.has_node(target):
            raise InvalidActionError(
                f"Target node {target} not in solution yet - must be added before edge"
            )

        # Check if making a merge. If yes and force, remove the other edge and update
        # track ids.
        in_degree_target = len(self.tracks.predecessors(target))  # type: ignore
        merge_edge = None
        if in_degree_target > 0:
            if not force:
                raise InvalidActionError(
                    f"Cannot make a merge edge in a tracking solution: node {target} "
                    "already has an in edge",
                    forceable=True,
                )
            pred = next(iter(self.tracks.graph_solution.predecessors(target)))
            merge_edge = (pred, target)

        # Check the out degree of the source before applying anything, so that a
        # failing check leaves the tracks unchanged. Removing the merge edge lowers it
        # when that edge starts at the source.
        out_degree_source = len(self.tracks.successors(source))
        if merge_edge is not None and merge_edge[0] == source:
            out_degree_source -= 1
        if out_degree_source > 1:
            raise InvalidActionError(
                f"Expected degree of 0 or 1 before adding edge, got {out_degree_source}"
            )

        if merge_edge is not None:
            warnings.warn(
                f"Removing edge {merge_edge} to add new edge without merging.",
                stacklevel=2,
            )
            self.actions.append(UserDeleteEdge(self.tracks, merge_edge, _top_level=False))

        # update track ids if needed
        if out_degree_source == 0:  # joining two segments
            # assign the track id and lineage id of the source node to the target
            # and all downstream nodes
            new_track_id = self.tracks.get_track_id(source)
            new_lineage_id = self.tracks.get_lineage_id(source)
            self.actions.append(
                UpdateTrackIDs(self.tracks, target, new_track_id, new_lineage_id)
            )
        else:  # creating a division
            # assign a new track id to existing child (lineage stays the same)
            successor = next(iter(self.tracks.graph_solution.successors(source)))
            self.actions.append(
                UpdateTrackIDs(self.tracks, successor, self.tracks.get_next_track_id())
            )
            # Assign source's lineage id to the new child (target joins source's lineage)
            self.actions.append(
                UpdateTrackIDs(
                    self.tracks, target, lineage_id=self.tracks.get_lineage_id(source)
                )
            )

        self.actions.append(AddEdge(tracks, edge))

        if _top_level:
            self.tracks.action_history.add_new_action(self)
            self.tracks.refresh.emit()
