from __future__ import annotations

from typing import TYPE_CHECKING

from typing_extensions import override

if TYPE_CHECKING:
    from funtracks.data_model import Tracks


class Action:
    """Base class for all actions that can be applied to Tracks.

    Two types of actions exist:
    - BasicAction: Atomic operations that directly modify tracks and trigger annotations
    - ActionGroup: Composite operations that contain and execute BasicActions
    """

    def __init__(self, tracks: Tracks):
        """A modular change that can be applied to the given Tracks. The tracks must
        be passed in at construction time so that metadata needed to invert the action
        can be extracted.
        The change should be applied in the init function.

        Args:
            tracks (Tracks): The tracks that this action will edit
        """
        self.tracks = tracks

    def inverse(self) -> Action:
        """Get the inverse of this action. Calling this function does undo the action,
        since the change is applied in the action constructor.

        Raises:
            NotImplementedError: if the inverse is not implemented in the subclass

        Returns:
            Action: An action that un-does this action, bringing the tracks
                back to the exact state it had before applying this action.
        """
        raise NotImplementedError("Inverse not implemented")


class BasicAction(Action):
    """Atomic action that directly modifies tracks and triggers annotation updates.

    BasicActions are the primitive operations that annotators listen to and respond to.
    Examples: AddNode, DeleteEdge, UpdateNodeSeg, etc.
    """


class ActionGroup(Action):
    """Composite action that contains and executes multiple Actions.

    ActionGroups are high-level operations that encapsulate application logic.
    They can contain BasicActions or other ActionGroups.
    They are not passed to annotators - only the BasicActions they contain are.
    Examples: UserAddNode, UserDeleteEdge, etc.
    """

    def __init__(
        self,
        tracks: Tracks,
        actions: list[Action],
    ):
        """A group of actions that is also an action.

        This is useful for creating composite actions from atomic BasicActions or other
        ActionGroups. Composite actions can contain application logic and can be un-done
        as a group.

        Args:
            tracks (Tracks): The tracks that this action will edit
            actions (list[Action]): A list of actions contained within the group, in the
                order in which they should be executed. Can be BasicActions or
                ActionGroups.
        """
        super().__init__(tracks)
        self.actions = actions

    @override
    def inverse(self) -> ActionGroup:
        actions = [action.inverse() for action in self.actions[::-1]]
        return ActionGroup(self.tracks, actions)
