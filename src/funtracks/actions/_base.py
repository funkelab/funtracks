from __future__ import annotations

import functools
import inspect
from typing import TYPE_CHECKING, Any

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

    def to_dict(self) -> dict[str, Any]:
        """The action as a JSON-ready dict for the history log.

        The dict holds the class name under ``"a"`` and the constructor arguments
        needed to apply the action again with ``from_dict``.

        Raises:
            NotImplementedError: If this action is not logged (e.g. UpdateTrackIDs,
                since track ids are recomputed on replay).
        """
        raise NotImplementedError(f"{type(self).__name__} is not written to the log")

    @classmethod
    def from_dict(cls, tracks: Tracks, d: dict[str, Any]) -> BasicAction:
        """Apply the action described by a dict from ``to_dict`` to the tracks.

        Raises:
            NotImplementedError: If this action is not logged.
        """
        raise NotImplementedError(f"{cls.__name__} is not read from the log")


class ActionGroup(Action):
    """Composite action that contains and executes multiple Actions.

    ActionGroups are high-level operations that encapsulate application logic.
    They can contain BasicActions or other ActionGroups.
    They are not passed to annotators - only the BasicActions they contain are.
    Examples: UserAddNode, UserDeleteEdge, etc.

    Subclasses apply their actions in ``__init__``. If that ``__init__`` raises, the
    actions applied so far are undone, newest first, so a failing group leaves no
    change outside the undo history. Like undo, this restores the solution, but
    elements the group added stay in ``graph_full`` as soft-deleted candidates.
    """

    # The constructor arguments by name, set by the subclass __init__ wrapper
    _init_args: dict[str, Any]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Only wrap an __init__ the class defines itself: an inherited one is
        # already wrapped on the parent.
        if "__init__" not in cls.__dict__:
            return
        init = cls.__init__
        signature = inspect.signature(init)

        @functools.wraps(init)
        def init_or_roll_back(
            self: ActionGroup, tracks: Tracks, *args: Any, **kwargs: Any
        ) -> None:
            # Kept for to_summary; the arguments are bound before anything runs.
            self._init_args = signature.bind(self, tracks, *args, **kwargs).arguments
            # Only the outermost user action reports to the tracks: a nested one is
            # part of it.
            top_level = tracks._user_action_depth == 0
            if top_level:
                tracks.user_action_started.emit(self)
            tracks._user_action_depth += 1
            try:
                init(self, tracks, *args, **kwargs)
            except Exception as error:
                # Once the action is in the undo history, Ctrl+Z can undo it, so an
                # error raised after that (e.g. by a refresh listener) must not undo
                # it here too.
                undo_stack = tracks.action_history.undo_stack
                rolled_back = not (undo_stack and undo_stack[-1] is self)
                if rolled_back:
                    # Undo what was applied before the error, newest first. actions
                    # does not exist yet if the error came before super().__init__.
                    for action in reversed(getattr(self, "actions", [])):
                        action.inverse()
                if top_level:
                    tracks.user_action_failed.emit(self, error, rolled_back)
                raise
            finally:
                tracks._user_action_depth -= 1

        cls.__init__ = init_or_roll_back  # type: ignore[method-assign]

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

    def to_summary(self) -> dict[str, Any]:
        """A short, human-readable summary of this action, for the history log.

        Defaults to the constructor arguments, without ``tracks`` and private
        (``_``-prefixed) ones. Masks and pixel coordinates are summarized as their
        bounding box and number of pixels. Subclasses may override this.

        Returns:
            dict[str, Any]: The summary. Empty for a plain ActionGroup, which has no
                arguments of its own to show.
        """
        from funtracks.actions.serialization import summarize_value

        args = getattr(self, "_init_args", {})
        return {
            key: summarize_value(value)
            for key, value in args.items()
            if key not in ("self", "tracks") and not key.startswith("_")
        }
