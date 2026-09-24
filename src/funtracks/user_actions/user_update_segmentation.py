from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import TYPE_CHECKING, TypeAlias, cast

import numpy as np
from tracksdata.nodes import Mask

from funtracks.utils.tracksdata_utils import pixels_to_td_mask, union_td_masks

from ..actions._base import ActionGroup
from ..actions.update_segmentation import UpdateNodeSeg
from .user_add_node import UserAddNode
from .user_delete_node import UserDeleteNode

if TYPE_CHECKING:
    from funtracks.data_model import Tracks

# The two forms an updated-pixels entry can take: the mask form the segmentation
# is stored as, and the multi-index form callers that only have pixel coordinates
# report. Both carry the old value last.
MaskUpdate: TypeAlias = tuple[Mask, int, int]
MultiIndexUpdate: TypeAlias = tuple[tuple[np.ndarray, ...], int]


def _create_masks_from_multi_index(
    updates: Sequence[MultiIndexUpdate],
    ndim: int,
) -> list[MaskUpdate]:
    """Turn multi-index updated-pixels entries into one mask per node.

    Entries come as (multi-index, old value) from callers that only have pixel
    coordinates. A label may be reported more than once, so entries are grouped
    per label and time point, and one mask is built per group.

    Args:
        updates: The entries to combine. A multi-index entry has one array of
            coordinates per dimension, time first, and all of its coordinates
            belong to the same time point.
        ndim: The number of dimensions of the segmentation, time included.

    Returns:
        list[tuple[Mask, int, int]]: one (mask, time, old value) per label and
            time point, ordered by those, so the actions built from them do not
            depend on the order the entries came in.

    Raises:
        ValueError: If an entry's coordinates do not all belong to one time point.
            The mask built from an entry has no time axis, so pixels from two time
            points would silently collapse into one mask.
    """

    pixels_per_node: dict[tuple[int, int], list[tuple[np.ndarray, ...]]] = defaultdict(
        list
    )
    for pixels, old_value in updates:
        if len(pixels[0]) == 0:
            # An entry with no pixels changes nothing, so it has no time point to
            # group it under either.
            continue
        times = np.unique(pixels[0])
        if len(times) != 1:
            raise ValueError(
                "Each updated-pixels entry must cover a single time point, but one "
                f"for value {old_value} covers {times.tolist()}. Split it per time "
                "point before passing it in."
            )
        pixels_per_node[(old_value, int(times[0]))].append(pixels)

    combined = []
    for (old_value, time), entries in sorted(pixels_per_node.items()):
        # Gather this node's coordinates before building anything, so the mask is
        # built once for the node rather than once per entry.
        coordinates = tuple(
            np.concatenate([pixels[dim] for pixels in entries]) for dim in range(ndim)
        )
        combined.append((pixels_to_td_mask(coordinates, ndim), time, old_value))

    return combined


def _create_masks_from_bboxes(
    updates: Sequence[MaskUpdate],
) -> list[MaskUpdate]:
    """Combine mask updated-pixels entries into one mask per node.

    Entries come as (mask, time, old value), the form the segmentation is stored
    as. A label may be reported more than once, so entries are grouped per label
    and time point and their masks combined.

    Args:
        updates: The entries to combine.

    Returns:
        list[tuple[Mask, int, int]]: one (mask, time, old value) per label and
            time point, ordered by those, so the actions built from them do not
            depend on the order the entries came in. Each mask's bounding box is
            tightened around its own pixels, whatever box it came in with.
    """

    masks_per_node: dict[tuple[int, int], list[Mask]] = defaultdict(list)
    for mask, time, old_value in updates:
        masks_per_node[(old_value, time)].append(mask)

    return [
        (union_td_masks(masks), time, old_value)
        for (old_value, time), masks in sorted(masks_per_node.items())
    ]


class UserUpdateSegmentation(ActionGroup):
    def __init__(
        self,
        tracks: Tracks,
        new_value: int,
        updated_pixels: list[MultiIndexUpdate | MaskUpdate],
        current_track_id: int,
        force: bool = False,
        _top_level: bool = True,
    ):
        """Assumes that the pixels have already been updated in the project.segmentation
        NOTE: Re discussion with Kasia: we should have a basic action that updates the
        segmentation, and that is the only place the segmentation is updated. The basic
        add_node action doesn't have anything with pixels.

        Args:
            tracks (Tracks): The solution tracks that the user is updating.
            new_value (int): The new value that the user painted with
            updated_pixels: The pixels that changed, as a list of entries that are
                each either a (mask, time, old value) triple, or a (multi-index, old
                value) pair whose multi-index points to the array elements that were
                changed (a tuple with len ndims). Prefer the mask form since it is in
                the right format already. A label may appear in more than one entry and
                those entries are combined.
            current_track_id (int): The track id to use if adding a new node, usually
                the currently selected track id in the viewer.
            force (bool): Whether to force the operation by removing conflicting edges.
                Defaults to False.
            _top_level (bool): If True, add this action to the history and emit the
                refresh signal. Set to False when this action is part of a bigger action
                group, so that the whole group is undone in one step. Defaults to True.
        """
        super().__init__(tracks, actions=[])
        self.tracks: Tracks  # Narrow type from base class
        node_to_select = None
        if self.tracks.segmentation is None:
            raise ValueError("Cannot update non-existing segmentation.")

        # Discard entries where pixels get overwritten with the same value. Both
        # accepted forms carry the old value last.
        updated_pixels = [update for update in updated_pixels if update[-1] != new_value]

        # Updated pixels can either be provided as (Mask, time, old value) triple, or
        # as a multi-index entry (multi-index, old value) pair (napari ≤0.7). The
        # whole list must use one form, so look at every entry rather than the first:
        # a mixed list would otherwise pick a branch and fail on an unpack later.
        # mypy cannot narrow a union of tuple types by their length, so name the
        # form each branch has established.
        updates: list[MaskUpdate]
        lengths = {len(update) for update in updated_pixels}
        if not updated_pixels:
            updates = []
        elif lengths == {3}:
            updates = _create_masks_from_bboxes(
                cast("Sequence[MaskUpdate]", updated_pixels)
            )
        elif lengths == {2}:
            updates = _create_masks_from_multi_index(
                cast("Sequence[MultiIndexUpdate]", updated_pixels), self.tracks.ndim
            )
        else:
            raise ValueError(
                "updated_pixels entries must all use the same form, either "
                "(mask, time, old value) triples or (multi-index, old value) pairs, "
                f"but entry lengths were {sorted(lengths)}."
            )

        if new_value != 0 and updates:
            # name the node that was painted with
            node_to_select = new_value
            times = {time for _, time, _ in updates}
            assert len(times) == 1, "Can only update one time point at a time"
            time = int(times.pop())
            all_mask = union_td_masks([mask for mask, _, _ in updates])
            if self.tracks.graph_full.has_node(new_value):
                # An id that already names a node must take the update path: you cannot
                # create a *new* node on a taken id. Ids are globally unique across
                # graph_full (full ⊇ solution, never reused), so graph_full is the
                # correct "does this id name a node?" check; using graph_solution here
                # would misroute a soft-deleted id to the add path and silently revive
                # the old node with stale attributes.
                # NOTE: reviving a soft-deleted (solution=False) node by painting it
                # back is not supported yet — UpdateNodeSeg reads the solution view and
                # would fail. Guard explicitly until revive-by-paint is implemented.
                if not self.tracks.graph_solution.has_node(new_value):
                    raise NotImplementedError(
                        f"Cannot paint onto node {new_value}: it is soft-deleted (not "
                        "in the solution). Revive-by-paint is not supported yet."
                    )
                self.actions.append(
                    UpdateNodeSeg(tracks, new_value, all_mask, added=True)
                )
            else:
                time_key = tracks.features.time_key
                tracklet_key = tracks.features.tracklet_key
                attrs: dict[str, int] = {
                    time_key: time,
                    tracklet_key: current_track_id,
                }
                self.actions.append(
                    UserAddNode(
                        tracks,
                        new_value,
                        attributes=attrs,
                        mask=all_mask,
                        force=force,
                        _top_level=False,
                    )
                )

        # Now that the InvalidAction check for adding a new node has passed, we can add
        # actions for updating/deleting existing nodes
        for mask, _time, old_value in updates:
            if old_value == 0:
                continue
            # check if all pixels of old_value are removed
            mask_old_value = self.tracks.graph_full.nodes[old_value]["mask"]
            # If pixels fully overlaps with old_value mask, delete node
            if mask.intersection(mask_old_value) == mask_old_value.mask.sum():
                self.actions.append(UserDeleteNode(tracks, old_value, _top_level=False))
            else:
                self.actions.append(UpdateNodeSeg(tracks, old_value, mask, added=False))
        self.node_to_select = node_to_select
        if _top_level:
            self.tracks.action_history.add_new_action(self)
            self.tracks.refresh.emit(node_to_select)
