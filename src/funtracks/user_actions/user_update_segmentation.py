from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from funtracks.data_model import NodeAttr

from ..actions._base import ActionGroup
from ..actions.update_segmentation import UpdateNodeSeg
from .user_add_node import UserAddNode
from .user_delete_node import UserDeleteNode

if TYPE_CHECKING:
    from funtracks.data_model import SolutionTracks


class UserUpdateSegmentation(ActionGroup):
    def __init__(
        self,
        tracks: SolutionTracks,
        new_value: int,
        updated_pixels: list[tuple[tuple[np.ndarray, ...], int]],
    ):
        """Assumes that the pixels have already been updated in the project.segmentation
        NOTE: Re discussion with Kasia: we should have a basic action that updates the
        segmentation, and that is the only place the segmentation is updated. The basic
        add_node action doesn't have anything with pixels.

        Args:
            tracks (SolutiuonTracks): The solution tracks that the user is updating.
            new_value (int): The new value that the user painted with
            updated_pixels (list[tuple[tuple[np.ndarray, ...], int]]): A list of node
                update actions, consisting of a numpy multi-index, pointing to the array
                elements that were changed (a tuple with len ndims), and the value
                before the change
        """
        super().__init__(tracks, actions=[])
        if self.tracks.segmentation is None:
            raise ValueError("Cannot update non-existing segmentation.")
        for pixels, old_value in updated_pixels:
            ndim = len(pixels)
            if old_value == 0:
                continue
            time = pixels[0][0]
            # check if all pixels of old_value are removed
            # TODO: this assumes the segmentation is already updated, but then we can't
            # recover the pixels, so we have to pass them here for undo purposes
            if np.sum(self.tracks.segmentation[time] == old_value) == 0:
                self.actions.append(UserDeleteNode(tracks, old_value, pixels=pixels))
            else:
                self.actions.append(UpdateNodeSeg(tracks, old_value, pixels, added=False))
        if new_value != 0:
            all_pixels = tuple(
                np.concatenate([pixels[dim] for pixels, _ in updated_pixels])
                for dim in range(ndim)
            )
            assert len(np.unique(all_pixels[0])) == 1, (
                "Can only update one time point at a time"
            )
            time = all_pixels[0][0]
            if self.tracks.graph.has_node(new_value):
                self.actions.append(
                    UpdateNodeSeg(tracks, new_value, all_pixels, added=True)
                )
            else:
                attrs = {
                    NodeAttr.TIME.value: time,
                    # TODO: allow passing in the current track id, or just use UserAddNode
                    NodeAttr.TRACK_ID.value: tracks.get_next_track_id(),
                }
                self.actions.append(
                    UserAddNode(
                        tracks,
                        new_value,
                        attributes=attrs,
                        pixels=all_pixels,
                    )
                )
