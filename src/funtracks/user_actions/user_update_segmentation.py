from __future__ import annotations

import numpy as np

from ..actions._base import ActionGroup
from ..actions.add_delete_node import AddNode, DeleteNode
from ..actions.update_segmentation import UpdateNodeSeg


class UserUpdateSegmentation(ActionGroup):
    def __init__(
        self,
        project,
        new_value: int,
        updated_pixels: list[tuple[tuple[np.ndarray, ...], int]],
    ):
        """tuple(int, list[tuple]): The new value, and a list of node update actions
                defined by the time point and node update item
                Each "action" is a 2-tuple containing:
        Args:
            new_value (int): The new value that the user painted with
            updated_pixels (list[tuple[tuple[np.ndarray, ...], int]]): A list of node
                update actions, consisting of a numpy multi-index, pointing to the array
                elements that were changed (a tuple with len ndims), and the value
                before the change
        """
        super().__init__(project, actions=[])
        for pixels, old_value in updated_pixels:
            ndim = len(pixels)
            if old_value == 0:
                continue
            time = pixels[0][0]
            # check if all pixels of old_value are removed
            # TODO: this assumes the segmentation is already updated, but then we can't
            # recover the pixels, so we have to pass them here for undo purposes
            if np.sum(self.project.segmentation[time] == old_value) == 0:
                self.actions.append(DeleteNode(project, old_value, pixels=pixels))
            else:
                self.actions.append(
                    UpdateNodeSeg(project, old_value, pixels, added=False)
                )
        if new_value != 0:
            all_pixels = tuple(
                np.concatenate([pixels[dim] for pixels, _ in updated_pixels])
                for dim in range(ndim)
            )
            assert len(np.unique(all_pixels[0])) == 1, (
                "Can only update one time point at a time"
            )
            if self.project.cand_graph.has_node(new_value):
                self.actions.append(
                    UpdateNodeSeg(project, new_value, all_pixels, added=True)
                )
            else:
                # TODO: Get default attributes for non-computed attributes
                self.actions.append(
                    AddNode(project, new_value, attributes={}, pixels=all_pixels)
                )
