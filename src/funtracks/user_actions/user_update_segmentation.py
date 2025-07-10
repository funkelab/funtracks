from __future__ import annotations

import numpy as np

from ..actions._base import ActionGroup
from ..actions.set_feature_values import SetFeatureValues
from ..actions.update_segmentation import UpdateNodeSeg
from .user_add_node import UserAddNode
from .user_delete_node import UserDeleteNode


class UserUpdateSegmentation(ActionGroup):
    def __init__(
        self,
        project,
        new_value: int,
        updated_pixels: list[tuple[tuple[np.ndarray, ...], int]],
    ):
        """Assumes that the pixels have already been updated in the project.segmentation
        Args:
            new_value (int): The new value that the user painted with
            updated_pixels (list[tuple[tuple[np.ndarray, ...], int]]): A list of node
                update actions, consisting of a numpy multi-index, pointing to the array
                elements that were changed (a tuple with len ndims), and the value
                before the change
        """
        super().__init__(project, actions=[])
        pin_attrs = {self.project.graph.features.node_selection_pin: True}
        for pixels, old_value in updated_pixels:
            ndim = len(pixels)
            if old_value == 0:
                continue
            time = pixels[0][0]
            # check if all pixels of old_value are removed
            # TODO: this assumes the segmentation is already updated, but then we can't
            # recover the pixels, so we have to pass them here for undo purposes
            if np.sum(self.project.segmentation[time] == old_value) == 0:
                self.actions.append(UserDeleteNode(project, old_value, pixels=pixels))
            else:
                self.actions.append(
                    UpdateNodeSeg(project, old_value, pixels, added=False)
                )
                if new_value == 0:
                    # if the user intentionally made the seg smaller, assume it is a real
                    # node and pin it. If the user just drew over it with another value,
                    # don't pin it
                    self.actions.append(SetFeatureValues(project, old_value, pin_attrs))
        if new_value != 0:
            all_pixels = tuple(
                np.concatenate([pixels[dim] for pixels, _ in updated_pixels])
                for dim in range(ndim)
            )
            assert len(np.unique(all_pixels[0])) == 1, (
                "Can only update one time point at a time"
            )
            time = all_pixels[0][0]
            if self.project.graph.has_node(new_value):
                self.actions.append(
                    UpdateNodeSeg(project, new_value, all_pixels, added=True)
                )
                # pin the node that you edited the segmentation of to be selected
                self.actions.append(SetFeatureValues(project, new_value, pin_attrs))
            else:
                pin_attrs[self.project.graph.features.time] = time
                pin_attrs[self.project.graph.features.node_selected] = True
                self.actions.append(
                    UserAddNode(
                        project, new_value, attributes=pin_attrs, pixels=all_pixels
                    )
                )
