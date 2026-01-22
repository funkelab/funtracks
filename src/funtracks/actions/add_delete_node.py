from __future__ import annotations

from typing import TYPE_CHECKING

from ._base import BasicAction

if TYPE_CHECKING:
    from typing import Any

    from funtracks.data_model.solution_tracks import SolutionTracks
    from funtracks.data_model.tracks import Node, SegMask

import numpy as np
import tracksdata as td
from tracksdata.nodes._mask import Mask

from funtracks.utils.tracksdata_utils import (
    pixels_to_td_mask,
)


class AddNode(BasicAction):
    """Action for adding new nodes. If a segmentation should also be added, the
    pixels for each node should be provided. The label to set the pixels will
    be taken from the node id. The existing pixel values are assumed to be
    zero - you must explicitly update any other segmentations that were overwritten
    using an UpdateNodes action if you want to be able to undo the action.
    """

    def __init__(
        self,
        tracks: SolutionTracks,
        node: Node,
        attributes: dict[str, Any],
        pixels: SegMask | None = None,
    ):
        """Create an action to add a new node, with optional segmentation

        Args:
            tracks (Tracks): The Tracks to add the node to
            node (Node): A node id
            attributes (Attrs): Includes times, track_ids, and optionally positions
            pixels (SegMask | None, optional): The segmentation associated with
                the node. Defaults to None.
        Raises:
            ValueError: If time attribute is not in attributes.
            ValueError: If track_id is not in attributes.
            ValueError: If pixels is None and position is not in attributes.
        """
        super().__init__(tracks)
        self.tracks: SolutionTracks  # Narrow type from base class
        self.node = node

        # Get keys from tracks features
        time_key = tracks.features.time_key
        track_id_key = tracks.features.tracklet_key
        pos_key = tracks.features.position_key

        # validate the input
        if time_key not in attributes:
            raise ValueError(f"Must provide a time attribute for node {node}")
        if track_id_key not in attributes:
            raise ValueError(f"Must provide a {track_id_key} attribute for node {node}")

        # Check for position - handle both single key and list of keys
        if pixels is None:
            if isinstance(pos_key, list):
                # Multi-axis position keys
                if not all(key in attributes for key in pos_key):
                    raise ValueError(
                        f"Must provide position or segmentation for node {node}"
                    )
            else:
                # Single position key
                if pos_key not in attributes:
                    raise ValueError(
                        f"Must provide position or segmentation for node {node}"
                    )
        self.pixels = pixels
        self.attributes = attributes
        self._apply()

    def inverse(self) -> BasicAction:
        """Invert the action to delete nodes instead"""
        return DeleteNode(self.tracks, self.node)

    def _apply(self) -> None:
        """Apply the action, and set segmentation if provided in self.pixels"""
        attrs = self.attributes

        if self.tracks.segmentation is not None:
            if self.pixels is not None:
                mask_obj, _ = pixels_to_td_mask(
                    self.pixels, self.tracks.ndim, self.tracks.scale
                )
                attrs[td.DEFAULT_ATTR_KEYS.MASK] = mask_obj
                attrs[td.DEFAULT_ATTR_KEYS.BBOX] = mask_obj.bbox
            else:
                # TODO Teun: remove this defaulting behavior, see new tracksdata PR
                if len(self.tracks.segmentation.shape) == 3:
                    attrs[td.DEFAULT_ATTR_KEYS.MASK] = Mask(
                        np.array([[False]]), bbox=[0, 0, 1, 1]
                    )
                    attrs[td.DEFAULT_ATTR_KEYS.BBOX] = [0, 0, 1, 1]
                elif len(self.tracks.segmentation.shape) == 4:
                    attrs[td.DEFAULT_ATTR_KEYS.MASK] = Mask(
                        np.array([[[False]]]), bbox=[0, 0, 0, 1, 1, 1]
                    )
                    attrs[td.DEFAULT_ATTR_KEYS.BBOX] = [0, 0, 0, 1, 1, 1]
                else:
                    raise ValueError(
                        "Must provide pixels or mask when adding node to tracks with seg"
                    )

        # Position is already set in attrs above
        # Add nodes to td graph
        required_attrs = self.tracks.graph.node_attr_keys().copy()
        if td.DEFAULT_ATTR_KEYS.NODE_ID in required_attrs:
            required_attrs.remove(td.DEFAULT_ATTR_KEYS.NODE_ID)
        if td.DEFAULT_ATTR_KEYS.SOLUTION not in attrs:
            attrs[td.DEFAULT_ATTR_KEYS.SOLUTION] = 1
        for attr in required_attrs:
            if attr not in attrs:
                # TODO Teun: remove this logic when td has default values (PR)
                attrs[attr] = self.tracks.features[attr]["default_value"]

        node_dict = {
            attr: np.array(values) if attr == "pos" else values
            for attr, values in attrs.items()
        }

        self.tracks.graph.add_node(attrs=node_dict, index=self.node)

        if self.pixels is not None:
            self.tracks.set_pixels(self.pixels, self.node)

        # Always notify annotators - they will check their own preconditions
        self.tracks.notify_annotators(self)


class DeleteNode(BasicAction):
    """Action of deleting existing nodes
    If the tracks contain a segmentation, this action also constructs a reversible
    operation for setting involved pixels to zero
    """

    def __init__(
        self,
        tracks: SolutionTracks,
        node: Node,
        pixels: SegMask | None = None,
    ):
        super().__init__(tracks)
        self.tracks: SolutionTracks  # Narrow type from base class
        self.node = node

        # Save all node feature values from the features dict
        self.attributes = {}
        for key in self.tracks.features.node_features:
            val = self.tracks.get_node_attr(node, key)
            if val is not None:
                self.attributes[key] = val

        if td.DEFAULT_ATTR_KEYS.MASK in self.tracks.graph.node_attr_keys():
            self.attributes[td.DEFAULT_ATTR_KEYS.MASK] = self.tracks.get_nodes_attr(
                [self.node], td.DEFAULT_ATTR_KEYS.MASK
            )[0]
            self.attributes[td.DEFAULT_ATTR_KEYS.BBOX] = self.tracks.get_nodes_attr(
                [self.node], td.DEFAULT_ATTR_KEYS.BBOX
            )[0]
        self.attributes[td.DEFAULT_ATTR_KEYS.SOLUTION] = self.tracks.get_nodes_attr(
            [self.node], td.DEFAULT_ATTR_KEYS.SOLUTION
        )[0]

        self.pixels = self.tracks.get_pixels(node) if pixels is None else pixels
        self._apply()

    def inverse(self) -> BasicAction:
        """Invert this action, and provide inverse segmentation operation if given"""

        return AddNode(self.tracks, self.node, self.attributes, pixels=self.pixels)

    def _apply(self) -> None:
        """ASSUMES THERE ARE NO INCIDENT EDGES - raises valueerror if an edge will be
        removed by this operation
        Steps:
        - For each node
            set pixels to 0 if self.pixels is provided
        - Remove nodes from graph
        """

        self.tracks.graph.remove_node(self.node)
        self.tracks.notify_annotators(self)
