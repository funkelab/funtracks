from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import (
    TYPE_CHECKING,
    Any,
    TypeAlias,
)
from warnings import warn

import networkx as nx
import numpy as np
from psygnal import Signal
from skimage import measure

from ..features.edge_features import IoU
from ..features.node_features import Area, Position, Time, TrackID
from .compute_ious import _compute_ious
from .graph_attributes import EdgeAttr, NodeAttr

if TYPE_CHECKING:
    from pathlib import Path

    from ..features._base import Feature

AttrValue: TypeAlias = Any
Node: TypeAlias = int
Edge: TypeAlias = tuple[Node, Node]
AttrValues: TypeAlias = list[AttrValue]
Attrs: TypeAlias = dict[str, AttrValues]
SegMask: TypeAlias = tuple[np.ndarray, ...]

logger = logging.getLogger(__name__)


def required_features(solution: bool, ndim: int) -> list[Feature]:
    axes = ("z", "y", "x") if ndim == 4 else ("y", "x")
    feats = [Time(), Position(axes=axes)]
    if solution:
        feats.append(TrackID())
    return feats


def optional_features(ndim: int, seg: bool) -> list[Feature]:
    feats: list[Feature] = []
    if seg:
        feats.extend([Area(ndim=ndim), IoU()])
    return feats


class Tracks:
    """A set of tracks consisting of a graph and an optional segmentation.
    The graph nodes represent detections and must have a time attribute and
    position attribute. Edges in the graph represent links across time.

    Attributes:
        graph (nx.DiGraph): A graph with nodes representing detections and
            and edges representing links across time.
        segmentation (Optional(np.ndarray)): An optional segmentation that
            accompanies the tracking graph. If a segmentation is provided,
            the node ids in the graph must match the segmentation labels.
            Defaults to None.
        time_attr (str): The attribute in the graph that specifies the time
            frame each node is in.
        pos_attr (str | tuple[str] | list[str]): The attribute in the graph
            that specifies the position of each node. Can be a single attribute
            that holds a list, or a list of attribute keys.

    For bulk operations on attributes, a KeyError will be raised if a node or edge
    in the input set is not in the graph. All operations before the error node will
    be performed, and those after will not.
    """

    refresh = Signal(object)

    def __init__(
        self,
        graph: nx.DiGraph,
        segmentation: np.ndarray | None = None,
        time_attr: str = NodeAttr.TIME.value,
        pos_attr: str | tuple[str] | list[str] = NodeAttr.POS.value,
        scale: list[float] | None = None,
        ndim: int | None = None,
        features: list[Feature] | None = None,
    ):
        self.graph = graph
        self.segmentation = segmentation
        self.time_attr = time_attr
        self.pos_attr = pos_attr
        self.scale = scale
        self.ndim = self._compute_ndim(segmentation, scale, ndim)
        required_feats = required_features(solution=False, ndim=self.ndim)
        self.features = (
            list(set(required_feats).union(features))
            if features is not None
            else required_feats
        )

    def _compute_node_attrs(self, nodes: Iterable[Node], times: Iterable[int]) -> Attrs:
        """Get the segmentation controlled node attributes (area and position)
        from the segmentation with label based on the node id in the given time point.

        Args:
            nodes (Iterable[int]): The node ids to query the current segmentation for
            time (int): The time frames of the current segmentation to query

        Returns:
            dict[str, int]: A dictionary containing the attributes that could be
                determined from the segmentation. It will be empty if self.segmentation
                is None. If self.segmentation exists but node id is not present in time,
                area will be 0 and position will be None. If self.segmentation
                exists and node id is present in time, area and position will be included.
        """
        if self.segmentation is None:
            return {}

        attrs: dict[str, list[Any]] = {
            NodeAttr.POS.value: [],
            NodeAttr.AREA.value: [],
        }
        for node, time in zip(nodes, times, strict=False):
            seg = self.segmentation[time] == node
            pos_scale = self.scale[1:] if self.scale is not None else None
            area = np.sum(seg)
            if pos_scale is not None:
                area *= np.prod(pos_scale)
            # only include the position if the segmentation was actually there
            pos = (
                measure.centroid(seg, spacing=pos_scale)
                if area > 0
                else np.array(
                    [
                        None,
                    ]
                    * (self.ndim - 1)
                )
            )
            attrs[NodeAttr.AREA.value].append(area)
            attrs[NodeAttr.POS.value].append(pos)
        return attrs

    def _compute_edge_attrs(self, edges: Iterable[Edge]) -> Attrs:
        """Get the segmentation controlled edge attributes (IOU)
        from the segmentations associated with the endpoints of the edge.
        The endpoints should already exist and have associated segmentations.

        Args:
            edge (Edge): The edge to compute the segmentation-based attributes from

        Returns:
            dict[str, int]: A dictionary containing the attributes that could be
                determined from the segmentation. It will be empty if self.segmentation
                is None or if self.segmentation exists but the endpoint segmentations
                are not found.
        """
        if self.segmentation is None:
            return {}

        attrs: dict[str, list[Any]] = {EdgeAttr.IOU.value: []}
        for edge in edges:
            source, target = edge
            source_time = self.get_time(source)
            target_time = self.get_time(target)

            source_arr = self.segmentation[source_time] == source
            target_arr = self.segmentation[target_time] == target

            iou_list = _compute_ious(source_arr, target_arr)  # list of (id1, id2, iou)
            iou = 0 if len(iou_list) == 0 else iou_list[0][2]

            attrs[EdgeAttr.IOU.value].append(iou)
        return attrs

    def save(self, directory: Path):
        """Save the tracks to the given directory.
        Currently, saves the graph as a json file in networkx node link data format,
        saves the segmentation as a numpy npz file, and saves the time and position
        attributes and scale information in an attributes json file.
        Args:
            directory (Path): The directory to save the tracks in.
        """
        warn(
            "`Tracks.save` is deprecated and will be removed in 2.0, use "
            "`funtracks.import_export.internal_format.save` instead",
            DeprecationWarning,
            stacklevel=2,
        )
        from ..import_export.internal_format import save_tracks

        save_tracks(self, directory)

    @classmethod
    def load(cls, directory: Path, seg_required=False) -> Tracks:
        """Load a Tracks object from the given directory. Looks for files
        in the format generated by Tracks.save.
        Args:
            directory (Path): The directory containing tracks to load
            seg_required (bool, optional): If true, raises a FileNotFoundError if the
                segmentation file is not present in the directory. Defaults to False.
        Returns:
            Tracks: A tracks object loaded from the given directory
        """
        warn(
            "`Tracks.load` is deprecated and will be removed in 2.0, use "
            "`funtracks.import_export.internal_format.load` instead",
            DeprecationWarning,
            stacklevel=2,
        )
        from ..import_export.internal_format import load_tracks

        return load_tracks(directory, seg_required=seg_required, solution=False)

    @classmethod
    def delete(cls, directory: Path):
        """Delete the tracks in the given directory. Also deletes the directory.

        Args:
            directory (Path): Directory containing tracks to be deleted
        """
        warn(
            "`Tracks.delete` is deprecated and will be removed in 2.0, use "
            "`funtracks.import_export.internal_format.delete` instead",
            DeprecationWarning,
            stacklevel=2,
        )
        from ..import_export.internal_format import delete_tracks

        delete_tracks(directory)
