from __future__ import annotations

import logging
import warnings
from collections.abc import Iterable, Sequence
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

from funtracks.features import Feature, FeatureSet, FeatureType, Position, Time

from .compute_ious import _compute_ious
from .graph_attributes import EdgeAttr, NodeAttr

if TYPE_CHECKING:
    from pathlib import Path

AttrValue: TypeAlias = Any
Node: TypeAlias = int
Edge: TypeAlias = tuple[Node, Node]
AttrValues: TypeAlias = list[AttrValue]
Attrs: TypeAlias = dict[str, AttrValues]
SegMask: TypeAlias = tuple[np.ndarray, ...]

logger = logging.getLogger(__name__)


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
        time_attr: str | None = NodeAttr.TIME.value,
        pos_attr: str | tuple[str] | list[str] | None = NodeAttr.POS.value,
        scale: list[float] | None = None,
        ndim: int | None = None,
        features: FeatureSet | None = None,
    ):
        if features is not None and (time_attr is not None or pos_attr is not None):
            warn(
                "Provided both FeatureSet and pos_attr or time_attr: ignoring attr "
                "arguments.",
                stacklevel=2,
            )
        self.graph = graph
        self.segmentation = segmentation
        self._time_attr = time_attr
        self._pos_attr = pos_attr
        self.scale = scale
        self.ndim = self._compute_ndim(segmentation, scale, ndim)
        self.features = (
            self._get_feature_set(time_attr, pos_attr) if features is None else features
        )

    @property
    def time_attr(self):
        warn(
            "Deprecating Tracks.time_attr in favor of tracks.features.time."
            " Will be removed in funtracks v2.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._time_attr

    @property
    def pos_attr(self):
        warn(
            "Deprecating Tracks.pos_attr in favor of tracks.features.position."
            " Will be removed in funtracks v2.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._pos_attr

    def _get_feature_set(
        self, time_attr: str | None, pos_attr: str | tuple[str] | list[str] | None
    ):
        time = Time(key=time_attr)
        recompute = self.segmentation is not None
        if isinstance(pos_attr, tuple | list):
            pos_feature = [
                Feature(
                    key=attr,
                    feature_type=FeatureType.NODE,
                    value_type=float,
                    required=True,
                    recompute=recompute,
                )
                for attr in pos_attr
            ]
        else:
            axis_names = ["z", "y", "x"] if self.ndim == 4 else ["y", "x"]
            pos_feature = Position(key=pos_attr, axes=axis_names, recompute=recompute)

        # TODO: use RegionpropsAnnotator to add area feature and others to the feature set
        extra_features = []
        if self.segmentation is not None:
            area_name = "Area" if self.ndim == 3 else "Volume"
            extra_features.append(
                Feature(
                    key=NodeAttr.AREA.value,
                    display_name=area_name,
                    feature_type=FeatureType.NODE,
                    value_type=float,
                    recompute=True,
                )
            )
        return FeatureSet(time, pos_feature, extra_features)

    def nodes(self):
        return np.array(self.graph.nodes())

    def edges(self):
        return np.array(self.graph.edges())

    def in_degree(self, nodes: np.ndarray | None = None) -> np.ndarray:
        if nodes is not None:
            return np.array([self.graph.in_degree(node.item()) for node in nodes])
        else:
            return np.array(self.graph.in_degree())

    def out_degree(self, nodes: np.ndarray | None = None) -> np.ndarray:
        if nodes is not None:
            return np.array([self.graph.out_degree(node.item()) for node in nodes])
        else:
            return np.array(self.graph.out_degree())

    def predecessors(self, node: int) -> list[int]:
        return list(self.graph.predecessors(node))

    def successors(self, node: int) -> list[int]:
        return list(self.graph.successors(node))

    def get_positions(self, nodes: Iterable[Node], incl_time: bool = False) -> np.ndarray:
        """Get the positions of nodes in the graph. Optionally include the
        time frame as the first dimension. Raises an error if any of the nodes
        are not in the graph.

        Args:
            node (Iterable[Node]): The node ids in the graph to get the positions of
            incl_time (bool, optional): If true, include the time as the
                first element of each position array. Defaults to False.

        Returns:
            np.ndarray: A N x ndim numpy array holding the positions, where N is the
                number of nodes passed in
        """
        if isinstance(self.features.position, list):
            positions = np.stack(
                [
                    self.get_nodes_attr(nodes, feat.key, required=True)
                    for feat in self.features.position
                ],
                axis=1,
            )
        else:
            positions = np.array(
                self.get_nodes_attr(nodes, self.features.position.key, required=True)
            )

        if incl_time:
            times = np.array(
                self.get_nodes_attr(nodes, self.features.time.key, required=True)
            )
            positions = np.c_[times, positions]

        return positions

    def get_position(self, node: Node, incl_time=False) -> list:
        return self.get_positions([node], incl_time=incl_time)[0].tolist()

    def set_positions(
        self,
        nodes: Iterable[Node],
        positions: np.ndarray,
        incl_time: bool = False,
    ):
        """Set the location of nodes in the graph. Optionally include the
        time frame as the first dimension. Raises an error if any of the nodes
        are not in the graph.

        Args:
            nodes (Iterable[node]): The node ids in the graph to set the location of.
            positions (np.ndarray): An (ndim, num_nodes) shape array of positions to set.
            f incl_time is true, time is the first column and is included in ndim.
            incl_time (bool, optional): If true, include the time as the
                first column of the position array. Defaults to False.
        """
        if not isinstance(positions, np.ndarray):
            positions = np.array(positions)
        if incl_time:
            times = positions[:, 0].tolist()  # we know this is a list of ints
            self.set_times(nodes, times)  # type: ignore
            positions = positions[:, 1:]

        if isinstance(self.features.position, list):
            for idx, feat in enumerate(self.features.position):
                self._set_nodes_attr(nodes, feat.key, positions[:, idx].tolist())
        else:
            self._set_nodes_attr(nodes, self.features.position.key, positions.tolist())

    def set_position(self, node: Node, position: list, incl_time=False):
        self.set_positions(
            [node], np.expand_dims(np.array(position), axis=0), incl_time=incl_time
        )

    def get_times(self, nodes: Iterable[Node]) -> Sequence[int]:
        return self.get_nodes_attr(nodes, self.features.time.key, required=True)

    def get_time(self, node: Node) -> int:
        """Get the time frame of a given node. Raises an error if the node
        is not in the graph.

        Args:
            node (Any): The node id to get the time frame for

        Returns:
            int: The time frame that the node is in
        """
        return int(self.get_times([node])[0])

    def set_times(self, nodes: Iterable[Node], times: Iterable[int]):
        times = [int(t) for t in times]
        self._set_nodes_attr(nodes, self.features.time.key, times)

    def set_time(self, node: Any, time: int):
        """Set the time frame of a given node. Raises an error if the node
        is not in the graph.

        Args:
            node (Any): The node id to set the time frame for
            time (int): The time to set

        """
        self.set_times([node], [int(time)])

    def get_areas(self, nodes: Iterable[Node]) -> Sequence[int | None]:
        """Get the area/volume of a given node. Raises a KeyError if the node
        is not in the graph. Returns None if the given node does not have an Area
        attribute.

        Args:
            node (Node): The node id to get the area/volume for

        Returns:
            int: The area/volume of the node
        """
        return self.get_nodes_attr(nodes, NodeAttr.AREA.value)

    def get_area(self, node: Node) -> int | None:
        """Get the area/volume of a given node. Raises a KeyError if the node
        is not in the graph. Returns None if the given node does not have an Area
        attribute.

        Args:
            node (Node): The node id to get the area/volume for

        Returns:
            int: The area/volume of the node
        """
        return self.get_areas([node])[0]

    def get_ious(self, edges: Iterable[Edge]):
        return self.get_edges_attr(edges, EdgeAttr.IOU.value)

    def get_iou(self, edge: Edge):
        return self.get_edge_attr(edge, EdgeAttr.IOU.value)

    def get_pixels(self, nodes: Iterable[Node]) -> list[tuple[np.ndarray, ...]] | None:
        """Get the pixels corresponding to each node in the nodes list.

        Args:
            nodes (list[Node]): A list of node to get the values for.

        Returns:
            list[tuple[np.ndarray, ...]] | None: A list of tuples, where each tuple
            represents the pixels for one of the input nodes, or None if the segmentation
            is None. The tuple will have length equal to the number of segmentation
            dimensions, and can be used to index the segmentation.
        """
        if self.segmentation is None:
            return None
        pix_list = []
        for node in nodes:
            time = self.get_time(node)
            loc_pixels = np.nonzero(self.segmentation[time] == node)
            time_array = np.ones_like(loc_pixels[0]) * time
            pix_list.append((time_array, *loc_pixels))
        return pix_list

    def set_pixels(
        self, pixels: Iterable[tuple[np.ndarray, ...]], values: Iterable[int | None]
    ):
        """Set the given pixels in the segmentation to the given value.

        Args:
            pixels (Iterable[tuple[np.ndarray]]): The pixels that should be set,
                formatted like the output of np.nonzero (each element of the tuple
                represents one dimension, containing an array of indices in that
                dimension). Can be used to directly index the segmentation.
            value (Iterable[int | None]): The value to set each pixel to
        """
        if self.segmentation is None:
            raise ValueError("Cannot set pixels when segmentation is None")
        for pix, val in zip(pixels, values, strict=False):
            if val is None:
                raise ValueError("Cannot set pixels to None value")
            self.segmentation[pix] = val

    def _set_node_attributes(self, nodes: Iterable[Node], attributes: Attrs):
        """Update the attributes for given nodes"""

        for idx, node in enumerate(nodes):
            if node in self.graph:
                for key, values in attributes.items():
                    self.graph.nodes[node][key] = values[idx]
            else:
                logger.info("Node %d not found in the graph.", node)

    def _set_edge_attributes(self, edges: Iterable[Edge], attributes: Attrs) -> None:
        """Set the edge attributes for the given edges. Attributes should already exist
        (although adding will work in current implementation, they cannot currently be
        removed)

        Args:
            edges (list[Edge]): A list of edges to set the attributes for
            attributes (Attributes): A dictionary of attribute name -> numpy array,
                where the length of the arrays matches the number of edges.
                Attributes should already exist: this function will only
                update the values.
        """
        for idx, edge in enumerate(edges):
            if self.graph.has_edge(*edge):
                for key, value in attributes.items():
                    self.graph.edges[edge][key] = value[idx]
            else:
                logger.info("Edge %d not found in the graph.", edge)

    def _compute_ndim(
        self,
        seg: np.ndarray | None,
        scale: list[float] | None,
        provided_ndim: int | None,
    ):
        seg_ndim = seg.ndim if seg is not None else None
        scale_ndim = len(scale) if scale is not None else None
        ndims = [seg_ndim, scale_ndim, provided_ndim]
        ndims = [d for d in ndims if d is not None]
        if len(ndims) == 0:
            raise ValueError(
                "Cannot compute dimensions from segmentation or scale: please provide "
                "ndim argument"
            )
        ndim = ndims[0]
        if not all(d == ndim for d in ndims):
            raise ValueError(
                f"Dimensions from segmentation {seg_ndim}, scale {scale_ndim}, and ndim "
                f"{provided_ndim} must match"
            )
        return ndim

    def _set_node_attr(self, node: Node, attr: str, value: Any):
        if isinstance(value, np.ndarray):
            value = list(value)
        self.graph.nodes[node][attr] = value

    def _set_nodes_attr(self, nodes: Iterable[Node], attr: str, values: Iterable[Any]):
        for node, value in zip(nodes, values, strict=False):
            if isinstance(value, np.ndarray):
                value = list(value)
            self.graph.nodes[node][attr] = value

    def get_node_attr(self, node: Node, attr: str, required: bool = False):
        if required:
            return self.graph.nodes[node][attr]
        else:
            return self.graph.nodes[node].get(attr, None)

    def _get_node_attr(self, node, attr, required=False):
        warnings.warn(
            "_get_node_attr deprecated in favor of public method get_node_attr",
            DeprecationWarning,
            stacklevel=2,
        )
        self.get_node_attr(node, attr, required=required)

    def get_nodes_attr(self, nodes: Iterable[Node], attr: str, required: bool = False):
        return [self.get_node_attr(node, attr, required=required) for node in nodes]

    def _get_nodes_attr(self, nodes, attr, required=False):
        warnings.warn(
            "_get_nodes_attr deprecated in favor of public method get_nodes_attr",
            DeprecationWarning,
            stacklevel=2,
        )
        self.get_nodes_attr(nodes, attr, required=required)

    def _set_edge_attr(self, edge: Edge, attr: str, value: Any):
        self.graph.edge[edge][attr] = value

    def _set_edges_attr(self, edges: Iterable[Edge], attr: str, values: Iterable[Any]):
        for edge, value in zip(edges, values, strict=False):
            self.graph.edges[edge][attr] = value

    def get_edge_attr(self, edge: Edge, attr: str, required: bool = False):
        if required:
            return self.graph.edges[edge][attr]
        else:
            return self.graph.edges[edge].get(attr, None)

    def get_edges_attr(self, edges: Iterable[Edge], attr: str, required: bool = False):
        return [self.get_edge_attr(edge, attr, required=required) for edge in edges]

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
                measure.centroid(seg, spacing=pos_scale)  # type: ignore
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
