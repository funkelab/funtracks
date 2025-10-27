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

from funtracks.features import Feature, FeatureDict, Position, Time

from .graph_attributes import EdgeAttr, NodeAttr

if TYPE_CHECKING:
    from pathlib import Path

    from funtracks.actions import BasicAction

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
        scale (list[float] | None): How much to scale each dimension by, including time.

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
        pos_attr: str | tuple[str, ...] | list[str] | None = NodeAttr.POS.value,
        scale: list[float] | None = None,
        ndim: int | None = None,
        features: FeatureDict | None = None,
        existing_features: list[str] | None = None,
    ):
        if features is not None and (time_attr is not None or pos_attr is not None):
            warn(
                "Provided both FeatureDict and pos_attr or time_attr: ignoring attr "
                "arguments ({pos_attr=}, {time_attr=}).",
                stacklevel=2,
            )
        self.graph = graph
        self.segmentation = segmentation
        self._time_attr = time_attr
        self._pos_attr = pos_attr
        self.scale = scale
        self.ndim = self._compute_ndim(segmentation, scale, ndim)
        self.axis_names = ["z", "y", "x"] if self.ndim == 4 else ["y", "x"]
        self.features = (
            self._get_feature_set(time_attr, pos_attr) if features is None else features
        )

        # Initialize AnnotatorRegistry for managing feature computation
        # Import here to avoid circular dependency
        from funtracks.annotators import (
            AnnotatorRegistry,
            RegionpropsAnnotator,
            TrackAnnotator,
        )

        self.annotators = AnnotatorRegistry(self)

        # Register core features from annotators in the features dict
        for annotator in self.annotators.annotators:
            if isinstance(annotator, RegionpropsAnnotator):
                feature = annotator.all_features[annotator.pos_key][0]
                self.features.register_position_feature(annotator.pos_key, feature)
            elif isinstance(annotator, TrackAnnotator):
                feature = annotator.all_features[annotator.tracklet_key][0]
                self.features.register_tracklet_feature(annotator.tracklet_key, feature)

        if existing_features is not None:
            for key in existing_features:
                if key in self.annotators.all_features:
                    feature, _ = self.annotators.all_features[key]
                    # Add to FeatureDict if not already there
                    if key not in self.features:
                        self.features[key] = feature
                    self.annotators.enable_features([key])

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
        self, time_attr: str | None, pos_attr: str | tuple[str, ...] | list[str] | None
    ):
        """Create a FeatureDict with static features only.

        Static features are those provided by the user on the graph (time, position
        when segmentation is None). Managed features (computed from segmentation or
        graph structure) are added by annotators after initialization.

        Args:
            time_attr: The attribute name for time
            pos_attr: The attribute name(s) for position

        Returns:
            FeatureDict with static features
        """
        # Determine keys
        time_key = time_attr if time_attr is not None else "time"

        # Build static features dict - always include time
        features: dict[str, Feature] = {time_key: Time()}

        # Create FeatureDict with time feature
        # Position and tracklet features will be registered separately
        feature_dict = FeatureDict(
            features=features,
            time_key=time_key,
            position_key=None,
            tracklet_key=None,
        )

        # Register position feature if no segmentation (static position)
        if self.segmentation is None:
            # No segmentation - position is provided by user (static)
            if isinstance(pos_attr, tuple | list):
                # Multiple position attributes (one per axis)
                multi_position_key = list(pos_attr)
                for attr in pos_attr:
                    features[attr] = {
                        "feature_type": "node",
                        "value_type": "float",
                        "num_values": 1,
                        "display_name": None,
                        "required": True,
                        "default_value": None,
                    }
                # For multi-axis, set position_key directly
                # (not a single feature to register)
                feature_dict.position_key = multi_position_key
            else:
                # Single position attribute
                single_position_key = pos_attr if pos_attr is not None else "pos"
                pos_feature = Position(axes=self.axis_names)
                feature_dict.register_position_feature(single_position_key, pos_feature)

        return feature_dict

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
        if self.features.position_key is None:
            raise ValueError("position_key must be set")

        if isinstance(self.features.position_key, list):
            positions = np.stack(
                [
                    self.get_nodes_attr(nodes, key, required=True)
                    for key in self.features.position_key
                ],
                axis=1,
            )
        else:
            positions = np.array(
                self.get_nodes_attr(nodes, self.features.position_key, required=True)
            )

        if incl_time:
            times = np.array(
                self.get_nodes_attr(nodes, self.features.time_key, required=True)
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
        if self.features.position_key is None:
            raise ValueError("position_key must be set")

        if not isinstance(positions, np.ndarray):
            positions = np.array(positions)
        if incl_time:
            times = positions[:, 0].tolist()  # we know this is a list of ints
            self.set_times(nodes, times)  # type: ignore
            positions = positions[:, 1:]

        if isinstance(self.features.position_key, list):
            for idx, key in enumerate(self.features.position_key):
                self._set_nodes_attr(nodes, key, positions[:, idx].tolist())
        else:
            self._set_nodes_attr(nodes, self.features.position_key, positions.tolist())

    def set_position(
        self, node: Node, position: list | np.ndarray, incl_time: bool = False
    ):
        self.set_positions(
            [node], np.expand_dims(np.array(position), axis=0), incl_time=incl_time
        )

    def get_times(self, nodes: Iterable[Node]) -> Sequence[int]:
        return self.get_nodes_attr(nodes, self.features.time_key, required=True)

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
        self._set_nodes_attr(nodes, self.features.time_key, times)

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

    def get_pixels(self, node: Node) -> tuple[np.ndarray, ...] | None:
        """Get the pixels corresponding to each node in the nodes list.

        Args:
            node (Node): A  node to get the pixels for.

        Returns:
            tuple[np.ndarray, ...] | None: A tuple representing the pixels for the input
            node, or None if the segmentation is None. The tuple will have length equal
            to the number of segmentation dimensions, and can be used to index the
            segmentation.
        """
        if self.segmentation is None:
            return None
        time = self.get_time(node)
        loc_pixels = np.nonzero(self.segmentation[time] == node)
        time_array = np.ones_like(loc_pixels[0]) * time
        return (time_array, *loc_pixels)

    def set_pixels(self, pixels: tuple[np.ndarray, ...], value: int) -> None:
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
        self.segmentation[pixels] = value

    def _set_node_attributes(self, node: Node, attributes: dict[str, Any]) -> None:
        """Set the attributes for the given node

        Args:
            node (Node): The node to set the attributes for
            attributes (dict[str, Any]): A mapping from attribute name to value
        """
        if node in self.graph:
            for key, value in attributes.items():
                self.graph.nodes[node][key] = value
        else:
            logger.info("Node %d not found in the graph.", node)

    def _set_edge_attributes(self, edge: Edge, attributes: dict[str, Any]) -> None:
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
        if self.graph.has_edge(*edge):
            for key, value in attributes.items():
                self.graph.edges[edge][key] = value
        else:
            logger.info("Edge %s not found in the graph.", edge)

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
        return self.get_node_attr(node, attr, required=required)

    def get_nodes_attr(self, nodes: Iterable[Node], attr: str, required: bool = False):
        return [self.get_node_attr(node, attr, required=required) for node in nodes]

    def _get_nodes_attr(self, nodes, attr, required=False):
        warnings.warn(
            "_get_nodes_attr deprecated in favor of public method get_nodes_attr",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.get_nodes_attr(nodes, attr, required=required)

    def _set_edge_attr(self, edge: Edge, attr: str, value: Any):
        self.graph.edges[edge][attr] = value

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

    # ========== Feature Management ==========

    def notify_annotators(self, action: BasicAction) -> None:
        """Notify annotators about an action so they can recompute affected features.

        Delegates to the annotator registry which broadcasts to all annotators.
        The action contains all necessary information about which elements to update.

        Args:
            action: The action that triggered this notification
        """
        self.annotators.update(action)

    def get_available_features(self) -> dict[str, Feature]:
        """Get all features that can be computed across all annotators.

        Returns:
            Dictionary mapping feature keys to Feature definitions
        """
        return {k: feat for k, (feat, _) in self.annotators.all_features.items()}

    def get_active_features(self) -> dict[str, Feature]:
        """Get all currently active (included) features.

        Returns:
            Dictionary mapping feature keys to Feature definitions
        """
        return self.annotators.features

    def enable_features(self, feature_keys: list[str]) -> None:
        """Enable multiple features for computation efficiently.

        Adds features to annotators and FeatureDict, then computes their values.

        Args:
            feature_keys: List of feature keys to enable

        Raises:
            KeyError: If any feature is not available (raised by annotators)
        """
        # Registry validates and enables features (will raise if invalid)
        self.annotators.enable_features(feature_keys)

        # Add to FeatureDict
        for key in feature_keys:
            if key not in self.features:
                feature, _ = self.annotators.all_features[key]
                self.features[key] = feature

        # Compute the features
        self.annotators.compute(feature_keys)

    def disable_features(self, feature_keys: list[str]) -> None:
        """Disable multiple features from computation.

        Removes features from annotators and FeatureDict.

        Args:
            feature_keys: List of feature keys to disable

        Raises:
            KeyError: If any feature is not available (raised by annotators)
        """
        # Registry validates and disables features (will raise if invalid)
        self.annotators.disable_features(feature_keys)

        # Remove from FeatureDict
        for key in feature_keys:
            if key in self.features:
                del self.features[key]

    # ========== Persistence ==========

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
    def load(cls, directory: Path, seg_required=False, solution=False) -> Tracks:
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

        return load_tracks(directory, seg_required=seg_required, solution=solution)

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
