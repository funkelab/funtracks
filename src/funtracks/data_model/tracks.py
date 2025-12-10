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

from funtracks.features import Feature, FeatureDict, Position, Time

if TYPE_CHECKING:
    from pathlib import Path

    from funtracks.actions import BasicAction
    from funtracks.annotators import AnnotatorRegistry, GraphAnnotator

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
        segmentation (np.ndarray | None): An optional segmentation that
            accompanies the tracking graph. If a segmentation is provided,
            the node ids in the graph must match the segmentation labels.
        features (FeatureDict): Dictionary of features tracked on graph nodes/edges.
        annotators (AnnotatorRegistry): List of annotators that compute features.
        scale (list[float] | None): How much to scale each dimension by, including time.
        ndim (int): Number of dimensions (3 for 2D+time, 4 for 3D+time).
    """

    refresh = Signal(object)

    def __init__(
        self,
        graph: nx.DiGraph,
        segmentation: np.ndarray | None = None,
        time_attr: str | None = None,
        pos_attr: str | tuple[str, ...] | list[str] | None = None,
        tracklet_attr: str | None = None,
        scale: list[float] | None = None,
        ndim: int | None = None,
        features: FeatureDict | None = None,
    ):
        """Initialize a Tracks object.

        Args:
            graph (nx.DiGraph): NetworkX directed graph with nodes as detections and
                edges as links.
            segmentation (np.ndarray | None): Optional segmentation array where labels
                match node IDs. Required for computing region properties (area, etc.).
            time_attr (str | None): Graph attribute name for time. Defaults to "time"
                if None.
            pos_attr (str | tuple[str, ...] | list[str] | None): Graph attribute
                name(s) for position. Can be:
                - Single string for one attribute containing position array
                - List/tuple of strings for multi-axis (one attribute per axis)
                Defaults to "pos" if None.
            tracklet_attr (str | None): Graph attribute name for tracklet/track IDs.
                Defaults to "track_id" if None.
            scale (list[float] | None): Scaling factors for each dimension (including
                time). If None, all dimensions scaled by 1.0.
            ndim (int | None): Number of dimensions (3 for 2D+time, 4 for 3D+time).
                If None, inferred from segmentation or scale.
            features (FeatureDict | None): Pre-built FeatureDict with feature
                definitions. If provided, time_attr/pos_attr/tracklet_attr are ignored.
                Assumes that all features in the dict already exist on the graph (will
                be activated but not recomputed). If None, core computed features (pos,
                area, track_id) are auto-detected by checking if they exist on the graph.
        """
        self.graph = graph
        self.segmentation = segmentation
        self.scale = scale
        self.ndim = self._compute_ndim(segmentation, scale, ndim)
        self.axis_names = ["z", "y", "x"] if self.ndim == 4 else ["y", "x"]

        # initialization steps:
        # 1. set up feature dict (or use provided)
        # 2. set up annotator registry
        # 3. activate existing features
        # 4. enable core features (compute them)

        # 1. set up feature dictionary for keeping track of features on graph
        if features is not None and (
            time_attr is not None or pos_attr is not None or tracklet_attr is not None
        ):
            warn(
                "Provided both FeatureDict and pos, time, or tracklet attr: ignoring attr"
                f" arguments ({pos_attr=}, {time_attr=}, {tracklet_attr=}).",
                stacklevel=2,
            )
        self.features = (
            self._get_feature_set(time_attr, pos_attr, tracklet_attr)
            if features is None
            else features
        )
        # 2. Set up annotator registry for managing feature computation
        self.annotators = self._get_annotators()

        # 3. Set up core computed features
        # If features FeatureDict was provided, activate those features in annotators
        if features is not None:
            self._activate_features_from_dict()
        else:
            self._setup_core_computed_features()

    @property
    def time_attr(self):
        warn(
            "Deprecating Tracks.time_attr in favor of tracks.features.time_key."
            " Will be removed in funtracks v2.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.features.time_key

    @property
    def pos_attr(self):
        warn(
            "Deprecating Tracks.pos_attr in favor of tracks.features.position_key."
            " Will be removed in funtracks v2.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.features.position_key

    def _get_feature_set(
        self,
        time_attr: str | None,
        pos_attr: str | tuple[str, ...] | list[str] | None,
        tracklet_key: str | None,
    ) -> FeatureDict:
        """Create a FeatureDict with static (user-provided) features only.

        Static features are those already present on the graph nodes (time, position
        when no segmentation). Managed features (computed from segmentation or graph
        structure) are added by annotators and registered later.

        Args:
            time_attr: Graph attribute name for time (e.g., "t", "time").
                If None, defaults to "time"
            pos_attr: Graph attribute name(s) for position. Can be:
                - Single string: one attribute containing position array (e.g., "pos")
                - List/tuple: multiple attributes, one per axis (e.g., ["y", "x"])
                - None: defaults to "pos"
            tracklet_key: Graph attribute name for tracklet/track IDs (e.g., "track_id").
                If None, defaults to "track_id"

        Returns:
            FeatureDict initialized with time feature and position if no segmentation
        """
        # Use defaults if not provided
        time_key = time_attr if time_attr is not None else "time"
        if pos_attr is None:
            pos_attr = "pos"
        if tracklet_key is None:
            tracklet_key = "track_id"

        # Build static features dict - always include time
        features: dict[str, Feature] = {time_key: Time()}

        # Create FeatureDict with time feature
        # Position and tracklet features will be registered separately
        feature_dict = FeatureDict(
            features=features,
            time_key=time_key,
            position_key=None,
            tracklet_key=tracklet_key,
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
                        "required": True,
                        "default_value": None,
                    }
                # For multi-axis, set position_key directly
                # (not a single feature to register)
                feature_dict.position_key = multi_position_key
            else:
                # Single position attribute
                single_position_key = pos_attr
                pos_feature = Position(axes=self.axis_names)
                feature_dict.register_position_feature(single_position_key, pos_feature)

        return feature_dict

    def _get_annotators(self) -> AnnotatorRegistry:
        """Instantiate and return core annotators based on available data.

        Creates annotators conditionally:
        - RegionpropsAnnotator: Only if segmentation is provided
        - EdgeAnnotator: Only if segmentation is provided
        - TrackAnnotator: Only if this is a SolutionTracks instance

        Each annotator is configured with appropriate keys from self.features.

        Returns:
            AnnotatorRegistry containing all applicable annotators
        """
        # Import here to avoid circular dependency
        from funtracks.annotators import (
            AnnotatorRegistry,
            EdgeAnnotator,
            RegionpropsAnnotator,
            TrackAnnotator,
        )

        annotator_list: list[GraphAnnotator] = []

        # RegionpropsAnnotator: requires segmentation
        if RegionpropsAnnotator.can_annotate(self):
            # Pass position_key only if it's a single string (not multi-axis list)
            pos_key = (
                self.features.position_key
                if isinstance(self.features.position_key, str)
                else None
            )
            annotator_list.append(RegionpropsAnnotator(self, pos_key=pos_key))

        # EdgeAnnotator: requires segmentation
        if EdgeAnnotator.can_annotate(self):
            annotator_list.append(EdgeAnnotator(self))

        # TrackAnnotator: requires SolutionTracks (checked in can_annotate)
        if TrackAnnotator.can_annotate(self):
            annotator_list.append(
                TrackAnnotator(self, tracklet_key=self.features.tracklet_key)  # type: ignore
            )
        return AnnotatorRegistry(annotator_list)

    def _activate_features_from_dict(self) -> None:
        """Activate features that exist in both the FeatureDict and annotators.

        Used when a pre-built FeatureDict is provided to __init__. Activates features
        in annotators (sets computation flags) but does NOT compute them, assuming
        they already exist on the graph.
        """
        # Activate all features that exist in both FeatureDict and annotators
        for key in self.features:
            if key in self.annotators.all_features:
                self.annotators.activate_features([key])

    def _check_existing_feature(self, key: str) -> bool:
        """Detect if a key already exists on the graph by sampling the first node.

        Returns:
            bool: True if the key is on the first sampled node or there are no nodes,
                and False if missing from the first node.
        """
        if self.graph.number_of_nodes() == 0:
            return True

        # Get a sample node to check which attributes exist
        sample_node = next(iter(self.graph.nodes()))
        node_attrs = set(self.graph.nodes[sample_node].keys())
        return key in node_attrs

    def _setup_core_computed_features(self) -> None:
        """Sets up the core computed features (area, position, tracklet if applicable)

        Registers position/tracklet features from annotators into FeatureDict
        For each core feature:
        - Activates any features listed that are detected to exist (without computing)
        - Enables any features that don't exist (compute fresh)
        """
        # Import here to avoid circular dependency
        from funtracks.annotators import RegionpropsAnnotator, TrackAnnotator

        # Register core features from annotators in the features dict
        core_computed_features: list[str] = []
        for annotator in self.annotators:
            if isinstance(annotator, RegionpropsAnnotator):
                pos_key = annotator.pos_key
                self.features.position_key = pos_key
                core_computed_features.append(pos_key)
                # special case for backward compatibility
                core_computed_features.append("area")
            elif isinstance(annotator, TrackAnnotator):
                tracklet_key = annotator.tracklet_key
                self.features.tracklet_key = tracklet_key
                core_computed_features.append(tracklet_key)
        for key in core_computed_features:
            if self._check_existing_feature(key):
                # Add to FeatureDict if not already there
                if key not in self.features:
                    feature, _ = self.annotators.all_features[key]
                    self.features[key] = feature
                self.annotators.activate_features([key])
            else:
                # enable it (compute it)
                self.enable_features([key])

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

        .. deprecated:: 1.0
            `get_areas` will be removed in funtracks v2.0.
            Use `get_nodes_attr(nodes, "area")` instead.

        Args:
            node (Node): The node id to get the area/volume for

        Returns:
            int: The area/volume of the node
        """
        warnings.warn(
            "`get_areas` is deprecated and will be removed in funtracks v2.0. "
            "Use `get_nodes_attr(nodes, 'area')` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.get_nodes_attr(nodes, "area")

    def get_area(self, node: Node) -> int | None:
        """Get the area/volume of a given node. Raises a KeyError if the node
        is not in the graph. Returns None if the given node does not have an Area
        attribute.

        .. deprecated:: 1.0
            `get_area` will be removed in funtracks v2.0.
            Use `get_node_attr(node, "area")` instead.

        Args:
            node (Node): The node id to get the area/volume for

        Returns:
            int: The area/volume of the node
        """
        warnings.warn(
            "`get_area` is deprecated and will be removed in funtracks v2.0. "
            "Use `get_node_attr(node, 'area')` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.get_areas([node])[0]

    def get_ious(self, edges: Iterable[Edge]):
        """Get the IoU values for the given edges.

        .. deprecated:: 1.0
            `get_ious` will be removed in funtracks v2.0.
            Use `get_edges_attr(edges, "iou")` instead.

        Args:
            edges: An iterable of edges to get IoU values for.

        Returns:
            The IoU values for the edges.
        """
        warnings.warn(
            "`get_ious` is deprecated and will be removed in funtracks v2.0. "
            "Use `get_edges_attr(edges, 'iou')` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.get_edges_attr(edges, "iou")

    def get_iou(self, edge: Edge):
        """Get the IoU value for the given edge.

        .. deprecated:: 1.0
            `get_iou` will be removed in funtracks v2.0.
            Use `get_edge_attr(edge, "iou")` instead.

        Args:
            edge: An edge to get the IoU value for.

        Returns:
            The IoU value for the edge.
        """
        warnings.warn(
            "`get_iou` is deprecated and will be removed in funtracks v2.0. "
            "Use `get_edge_attr(edge, 'iou')` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.get_edge_attr(edge, "iou")

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

    def enable_features(self, feature_keys: list[str], recompute: bool = True) -> None:
        """Enable multiple features for computation efficiently.

        Adds features to annotators and FeatureDict, optionally computes their values.

        Args:
            feature_keys: List of feature keys to enable
            recompute: If True, compute feature values. If False, assume values
                already exist in graph and just register the feature.

        Raises:
            KeyError: If any feature is not available (raised by annotators)
        """
        # Registry validates and activates features (will raise if invalid)
        self.annotators.activate_features(feature_keys)

        # Add to FeatureDict
        for key in feature_keys:
            if key not in self.features:
                feature, _ = self.annotators.all_features[key]
                self.features[key] = feature

        # Compute the features if requested
        if recompute:
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
        self.annotators.deactivate_features(feature_keys)

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

    def _compute_node_attrs(self, node: Node, time: int) -> dict[str, Any]:
        """Get the segmentation controlled node attributes (area and position)
        from the segmentation with label based on the node id in the given time point.

        Args:
            node (int): The node id to query the current segmentation for
            time (int): The time frame of the current segmentation to query

        Returns:
            dict[str, int]: A dictionary containing the attributes that could be
                determined from the segmentation. It will be empty if self.segmentation
                is None. If self.segmentation exists but node id is not present in time,
                area will be 0 and position will be None. If self.segmentation
                exists and node id is present in time, area and position will be included.

        Deprecated:
            This method is deprecated and will be removed in funtracks v2.0.
            Use the annotator system (enable_features) to compute node attributes instead.
        """
        warn(
            "`_compute_node_attrs` is deprecated and will be removed in funtracks v2.0. "
            "Use the annotator system (enable_features) to compute node attributes "
            "instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self.segmentation is None:
            return {}

        attrs: dict[str, Any] = {}
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
        attrs["area"] = area
        attrs["pos"] = pos
        return attrs
