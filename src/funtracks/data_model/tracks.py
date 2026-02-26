from __future__ import annotations

import itertools
import logging
from collections.abc import Iterable, Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    TypeAlias,
)
from warnings import warn

import numpy as np
import tracksdata as td
from psygnal import Signal
from tracksdata.array import GraphArrayView
from tracksdata.nodes._mask import Mask

from funtracks.actions.action_history import ActionHistory
from funtracks.features import Feature, FeatureDict, Position, Time
from funtracks.utils.tracksdata_utils import (
    to_polars_dtype,
)

if TYPE_CHECKING:
    import tracksdata as td

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
        graph (td.graph.GraphView): A graph with nodes representing detections and
            and edges representing links across time.
        features (FeatureDict): Dictionary of features tracked on graph nodes/edges.
        annotators (AnnotatorRegistry): List of annotators that compute features.
        scale (list[float] | None): How much to scale each dimension by, including time.
        ndim (int): Number of dimensions (3 for 2D+time, 4 for 3D+time).
    """

    refresh = Signal(object)

    def __init__(
        self,
        graph: td.graph.GraphView,
        time_attr: str | None = None,
        pos_attr: str | tuple[str, ...] | list[str] | None = None,
        tracklet_attr: str | None = None,
        lineage_attr: str | None = None,
        scale: list[float] | None = None,
        ndim: int | None = None,
        features: FeatureDict | None = None,
        _segmentation: GraphArrayView | None = None,
    ):
        """Initialize a Tracks object.

        Args:
            graph (td.graph.GraphView): NetworkX directed graph with nodes as detections
                and edges as links.
            time_attr (str | None): Graph attribute name for time. Defaults to "time"
                if None.
            pos_attr (str | tuple[str, ...] | list[str] | None): Graph attribute
                name(s) for position. Can be:
                - Single string for one attribute containing position array
                - List/tuple of strings for multi-axis (one attribute per axis)
                Defaults to "pos" if None.
            tracklet_attr (str | None): Graph attribute name for tracklet/track IDs.
                Defaults to "track_id" if None.
            lineage_attr (str | None): Graph attribute name for lineage IDs.
                Defaults to "lineage_id" if None.
            scale (list[float] | None): Scaling factors for each dimension (including
                time). If None, all dimensions scaled by 1.0.
            ndim (int | None): Number of dimensions (3 for 2D+time, 4 for 3D+time).
                If None, inferred from segmentation or scale.
            features (FeatureDict | None): Pre-built FeatureDict with feature
                definitions. If provided, time_attr/pos_attr/tracklet_attr are ignored.
                Assumes that all features in the dict already exist on the graph (will
                be activated but not recomputed). If None, core computed features (pos,
                area, track_id) are auto-detected by checking if they exist on the graph.
            _segmentation (GraphArrayView | None): Internal parameter for reusing an
                existing GraphArrayView instance. Not intended for public use.
        """
        self.graph = graph
        if _segmentation is not None:
            # Reuse provided segmentation instance (internal use only)
            self.segmentation = _segmentation
        elif "mask" in graph.node_attr_keys():
            # Create new GraphArrayView from graph metadata
            try:
                array_view = GraphArrayView(
                    graph=graph,
                    shape=graph.metadata()["segmentation_shape"],
                    attr_key="node_id",
                    offset=0,
                )
                self.segmentation = array_view
            except (ValueError, KeyError) as err:
                raise ValueError(
                    "segmentation_shape is incompatible with graph, "
                    "check if mask and bbox attrs exist on nodes"
                ) from err
        else:
            self.segmentation = None
        self.scale = scale
        self.ndim = self._compute_ndim(
            self.segmentation.shape if self.segmentation is not None else None,
            scale,
            ndim,
        )
        self.axis_names = ["z", "y", "x"] if self.ndim == 4 else ["y", "x"]
        self.action_history = ActionHistory()
        self.node_id_counter = 1

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
                "Provided both FeatureDict and pos, time, tracklet, or lineage attr: "
                "ignoring attr"
                f" arguments ({pos_attr=}, {time_attr=}, {tracklet_attr=}, "
                f"{lineage_attr=}).",
                stacklevel=2,
            )
        self.features = (
            self._get_feature_set(time_attr, pos_attr, tracklet_attr, lineage_attr)
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

    def _get_feature_set(
        self,
        time_attr: str | None,
        pos_attr: str | tuple[str, ...] | list[str] | None,
        tracklet_key: str | None,
        lineage_key: str | None,
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
            lineage_key: Graph attribute name for lineage IDs (e.g., "lineage_id").
                if None, defaults to "lineage_id"

        Returns:
            FeatureDict initialized with time feature and position if no segmentation
        """
        # Use defaults if not provided
        time_key = time_attr if time_attr is not None else "time"
        if pos_attr is None:
            pos_attr = "pos"
        if tracklet_key is None:
            tracklet_key = "track_id"
        if lineage_key is None:
            lineage_key = "lineage_id"

        # Build static features dict - always include time
        features: dict[str, Feature] = {time_key: Time()}

        # Create FeatureDict with time feature
        # Position and tracklet features will be registered separately
        feature_dict = FeatureDict(
            features=features,
            time_key=time_key,
            position_key=None,
            tracklet_key=tracklet_key,
            lineage_key=lineage_key,
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
                TrackAnnotator(
                    self,  # type: ignore[arg-type]
                    tracklet_key=self.features.tracklet_key,
                    lineage_key=self.features.lineage_key,
                )
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
        if self.graph.num_nodes() == 0:
            return True

        # Check which attributes exist
        node_attrs = set(self.graph.node_attr_keys())
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
                lineage_key = annotator.lineage_key
                self.features.lineage_key = lineage_key
                core_computed_features.append(lineage_key)
        for key in core_computed_features:
            if self._check_existing_feature(key):
                # Add to FeatureDict if not already there
                if key not in self.features:
                    feature, _ = self.annotators.all_features[key]
                    self.add_feature(key, feature)
                self.annotators.activate_features([key])
            else:
                # enable it (compute it)
                self.enable_features([key])

    def nodes(self):
        return np.array(self.graph.node_ids())

    def edges(self):
        return np.array(self.graph.edge_ids())

    def in_degree(self, nodes: np.ndarray | None = None) -> np.ndarray:
        """Get the in-degree edge_ids of the nodes in the graph."""
        if nodes is not None:
            # make sure nodes is a numpy array
            if not isinstance(nodes, np.ndarray):
                nodes = np.array(nodes)

            return np.array([self.graph.in_degree(node.item()) for node in nodes])
        else:
            return np.array(self.graph.in_degree())

    def out_degree(self, nodes: np.ndarray | None = None) -> np.ndarray:
        if nodes is not None:
            # make sure nodes is a numpy array
            if not isinstance(nodes, np.ndarray):
                nodes = np.array(nodes)

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

        if isinstance(self.features.position_key, list):
            for idx, key in enumerate(self.features.position_key):
                self._set_nodes_attr(nodes, key, positions[:, idx].tolist())
        else:
            self._set_nodes_attr(nodes, self.features.position_key, positions)

    def set_position(self, node: Node, position: list | np.ndarray):
        self.set_positions([node], np.expand_dims(np.array(position), axis=0))

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

    def get_mask(self, node: Node) -> Mask | None:
        """Get the segmentation mask associated with a given node.

        Args:
            node (Node): The node to get the mask for.

        Returns:
            Mask | None: The segmentation mask for the node, or None if no
            segmentation is available.
        """
        if self.segmentation is None:
            return None

        mask = self.graph.nodes[node][td.DEFAULT_ATTR_KEYS.MASK]
        return mask

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

        # Get time and mask for the node
        time = self.get_time(node)
        mask = self.graph.nodes[node][td.DEFAULT_ATTR_KEYS.MASK]

        # Get local coordinates and convert to global using bbox offset
        local_coords = np.nonzero(mask.mask)
        global_coords = [coord + mask.bbox[dim] for dim, coord in enumerate(local_coords)]

        # Create time array matching the number of points
        time_array = np.full_like(global_coords[0], time)

        return (time_array, *global_coords)

    def _update_segmentation_cache(self, mask: td.Mask, time: int) -> None:
        """Invalidate cached chunks that overlap with the given mask.

        Args:
            mask: Mask object with .bbox attribute defining the affected region
            time: Time point of the mask
        """
        if self.segmentation is None:
            return

        cache = self.segmentation._cache

        # Only invalidate if this time point is in the cache
        if time not in cache._store:
            return

        # Convert bbox to slices directly
        # bbox format: [z_min, y_min, x_min, z_max, y_max, x_max] (3D)
        # or [y_min, x_min, y_max, x_max] (2D)
        ndim = len(mask.bbox) // 2
        volume_slicing = tuple(
            slice(mask.bbox[i], mask.bbox[i + ndim] + 1) for i in range(ndim)
        )

        # Use cache's method to get chunk bounds (same logic as cache.get())
        bounds = cache._chunk_bounds(volume_slicing)
        chunk_ranges = [range(lo, hi + 1) for lo, hi in bounds]

        # Invalidate all affected chunks
        cache_entry = cache._store[time]
        for chunk_idx in itertools.product(*chunk_ranges):
            if all(
                0 <= idx < grid_size
                for idx, grid_size in zip(chunk_idx, cache.grid_shape, strict=True)
            ):
                cache_entry.ready[chunk_idx] = False
                # Clear the buffer to ensure stale data isn't used
                # when the chunk is recomputed
                chunk_slc = tuple(
                    slice(ci * cs, min((ci + 1) * cs, fs))
                    for ci, cs, fs in zip(
                        chunk_idx, cache.chunk_shape, cache.shape, strict=True
                    )
                )
                cache_entry.buffer[chunk_slc] = 0

    def undo(self) -> bool:
        """Undo the last performed action from the action history.

        Returns:
            bool: True if an action was undone, False if there were no actions to undo
        """
        if self.action_history.undo():
            self.refresh.emit()
            return True
        return False

    def redo(self) -> bool:
        """Redo the last undone action from the action history.

        Returns:
            bool: True if an action was redone, False if there were no actions to redo
        """
        if self.action_history.redo():
            self.refresh.emit()
            return True
        return False

    def _get_new_node_ids(self, n: int) -> list[Node]:
        """Get a list of new node ids for creating new nodes.
        They will be unique from all existing nodes, but have no other guarantees.

        Args:
            n (int): The number of new node ids to return

        Returns:
            list[Node]: A list of new node ids.
        """
        ids = [self.node_id_counter + i for i in range(n)]
        self.node_id_counter += n
        for idx, _id in enumerate(ids):
            while self.graph.has_node(_id):
                _id = self.node_id_counter
                self.node_id_counter += 1
            ids[idx] = _id
        return ids

    def _compute_ndim(
        self,
        segmentation_shape: tuple[int, ...] | None,
        scale: list[float] | None,
        provided_ndim: int | None,
    ):
        seg_ndim = len(segmentation_shape) if segmentation_shape is not None else None
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
        self.graph.nodes[node][attr] = [value]

    def _set_nodes_attr(self, nodes: Iterable[Node], attr: str, values: Iterable[Any]):
        for node, value in zip(nodes, values, strict=False):
            self.graph.nodes[node][attr] = [value]

    def get_node_attr(self, node: Node, attr: str, required: bool = False):
        return self.graph.nodes[int(node)][attr]

    def get_nodes_attr(self, nodes: Iterable[Node], attr: str, required: bool = False):
        return [self.get_node_attr(node, attr, required=required) for node in nodes]

    def _set_edge_attr(self, edge: Edge, attr: str, value: Any):
        edge_id = self.graph.edge_id(edge[0], edge[1])
        self.graph.update_edge_attrs(attrs={attr: value}, edge_ids=[edge_id])

    def _set_edges_attr(self, edges: Iterable[Edge], attr: str, values: Iterable[Any]):
        for edge, value in zip(edges, values, strict=False):
            edge_id = self.graph.edge_id(edge[0], edge[1])
            self.graph.update_edge_attrs(attrs={attr: value}, edge_ids=[edge_id])

    def get_edge_attr(self, edge: Edge, attr: str, required: bool = False):
        if attr not in self.graph.edge_attr_keys():
            if required:
                raise KeyError(attr)
            return None
        edge_id = self.graph.edge_id(edge[0], edge[1])
        return self.graph.edges[edge_id][attr]

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
                self.add_feature(key, feature)

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
                self.delete_feature(key)

    def add_feature(self, key: str, feature: Feature) -> None:
        """Add a feature to the features dictionary and perform graph operations.

        This is the preferred way to add new features as it ensures both the
        features dictionary is updated and any necessary graph operations are performed.

        Args:
            key: The key for the new feature
            feature: The Feature object to add
        """
        # Add to the features dictionary
        self.features[key] = feature

        # Perform custom graph operations when a feature is added
        if feature["feature_type"] == "node" and key not in self.graph.node_attr_keys():
            self.graph.add_node_attr_key(
                key,
                default_value=feature["default_value"],
                dtype=to_polars_dtype(feature["value_type"]),
            )
        elif feature["feature_type"] == "edge" and key not in self.graph.edge_attr_keys():
            self.graph.add_edge_attr_key(
                key,
                default_value=feature["default_value"],
                dtype=to_polars_dtype(feature["value_type"]),
            )

    def delete_feature(self, key: str) -> None:
        """Delete a feature from the features dictionary and perform graph operations.

        This is the preferred way to delete features as it ensures both the
        features dictionary is updated and any necessary graph operations are performed.

        Args:
            key: The key for the feature to delete
        """
        # Remove from the features dictionary
        del self.features[key]

        # Perform custom graph operations when a feature is deleted
        if feature := self.annotators.all_features.get(key):
            feature_type = feature[0]["feature_type"]
            if feature_type == "node" and key in self.graph.node_attr_keys():
                self.graph.remove_node_attr_key(key)
            elif feature_type == "edge" and key in self.graph.edge_attr_keys():
                self.graph.remove_edge_attr_key(key)
