from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    TypeAlias,
)
from warnings import warn

import numpy as np
import polars as pl
import tracksdata as td
from psygnal import Signal
from tracksdata.array import GraphArrayView
from tracksdata.nodes import Mask

from funtracks.actions.action_history import ActionHistory
from funtracks.annotators import TrackAnnotator
from funtracks.features import (
    Feature,
    FeatureDict,
    Position,
    SegBbox,
    SegMask,
    Solution,
    Time,
)
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

logger = logging.getLogger(__name__)


class Tracks:
    """A set of tracks consisting of a graph and an optional segmentation.

    The graph nodes represent detections and must have a time attribute and
    position attribute. Edges in the graph represent links across time.

    Attributes:
        graph_full (td.graph.BaseGraph): The full graph (first-class): every node/edge
            ever known, including soft-deleted (solution=False) candidates. Nodes
            represent detections, edges represent links across time.
        graph_solution (td.graph.GraphView): A solution==True view derived from
            graph_full; the user-visible tracking solution.
        features (FeatureDict): Dictionary of features tracked on graph nodes/edges.
        annotators (AnnotatorRegistry): List of annotators that compute features.
        scale (list[float] | None): How much to scale each dimension by, including time.
        ndim (int): Number of dimensions (3 for 2D+time, 4 for 3D+time).
    """

    refresh = Signal(object)
    action_applied = Signal(object)

    def __init__(
        self,
        graph: td.graph.BaseGraph,
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
            graph (td.graph.BaseGraph): the full base graph (graph_full) with nodes as
                detections and edges as links. Must be a base graph, not a view;
                graph_solution is built internally as its solution==True view.
            time_attr (str | None): Graph attribute name for time. Defaults to "time"
                if None.
            pos_attr (str | tuple[str, ...] | list[str] | None): Graph attribute
                name(s) for position. Can be:
                - Single string for one attribute containing position array
                - List/tuple of strings for multi-axis (one attribute per axis)
                Defaults to "pos" if None.
            tracklet_attr (str | None): Graph attribute name for tracklet/track IDs.
                Defaults to "tracklet_id" if None. Every Tracks gets a TrackAnnotator
                and track ids (no "plain" vs "solution" distinction); existing ids on
                the graph are reused, otherwise they are computed.
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
                tracklet_id) are auto-detected by checking if they exist on the graph.
            _segmentation (GraphArrayView | None): Internal parameter for reusing an
                existing GraphArrayView instance. Not intended for public use.
        """
        # graph_full is the first-class object: the base graph holding every node/edge
        # ever known, including soft-deleted (solution=False) candidates. graph_solution
        # is derived from it as a solution==True view.
        if isinstance(graph, td.graph.GraphView):
            raise ValueError(
                "Tracks requires the full base graph (graph_full), not a GraphView. "
                "graph_solution is built internally as a solution==True view of it."
            )
        if "solution" not in graph.node_attr_keys():
            graph.add_node_attr_key("solution", default_value=True, dtype=pl.Boolean)
        if "solution" not in graph.edge_attr_keys():
            graph.add_edge_attr_key("solution", default_value=True, dtype=pl.Boolean)
        self.graph_full = graph
        self.graph_solution = graph.filter(
            td.NodeAttr("solution") == True,  # noqa: E712
            td.EdgeAttr("solution") == True,  # noqa: E712
        ).subgraph()
        if _segmentation is not None:
            # Reuse provided segmentation instance (internal use only)
            self.segmentation = _segmentation
        elif "mask" in graph.node_attr_keys():
            # Create new GraphArrayView from graph metadata, but only if
            # segmentation_shape is present. A graph can carry per-node mask
            # attributes without a full dense segmentation array (e.g. after a
            # geff round-trip that was saved without segmentation).
            seg_shape = graph.metadata.get("segmentation_shape")
            if seg_shape is not None:
                try:
                    # Render the segmentation from the solution view so soft-deleted
                    # nodes drop out of the array, mirroring the user-visible graph.
                    array_view = GraphArrayView(
                        graph=self.graph_solution,
                        shape=seg_shape,
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

        # 4. Enforce the track-id invariant on BOTH paths: every Tracks has a tracklet
        # key and a registered TrackAnnotator, with tracklet_id/lineage_id registered
        # and computed. A provided FeatureDict that omitted them is completed here.
        self._ensure_track_features()

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
            tracklet_key: Graph attribute name for tracklet/track IDs.
                Defaults to "tracklet_id" if None (every Tracks gets track ids).
            lineage_key: Graph attribute name for lineage IDs.
                Defaults to "lineage_id" if None.

        Returns:
            FeatureDict initialized with time feature and position if no segmentation
        """
        # Use defaults if not provided
        time_key = time_attr if time_attr is not None else "time"
        if pos_attr is None:
            pos_attr = "pos"
        # Every Tracks has a tracklet/lineage key (no "plain" vs "solution" split):
        # default them like time/pos. _ensure_track_features() then registers and
        # computes the ids; on an empty solution view that is a no-op.
        tracklet_key = tracklet_key if tracklet_key is not None else "tracklet_id"
        lineage_key = lineage_key if lineage_key is not None else "lineage_id"

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

        # Register position feature
        if isinstance(pos_attr, tuple | list):
            # Multiple position attributes (one per axis) -
            # always static, already on nodes
            multi_position_key = list(pos_attr)
            for attr in pos_attr:
                feature_dict[attr] = {
                    "feature_type": "node",
                    "value_type": "float",
                    "num_values": 1,
                    "default_value": None,
                }
            # For multi-axis, set position_key directly
            # (not a single feature to register)
            feature_dict.position_key = multi_position_key
        elif self.segmentation is None:
            # Single position attribute without segmentation - static, provided by user
            single_position_key = pos_attr
            pos_feature = Position(axes=self.axis_names)
            feature_dict.register_position_feature(single_position_key, pos_feature)
        # else: single pos_attr with segmentation - RegionpropsAnnotator will handle it

        # Register solution feature when present on the graph
        if "solution" in self.graph_solution.node_attr_keys():
            feature_dict["solution"] = Solution()

        # Register mask and bbox features if segmentation exists
        if self.segmentation is not None:
            feature_dict[td.DEFAULT_ATTR_KEYS.MASK] = SegMask(
                self.ndim, bbox_key=td.DEFAULT_ATTR_KEYS.BBOX
            )
            feature_dict[td.DEFAULT_ATTR_KEYS.BBOX] = SegBbox(self.ndim)

        return feature_dict

    def _get_annotators(self) -> AnnotatorRegistry:
        """Instantiate and return core annotators based on available data.

        Creates annotators conditionally:
        - RegionpropsAnnotator: Only if segmentation is provided
        - EdgeAnnotator: Only if segmentation is provided
        - TrackAnnotator: Only if this is a Tracks instance

        Each annotator is configured with appropriate keys from self.features.

        Returns:
            AnnotatorRegistry containing all applicable annotators
        """
        # Import here to avoid circular dependency
        from funtracks.annotators import (
            AnnotatorRegistry,
            EdgeAnnotator,
            RegionpropsAnnotator,
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

        # TrackAnnotator is registered on every Tracks — track ids are a core feature,
        # not a separate "type" of tracks. On an empty solution view it simply computes
        # nothing until nodes are added.
        annotator_list.append(
            TrackAnnotator(
                self,
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
        if self.graph_solution.num_nodes() == 0:
            return True

        # Check which attributes exist
        node_attrs = set(self.graph_solution.node_attr_keys())
        return key in node_attrs

    def _setup_core_computed_features(self) -> None:
        """Sets up core computed position features.

        Registers the position feature from the RegionpropsAnnotator into the
        FeatureDict, activating it if it already exists on the graph or computing it
        otherwise. Track-id features are handled separately by _ensure_track_features.
        """
        # Import here to avoid circular dependency
        from funtracks.annotators import RegionpropsAnnotator

        core_features: list[str] = []
        for annotator in self.annotators:
            if isinstance(annotator, RegionpropsAnnotator):
                pos_key = annotator.pos_key
                if self.features.position_key is None:
                    self.features.position_key = pos_key
                core_features.append(pos_key)
        self._register_core_features(core_features)

    def _register_core_features(self, keys: list[str]) -> None:
        """Register each key as a feature: activate it if it already exists on the
        graph, otherwise enable (compute) it."""
        for key in keys:
            if self._check_existing_feature(key):
                if key not in self.features:
                    feature, _ = self.annotators.all_features[key]
                    self.add_feature(key, feature)
                self.annotators.activate_features([key])
            else:
                self.enable_features([key])

    def _ensure_track_features(self) -> None:
        """Ensure the track-id core features exist on this Tracks.

        Every Tracks has a registered TrackAnnotator and a tracklet key (no "plain"
        vs "solution" split). This syncs features.tracklet_key/lineage_key from the
        annotator and registers + computes (or activates, if already present)
        tracklet_id/lineage_id. Runs on both the provided-FeatureDict and the
        auto-detect construction paths; a no-op on an empty solution view.
        """
        annotator = self.track_annotator
        self.features.tracklet_key = annotator.tracklet_key
        self.features.lineage_key = annotator.lineage_key
        # A tracklet column can exist yet still hold the -1 sentinel ("not computed",
        # the column default). Trusting it would activate stale ids and seed a phantom
        # tracklet -1 in the annotator bookkeeping, so force a recompute from topology.
        if self._has_uncomputed_track_ids(annotator.tracklet_key):
            self.enable_features([annotator.tracklet_key, annotator.lineage_key])
        else:
            self._register_core_features([annotator.tracklet_key, annotator.lineage_key])

    def _has_uncomputed_track_ids(self, tracklet_key: str) -> bool:
        """True if the tracklet column exists but any node still holds the -1 sentinel.

        A missing column returns False: _register_core_features computes it from scratch.
        """
        if self.graph_solution.num_nodes() == 0:
            return False
        if tracklet_key not in self.graph_solution.node_attr_keys():
            return False
        values = self.graph_solution.node_attrs(attr_keys=[tracklet_key])[tracklet_key]
        return bool((values == -1).any())

    def nodes(self):
        return np.array(self.graph_solution.node_ids())

    def edges(self):
        return np.array(self.graph_solution.edge_ids())

    def predecessors(self, node: int) -> list[int]:
        return list(self.graph_solution.predecessors(node))

    def successors(self, node: int) -> list[int]:
        return list(self.graph_solution.successors(node))

    def get_positions(self, nodes: Iterable[Node], incl_time: bool = False) -> np.ndarray:
        """Get the positions of nodes in the graph. Optionally include the
        time frame as the first dimension. Raises an error if any of the nodes
        are not in the graph.

        NOTE: fetches all nodes in the graph internally. Optimised for bulk use.
        For a single node use get_position() instead.

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
        nodes = list(nodes)
        position_key = self.features.position_key
        pos_keys = (
            list(position_key) if isinstance(position_key, list) else [position_key]
        )
        attr_keys = (
            [td.DEFAULT_ATTR_KEYS.NODE_ID]
            + pos_keys
            + ([self.features.time_key] if incl_time else [])
        )

        # Read from graph_full (consistent with get_position / the attr-helper policy):
        # positions are intrinsic node attrs, so this also resolves soft-deleted
        # (solution=False) nodes instead of KeyError-ing like a graph_solution query.
        df = self.graph_full.node_attrs(attr_keys=attr_keys)
        id_to_row = {
            nid: i for i, nid in enumerate(df[td.DEFAULT_ATTR_KEYS.NODE_ID].to_list())
        }

        if len(pos_keys) == 1:
            pos_arr = df[pos_keys[0]].to_numpy()
            if pos_arr.ndim == 1:
                pos_arr = pos_arr[:, np.newaxis]
        else:
            pos_arr = np.stack([df[k].to_numpy() for k in pos_keys], axis=1)

        # Reorder rows to match input node order
        idx = [id_to_row[node] for node in nodes]
        positions = pos_arr[idx]

        if incl_time:
            times = df[self.features.time_key].to_numpy()[idx]
            positions = np.c_[times, positions]

        return positions

    def get_position(self, node: Node, incl_time=False) -> list:
        """Get position of a single node. Uses a direct per-node query — do not
        use in a loop over many nodes; use get_positions() instead."""

        position_key = self.features.position_key
        pos_keys = (
            list(position_key) if isinstance(position_key, list) else [position_key]
        )
        pos = [self.get_node_attr(node, k) for k in pos_keys]
        if len(pos) == 1:
            # pos may be a numpy array (fixed-size array column) or a scalar
            val = pos[0]
            pos = list(val) if hasattr(val, "__iter__") else pos
        if incl_time:
            pos = [self.get_node_attr(node, self.features.time_key)] + pos
        return pos

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
            incl_time (bool, optional): If true, time is the first column and is
                included in ndim. Defaults to False.
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
        """Batch fetch times for many nodes in one query.
        NOTE: fetches all nodes in the graph internally. Optimised for bulk use.
        For a single node use get_time() instead.
        """
        nodes = list(nodes)
        df = self.graph_full.node_attrs(
            attr_keys=[td.DEFAULT_ATTR_KEYS.NODE_ID, self.features.time_key]
        )
        id_to_val = dict(
            zip(
                df[td.DEFAULT_ATTR_KEYS.NODE_ID].to_list(),
                df[self.features.time_key].to_list(),
                strict=True,
            )
        )
        return [id_to_val[node] for node in nodes]

    def get_time(self, node: Node) -> int:
        """Get the time frame of a given node. Raises an error if the node
        is not in the graph.

        Args:
            node (Any): The node id to get the time frame for

        Returns:
            int: The time frame that the node is in
        """
        return int(self.get_node_attr(node, self.features.time_key))

    def get_mask(
        self, node: Node, mask_key: str = td.DEFAULT_ATTR_KEYS.MASK
    ) -> Mask | None:
        """Get the segmentation mask associated with a given node.

        Args:
            node: The node to get the mask for.
            mask_key: The feature key for the mask column.
                Defaults to the standard mask key.

        Returns:
            Mask | None: The segmentation mask for the node, or None if no
            segmentation is available.
        """
        if self.segmentation is None:
            return None

        mask = self.graph_full.nodes[node][mask_key]
        return mask

    def update_mask(
        self, node: Node, mask: Mask, mask_key: str = td.DEFAULT_ATTR_KEYS.MASK
    ) -> None:
        """Update the segmentation mask for an existing node, auto-syncing the bbox.

        Writes both the mask and its bbox to the graph. The bbox key is
        looked up from the mask Feature's derived_features list.

        Args:
            node: The node to update the mask for.
            mask: The Mask object (carries .bbox).
            mask_key: The feature key for the mask column.
                Defaults to the standard mask key.
        """
        self.graph_full.nodes[node][mask_key] = mask
        mask_feature = self.features.get(mask_key)
        if mask_feature is not None:
            # NOTE: all derived features of a mask are currently assumed to be
            # its bounding box. Revisit if non-bbox derived features are added.
            for derived_key in mask_feature.get("derived_features", []):
                self.graph_full.nodes[node][derived_key] = mask.bbox

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
        # Check against graph_full, not the solution view: a soft-deleted
        # (solution=False) node still occupies its id in the full graph, so reissuing
        # it to a genuinely new node would collide with the still-present root node.
        ids = [self.node_id_counter + i for i in range(n)]
        self.node_id_counter += n
        for idx, _id in enumerate(ids):
            while self.graph_full.has_node(_id):
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

    # NOTE: the low-level attribute get/set helpers below target `graph_full`, not the
    # solution view. Attribute *values* are intrinsic to a node/edge and live on the full
    # graph; the solution view is a membership filter over it. Because graph_full ⊇
    # graph_solution and their attr dicts are shared by reference, writing via graph_full
    # is identical to writing via the view for any in-solution node (the view sees it
    # automatically) and additionally works for soft-deleted (solution=False) candidates.
    def _set_node_attr(self, node: Node, attr: str, value: Any):
        if isinstance(value, np.ndarray):
            value = list(value)
        self.graph_full.nodes[node][attr] = value

    def _set_nodes_attr(self, nodes: Iterable[Node], attr: str, values: Iterable[Any]):
        nodes_list = list(nodes)
        values_list = list(values)
        if nodes_list:
            self.graph_full.update_node_attrs(
                attrs={attr: values_list}, node_ids=nodes_list
            )

    def get_node_attr(self, node: Node, attr: str):
        return self.graph_full.nodes[int(node)][attr]

    def get_nodes_attr(self, nodes: Iterable[Node], attr: str):
        return [self.get_node_attr(node, attr) for node in nodes]

    def _set_edge_attr(self, edge: Edge, attr: str, value: Any):
        edge_id = self.graph_full.edge_id(edge[0], edge[1])
        self.graph_full.update_edge_attrs(attrs={attr: value}, edge_ids=[edge_id])

    def _set_edges_attr(self, edges: Iterable[Edge], attr: str, values: Iterable[Any]):
        for edge, value in zip(edges, values, strict=False):
            edge_id = self.graph_full.edge_id(edge[0], edge[1])
            self.graph_full.update_edge_attrs(attrs={attr: value}, edge_ids=[edge_id])

    def get_edge_attr(self, edge: Edge, attr: str):
        if attr not in self.graph_full.edge_attr_keys():
            return None
        edge_id = self.graph_full.edge_id(edge[0], edge[1])
        return self.graph_full.edges[edge_id][attr]

    def get_edges_attr(self, edges: Iterable[Edge], attr: str):
        return [self.get_edge_attr(edge, attr) for edge in edges]

    # ========== Feature Management ==========

    def notify_annotators(self, action: BasicAction) -> None:
        """Notify annotators about an action so they can recompute affected features.

        Delegates to the annotator registry which broadcasts to all annotators.
        The action contains all necessary information about which elements to update.

        Args:
            action: The action that triggered this notification
        """
        self.action_applied.emit(action)
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

        # Add to FeatureDict and graph schema
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
        ft = feature["feature_type"]
        if "node" in ft and key not in self.graph_solution.node_attr_keys():
            # "mask" value_type maps to pl.Object via to_polars_dtype
            dtype = to_polars_dtype(feature["value_type"])
            num_values = feature.get("num_values")
            if num_values is not None and num_values > 1:
                dtype = pl.Array(dtype, num_values)
            self.graph_solution.add_node_attr_key(
                key,
                default_value=feature["default_value"],
                dtype=dtype,
            )
        if "edge" in ft and key not in self.graph_solution.edge_attr_keys():
            self.graph_solution.add_edge_attr_key(
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
        # Get feature metadata before removing from dict
        feature = self.features.get(key)

        # Remove from the features dictionary
        del self.features[key]

        # Cascade-delete any derived features
        if feature is not None:
            for derived_key in feature.get("derived_features", []):
                if derived_key in self.features:
                    self.delete_feature(derived_key)

        # Determine feature_type from FeatureDict entry or annotators
        if feature is not None:
            feature_type = feature["feature_type"]
        elif ann_entry := self.annotators.all_features.get(key):
            feature_type = ann_entry[0]["feature_type"]
        else:
            return

        # Perform custom graph operations when a feature is deleted
        if "node" in feature_type and key in self.graph_solution.node_attr_keys():
            self.graph_solution.remove_node_attr_key(key)
        if "edge" in feature_type and key in self.graph_solution.edge_attr_keys():
            self.graph_solution.remove_edge_attr_key(key)

    # ========== Track ID management (solution view) ==========
    # These operate on the solution view via the TrackAnnotator, which every Tracks
    # has (track ids are a core feature). On an empty solution view they are no-ops.

    @property
    def track_annotator(self):
        """The registered TrackAnnotator — always present, since track ids are a core
        feature of every Tracks."""
        for annotator in self.annotators:
            if isinstance(annotator, TrackAnnotator):
                return annotator
        return None

    @property
    def max_track_id(self) -> int:
        return self.track_annotator.max_tracklet_id

    def get_next_track_id(self) -> int:
        """Return the next available track_id.

        The max_tracklet_id in TrackAnnotator is updated automatically when
        a node is added or track IDs are updated via UpdateTrackIDs.
        """
        return self.track_annotator.max_tracklet_id + 1

    def get_next_lineage_id(self) -> int:
        """Return the next available lineage_id.

        The max_lineage_id in TrackAnnotator is updated automatically when
        a node is added or lineage IDs are updated via UpdateTrackIDs.
        """
        return self.track_annotator.max_lineage_id + 1

    def get_track_id(self, node) -> int:
        track_id = self.get_node_attr(node, self.features.tracklet_key)
        return track_id

    def get_track_ids(self, nodes) -> list[int]:
        """Batch version of get_track_id — one query fetching all nodes in the graph.
        NOTE: always fetches the entire graph internally. Optimised for bulk (all-node)
        calls. For small subsets or single nodes use get_track_id() instead."""

        tracklet_key = self.features.tracklet_key
        df = self.graph_full.node_attrs(
            attr_keys=[td.DEFAULT_ATTR_KEYS.NODE_ID, tracklet_key]
        )
        id_to_val = dict(
            zip(
                df[td.DEFAULT_ATTR_KEYS.NODE_ID].to_list(),
                df[tracklet_key].to_list(),
                strict=True,
            )
        )
        return [id_to_val[node] for node in nodes]

    def get_lineage_id(self, node) -> int:
        """Get the lineage ID for a node.

        Args:
            node: The node to get lineage ID for

        Returns:
            The lineage ID.
        """
        return self.get_node_attr(node, self.features.lineage_key)

    def get_track_neighbors(
        self, track_id: int, time: int
    ) -> tuple[Node | None, Node | None]:
        """Get the last node with the given track id before time, and the first node
        with the track id after time, if any. Does not assume that a node with
        the given track_id and time is already in tracks, but it can be.

        Args:
            track_id (int): The track id to search for
            time (int): The time point to find the immediate predecessor and successor
                for

        Returns:
            tuple[Node | None, Node | None]: The last node before time with the given
            track id, and the first node after time with the given track id,
            or Nones if there are no such nodes.
        """
        if (
            track_id not in self.track_annotator.tracklet_id_to_nodes
            or len(self.track_annotator.tracklet_id_to_nodes[track_id]) == 0
        ):
            return None, None
        candidates = self.track_annotator.tracklet_id_to_nodes[track_id]
        candidates.sort(key=lambda n: self.get_time(n))

        pred = None
        succ = None
        for cand in candidates:
            if self.get_time(cand) < time:
                pred = cand
            elif self.get_time(cand) > time:
                succ = cand
                break
        return (
            int(pred) if pred is not None else None,
            int(succ) if succ is not None else None,
        )

    def has_track_id_at_time(self, track_id: int, time: int) -> bool:
        """Function to check if a node with given track id exists at given time point.

        Args:
            track_id (int): The track id to search for.
            time (int): The time point to check.

        Returns:
            True if a node with given track id exists at given time point.
        """
        nodes = self.track_annotator.tracklet_id_to_nodes.get(track_id)
        if not nodes:
            return False

        return time in [self.get_time(node) for node in nodes]
