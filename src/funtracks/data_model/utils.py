from collections.abc import Sequence
from typing import Any

import numpy as np
import polars as pl
import rustworkx as rx
import tracksdata as td


def td_get_single_attr_from_node(graph, node_id: int, attrs: Sequence[str]):
    """Get a single attribute from a node in a tracksdata graph."""

    # TODO: typechecking should somehow resolve this...
    if not isinstance(node_id, int):
        if isinstance(node_id, list):
            if len(node_id) > 1:
                raise ValueError("node_id must be an single integer")
            else:
                node_id = int(node_id[0])
        node_id = int(node_id)

    item = graph.filter(node_ids=[node_id]).node_attrs(attrs).item()
    if isinstance(item, pl.Series):
        return item.to_list()
    else:
        return item


def td_get_single_attr_from_edge(graph, edge: tuple[int, int], attrs: Sequence[str]):
    """Get a single attribute from a edge in a tracksdata graph."""

    item = graph.filter(node_ids=[edge[0], edge[1]]).edge_attrs()[attrs].item()
    if isinstance(item, pl.Series):
        return item.to_list()
    else:
        return item


def convert_np_types(data):
    """Recursively convert numpy and polars types to native Python types."""
    if isinstance(data, dict):
        return {key: convert_np_types(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_np_types(item) for item in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()  # Convert numpy arrays to Python lists
    elif isinstance(data, np.integer):
        return int(data)  # Convert numpy integers to Python int
    elif isinstance(data, np.floating):
        return float(data)  # Convert numpy floats to Python float
    elif isinstance(data, pl.Series):
        return data.to_list()  # Convert polars Series to Python list
    else:
        return data  # Return the data as-is if it's already a native Python type


def td_to_dict(graph) -> dict:
    """Convert the tracks graph to a dictionary format similar to
    networkx.node_link_data.

    This is used within Tracks.save to save the graph to a json file.
    """
    node_attr_names = graph.node_attr_keys.copy()
    node_attr_names.insert(0, "node_id")
    node_data_all = graph.node_attrs()
    nodes = []
    for i, node in enumerate(graph.node_ids()):
        node_data = node_data_all[i]
        node_data_dict = {
            node_attr_names[i]: convert_np_types(node_data[node_attr_names[i]].item())
            for i in range(len(node_attr_names))
        }
        node_dict = {"id": node}
        node_dict.update(node_data_dict)  # Add all attributes to the dictionary
        node_dict.pop("id")
        nodes.append(node_dict)

    edge_attr_names = graph.edge_attr_keys.copy()
    edge_attr_names.insert(0, "edge_id")
    edge_attr_names.insert(1, "source_id")
    edge_attr_names.insert(2, "target_id")
    edges = []
    edge_data_all = graph.edge_attrs()
    for i, _ in enumerate(graph.edge_ids()):
        edge_data = edge_data_all[i]
        edge_data_dict = {
            edge_attr_names[i]: convert_np_types(edge_data[edge_attr_names[i]].item())
            for i in range(len(edge_attr_names))
        }
        edge_dict = {
            "source": edge_data_dict["source_id"],
            "target": edge_data_dict["target_id"],
        }
        edge_data_dict.pop("source_id")
        edge_data_dict.pop("target_id")
        edge_dict.update(edge_data_dict)  # Add all attributes to the dictionary
        edges.append(edge_dict)

    edges = sorted(edges, key=lambda edge: edge["edge_id"])

    return {
        "directed": True,  # all TracksData graphs are directed
        "multigraph": False,  # all TracksData garphs are not multigraphs
        "graph": {},  # Add any graph-level attributes if needed
        "nodes": nodes,
        "edges": edges,
    }


def td_from_dict(graph_dict):
    """Convert a dictionary to a rustworkx graph."""
    # Create a new directed graph
    graph_rx = rx.PyDiGraph()

    # Get the attribute keys in the order they appear in the first node
    node_attr_keys = list(graph_dict["nodes"][0].keys())
    node_attr_keys.remove("node_id")  # node_id is handled separately

    # Add nodes
    node_id_map = {}
    for node in graph_dict["nodes"]:
        # Create node data dict in the same order as original
        node_data = {k: node[k] for k in node_attr_keys}
        node_id = graph_rx.add_node(node_data)
        node_id_map[node["node_id"]] = node_id

    # Get edge attribute keys in order
    edge_attr_keys = list(graph_dict["edges"][0].keys())
    edge_attr_keys.remove("source")
    edge_attr_keys.remove("target")

    # Add edges
    for edge in graph_dict["edges"]:
        source_id = node_id_map[edge["source"]]
        target_id = node_id_map[edge["target"]]
        # Create edge data dict in the same order as original
        edge_data = {k: edge[k] for k in edge_attr_keys}
        graph_rx.add_edge(source_id, target_id, edge_data)

    # Use the same node_id_map we created while building the graph
    graph_td = td.graph.IndexedRXGraph(graph_rx, node_id_map=node_id_map)

    return graph_td


def td_graph_edge_list(graph):
    """Get list of edges from a tracksdata graph.

    Args:
        graph: A tracksdata graph

    Returns:
        list: List of edges: [[source_id, target_id], ...]
    """
    existing_edges = (
        graph.edge_attrs().select(["source_id", "target_id"]).to_numpy().tolist()
    )
    return existing_edges


def td_get_node_ids_from_df(df):
    """Get list of node_ids from a polars DataFrame, handling empty case.

    Args:
        df: A polars DataFrame that may contain a 'node_id' column

    Returns:
        list: List of node_ids if DataFrame has rows, empty list otherwise
    """
    return list(df["node_id"]) if len(df) > 0 else []


def td_get_predecessors(graph, node):
    """Get list of predecessor node IDs for a given node.

    Args:
        graph: A tracksdata graph
        node: Node ID to get predecessors for

    Returns:
        list: List of predecessor node IDs
    """
    predecessors_df = graph.predecessors(node)
    return td_get_node_ids_from_df(predecessors_df)


def td_get_successors(graph, node):
    """Get list of successor node IDs for a given node.

    Args:
        graph: A tracksdata graph
        node: Node ID to get successors for

    Returns:
        list: List of successor node IDs
    """
    successors_df = graph.successors(node)
    return td_get_node_ids_from_df(successors_df)


def values_are_equal(val1: Any, val2: Any) -> bool:
    """
    Compare two values that could be of any type (arrays, lists, scalars, etc.)

    Args:
        val1: First value to compare
        val2: Second value to compare

    Returns:
        bool: True if values are equal, False otherwise
    """
    # If both are None, they're equal
    if val1 is None and val2 is None:
        return True

    # If only one is None, they're not equal
    if val1 is None or val2 is None:
        return False

    # Handle numpy arrays
    if isinstance(val1, np.ndarray) or isinstance(val2, np.ndarray):
        try:
            return np.array_equal(np.asarray(val1), np.asarray(val2), equal_nan=True)
        except (ValueError, TypeError):
            # Return False if arrays cannot be compared (incompatible shapes or types)
            return False

    # Handle lists that might need to be compared as arrays
    if isinstance(val1, list) and isinstance(val2, list):
        try:
            return np.array_equal(np.asarray(val1), np.asarray(val2), equal_nan=True)
        except (ValueError, TypeError):
            # Return False if arrays cannot be compared (incompatible shapes or types)
            # If can't convert to numpy arrays, fall back to regular comparison
            return val1 == val2

    # Default comparison for other types
    return val1 == val2


def validate_and_merge_node_attrs(attrs_of_root_node: dict, node_dict: dict) -> dict:
    """
    Compare and validate two node attribute dictionaries.

    Args:
        attrs_of_root_node: Dictionary containing the root node attributes (reference)
        node_dict: Dictionary containing the node attributes to compare/merge

    Returns:
        Updated dictionary with merged values

    Raises:
        ValueError: If node_dict contains fields not present in attrs_of_root_node
    """
    # Check for invalid fields in node_dict
    invalid_fields = set(node_dict.keys()) - set(attrs_of_root_node.keys())
    if invalid_fields:
        raise ValueError(
            f"Node dictionary contains fields not present in root: {invalid_fields}"
        )

    # Create a new dict starting with root values
    merged_attrs = attrs_of_root_node.copy()

    # Compare and update values
    for field, value in node_dict.items():
        # Skip None values from node_dict to keep root values
        if value is not None and not values_are_equal(value, attrs_of_root_node[field]):
            merged_attrs[field] = value

    return merged_attrs
