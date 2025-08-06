from collections.abc import Sequence

import numpy as np
import polars as pl
import rustworkx as rx
import tracksdata as td


def td_get_single_attr_from_node(graph, node_ids: Sequence[int], attrs: Sequence[str]):
    """Get a single attribute from a node in a tracksdata graph."""
    item = graph.filter(node_ids=node_ids).node_attrs(attrs).item()
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

    # Add nodes
    node_id_map = {}
    for node in graph_dict["nodes"]:
        node_id = graph_rx.add_node(node)
        node_id_map[node["node_id"]] = node_id

    # Add edges
    for edge in graph_dict["edges"]:
        source_id = node_id_map[edge["source"]]
        target_id = node_id_map[edge["target"]]
        # Remove source and target from edge attributes if they exist
        edge_data = {k: v for k, v in edge.items() if k not in ["source", "target"]}
        graph_rx.add_edge(source_id, target_id, edge_data)

    node_ids = [node["node_id"] for node in graph_dict["nodes"]]
    node_id_map = {node: i for i, node in enumerate(node_ids)}
    graph_td = td.graph.IndexedRXGraph(graph_rx, node_id_map=node_id_map)

    return graph_td


def td_graph_has_edge(graph, edge):
    """Check if a graph has an edge between two nodes."""

    return (
        edge in graph.edge_attrs().select(["source_id", "target_id"]).to_numpy().tolist()
    )
