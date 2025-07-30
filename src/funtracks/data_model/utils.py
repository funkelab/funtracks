from typing import Sequence

import polars as pl
import tracksdata as td
import numpy as np
import rustworkx as rx

def td_get_single_attr_from_node(graph, node_ids: Sequence[int], attrs: Sequence[str]):
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
    """Convert the tracks graph to a dictionary format similar to networkx.node_link_data."""
    node_attr_names = graph.node_attrs().columns
    nodes = []
    for node_index in range(len(graph.node_ids())):
        node_data = graph.node_attrs()[node_index]
        node_data_dict = {node_attr_names[i]: convert_np_types(node_data[node_attr_names[i]].item()) for i in range(len(node_attr_names))}
        node_dict = {'id': graph.node_ids()[node_index]}
        node_dict.update(node_data_dict)  # Add all attributes to the dictionary
        node_dict.pop('id')
        nodes.append(node_dict)
    
    edge_attr_names = graph.edge_attrs().columns
    edges = []
    for edge_index in range(len(graph.edge_ids())):
        edge_data = graph.edge_attrs()[edge_index]
        edge_data_dict = {edge_attr_names[i]: convert_np_types(edge_data[edge_attr_names[i]].item()) for i in range(len(edge_attr_names))}
        edge_dict = {
            'source': edge_data_dict['source_id'],
            'target': edge_data_dict['target_id']
        }
        # edge_data_dict.pop('edge_id') #keep edge, needed for rx>td conversion in td_from_dict loading script
        edge_data_dict.pop('source_id')
        edge_data_dict.pop('target_id')
        edge_dict.update(edge_data_dict)  # Add all attributes to the dictionary
        edges.append(edge_dict)

    edges = sorted(edges, key=lambda edge: edge['edge_id'])


    return {
        'directed': True, #all TracksData graphs are directed
        'multigraph': False, #all TracksData garphs are not multigraphs (TODO: check this!)
        'graph': {},  # Add any graph-level attributes if needed
        'nodes': nodes,
        'edges': edges
    }

def td_from_dict(graph_dict):
    """Convert a dictionary to a rustworkx graph."""
    # Create a new directed graph
    graph_rx = rx.PyDiGraph()

    # Add nodes
    node_id_map = {}
    for node in graph_dict['nodes']:
        node_id = graph_rx.add_node(node)
        node_id_map[node['node_id']] = node_id

    # Add edges
    for edge in graph_dict['edges']:
        source_id = node_id_map[edge['source']]
        target_id = node_id_map[edge['target']]
        # Remove source and target from edge attributes if they exist
        edge_data = {k: v for k, v in edge.items() if k not in ['source', 'target']}
        graph_rx.add_edge(source_id, target_id, edge_data)
    
    node_ids = [node['node_id'] for node in graph_dict['nodes']]
    node_id_map = {node: i for i, node in enumerate(node_ids)}
    graph_td = td.graph.IndexedRXGraph(graph_rx, node_id_map=node_id_map)

    return graph_td

# Usage
# graph_dict = { ... }  # Your dictionary representation of the graph
# rustworkx_graph = dict_to_rustworkx_graph(graph_dict)