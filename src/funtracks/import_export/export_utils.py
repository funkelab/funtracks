import networkx as nx


def filter_graph_with_ancestors(graph: nx.DiGraph, nodes_to_keep: set[int]) -> list[int]:
    """
    Filter a graph to keep only the nodes in `nodes_to_keep` and their ancestors.

    Args:
        graph (nx.DiGraph): The original directed graph.
        nodes_to_keep (set[int]): The set of nodes to keep in the graph.

    Returns:
        list[int]: A subset of the original nodes in the graph containing only the nodes
            in `nodes_to_keep` and their ancestors.
    """
    # Set to store all nodes to keep, including ancestors
    all_nodes_to_keep = set(nodes_to_keep)

    # Traverse ancestors for each node in nodes_to_keep
    for node in nodes_to_keep:
        ancestors = nx.ancestors(graph, node)  # Get all ancestors of the node
        all_nodes_to_keep.update(ancestors)  # Add ancestors to the set

    return list(all_nodes_to_keep)
