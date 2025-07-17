import networkx as nx
from motile import TrackGraph
from funtracks.data_model.graph_utils import graph_to_nx
from networkx.utils import graphs_equal


def test_graph_to_nx(graph_3d: nx.DiGraph):
    track_graph = TrackGraph(nx_graph=graph_3d, frame_attribute="time")
    nx_graph = graph_to_nx(track_graph)
    assert graphs_equal(graph_3d, nx_graph)
