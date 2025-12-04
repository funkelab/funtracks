from .tracks import Tracks  # noqa
from .solution_tracks import SolutionTracks  # noqa
from .tracks_controller import TracksController  # noqa
from .graph_attributes import NodeAttr, EdgeAttr, NodeType  # noqa
from .tracksdata_overwrites import (
    overwrite_graphview_add_node,
    overwrite_graphview_add_edge,
)

# Apply the overwrites to tracksdata's BBoxSpatialFilterView
from tracksdata.graph import GraphView

GraphView.add_node = overwrite_graphview_add_node
GraphView.add_edge = overwrite_graphview_add_edge
