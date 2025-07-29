# %%
from funtracks.data_model.tracks import Tracks
import tracksdata as td

# %%

db_path = "/Users/teun.huijben/Downloads/test4d.db"

graph = td.graph.SQLGraph("sqlite", database=db_path)


Tracks_object = Tracks(
    graph = graph,
    ndim = 4,
)

node_ids = Tracks_object.graph.node_ids()
print(len(node_ids))



# %%


# import napari
# import tracksdata as td
# import numpy as np

# viewer = napari.Viewer()

# track_labels = td.array.GraphArrayView(
#     graph, shape=(20, 1, 19991, 15437),
#     attr_key="label", chunk_shape=(1, 2048, 2048), 
#     max_buffers=32, dtype=np.uint64
# )

# viewer.add_labels(track_labels[:,:,4000:5000, 4000:5000], name="track_labels",)