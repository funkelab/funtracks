import numpy as np
import polars as pl
import tracksdata as td
from hoct._dataset import _MEAN, _STD
from hoct.data import DataKeys, item_from_filter
from hoct.data._transforms import Standardize
from hoct.features.constants import REGIONPROPS
from torch.utils.data import Dataset
from tracksdata.nodes import Mask


class HoctDataset(Dataset):
    """Prepare only window-sized tables; never clone the tracking graph."""

    def __init__(self, tracks, window_size):
        self.tracks = tracks
        self.graph = tracks.graph_full
        node_times = self.graph.node_attrs(
            attr_keys=[DataKeys.NODE_ID, tracks.features.time_key]
        )
        time_map = dict(
            zip(
                node_times[DataKeys.NODE_ID].to_list(),
                node_times[tracks.features.time_key].to_list(),
                strict=True,
            )
        )
        times = list(time_map.values())
        # Include both endpoints even for skip edges at the end of the sequence.
        gaps = [time_map[v] - time_map[u] for u, v in self.graph.edge_list()]
        self.window_size = max(window_size, max(gaps, default=0) + 1)
        self.starts = (
            range(min(times), max(min(times) + 1, max(times) + 2 - self.window_size))
            if times
            else range(0)
        )
        self.standardize = Standardize(_MEAN, _STD)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index, extra_edge_attrs=()):
        start = self.starts[index]
        key = self.tracks.features.time_key
        window = self.graph.filter(
            td.NodeAttr(key) >= start,
            td.NodeAttr(key) < start + self.window_size,
        )
        nodes = window.node_attrs()
        if nodes.is_empty():
            return None
        rows = [self._node_features(row) for row in nodes.iter_rows(named=True)]
        prepared = pl.DataFrame(rows).with_columns(
            pl.col("inertia_tensor").cast(pl.Array(pl.Float32, (3, 3)))
        )
        return item_from_filter(
            window,
            ["z", "y", "x"],
            REGIONPROPS,
            [],
            [self.standardize],
            extra_edge_attrs=extra_edge_attrs,
            node_attrs=prepared,
        )

    def _node_features(self, row):
        node = row[DataKeys.NODE_ID]
        position_key = self.tracks.features.position_key
        pos = (
            [row[key] for key in position_key]
            if isinstance(position_key, list)
            else list(row[position_key])
        )
        row_out = {DataKeys.NODE_ID: node, "t": row[self.tracks.features.time_key]}
        row_out.update(
            zip(["z", "y", "x"], [0.0, *pos] if len(pos) == 2 else pos, strict=True)
        )
        row_out.update({key: row.get(key, 0.0) for key in REGIONPROPS})
        tensor = row.get("inertia_tensor")
        if tensor is None:
            row_out["inertia_tensor"] = np.zeros((3, 3)).tolist()
        else:
            tensor = np.asarray(tensor)
            if tensor.shape == (2, 2):
                tensor = np.pad(tensor, ((1, 0), (1, 0)))
            row_out["inertia_tensor"] = tensor.tolist()

        mask = row.get("mask")
        if mask is not None and mask.mask.any():
            # HOCT was trained on 3D regionprops, including singleton-z 2D images.
            spacing = list(self.tracks.scale or [1.0] * self.tracks.ndim)[1:]
            if mask.mask.ndim == 2:
                bbox = np.asarray([0, *mask.bbox[:2], 1, *mask.bbox[2:]])
                mask = Mask(mask.mask[None], bbox)
                spacing = [1.0, *spacing]
            region = mask.regionprops(spacing=tuple(spacing))
            if "equivalent_diameter_area" not in row:
                row_out["equivalent_diameter_area"] = region.equivalent_diameter_area
            if "inertia_tensor" not in row:
                row_out["inertia_tensor"] = region.inertia_tensor.tolist()
        if "border_dist" not in row:
            shape = self.graph.metadata.get("shape")
            if shape is not None:
                scale = np.asarray(self.tracks.scale or [1.0] * self.tracks.ndim)[1:]
                coords = np.asarray(pos) / scale
                distance = np.minimum(coords, np.asarray(shape[1:]) - coords).min()
                row_out["border_dist"] = float(1 - np.clip(distance / 5, 0, 1))
        return row_out
