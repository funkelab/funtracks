from __future__ import annotations

import logging
from importlib.metadata import version
from typing import TYPE_CHECKING

import funlib.persistence as fp
import numpy as np
import zarr
from psygnal import Signal

from funtracks.cand_graph import CandGraph
from funtracks.params import CandGraphParams, ProjectParams, SolverParams

from .cand_graph_utils import nodes_from_segmentation
from .tracking_graph import TrackingGraph

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


class Project:
    notify_solution_updated = Signal()

    def __init__(
        self,
        name: str,
        project_params: ProjectParams,
        raw: fp.Array | None = None,
        segmentation: fp.Array | None = None,
        cand_graph: TrackingGraph | None = None,
        zarr_path: Path | None = None,
    ):
        # one of segmentation and points
        if segmentation is None and cand_graph is None:
            raise ValueError(
                "At least one of segmentation or cand graph must be provided"
            )
        self.name = name
        self.params = project_params
        self.solver_params = SolverParams()
        self.raw = raw
        self.segmentation = segmentation
        self.zarr_path = zarr_path
        self.ndim: int
        if self.raw is not None:
            self.ndim = len(self.raw.physical_shape)
        elif segmentation is not None:
            self.ndim = len(self.segmentation.physical_shape)
        else:
            pos_feature = cand_graph.features.position
            if isinstance(pos_feature.value_names, str):
                if len(cand_graph) == 0:
                    raise ValueError("Cannot infer ndim from empty data")
                example_node = next(iter(cand_graph.nodes))
                spatial_dims = len(cand_graph.get_position(example_node))
            else:
                spatial_dims = len(pos_feature.value_names)
            # add 1 for time dimension
            self.ndim = spatial_dims + 1

        cand_graph_params = CandGraphParams()
        if cand_graph is not None:
            self.cand_graph = CandGraph.from_tracking_graph(cand_graph, cand_graph_params)
        else:
            # make a node only cand graph
            self.cand_graph = nodes_from_segmentation(
                self.segmentation, cand_graph_params
            )

    @property
    def solution(self):
        """Re-computing the solution every time you access it is a potential area
        of inefficiency that could be improved if it bottlenecks performance
        """
        return self.cand_graph.get_solution()

    def _save_fp_array(self, path: Path, array: fp.Array):
        raw_ds = fp.prepare_ds(
            path,
            shape=array.shape,
            dtype=array.dtype,
            offset=array.offset,
            voxel_size=array.voxel_size,
            axis_names=array.axis_names,
            units=array.units,
            types=array.types,
        )
        raw_ds[:] = array.data.compute()

    def save(self, zarr_path: Path | None = None):
        """Save the project to the given path. If no path is provided, will try to save
        to the path stored in the project. Path should be a zarr.

        Args:
            path (Path | None, optional): The path to save the project in. Defaults to
                None, which then uses the path attribute of the project.

        Raises:
            ValueError: If the project has no path attribute and no path is provided.
        """
        # TODO: check for valid directory
        if self.zarr_path is None and zarr_path is None:
            raise ValueError("Must provide a path to save the project in")
        elif zarr_path is None:
            zarr_path = self.zarr_path
        zroot = zarr.open(zarr_path, mode="a")
        metadata = {"name": self.name, "version": version("funtracks")}
        params = self.params.model_dump(mode="json")
        zroot.attrs["project_metadata"] = metadata
        zroot.attrs["project_params"] = params
        zroot.attrs["solver_params"] = self.solver_params.model_dump(mode="json")
        # solution is stored as an attribute on cand_graph
        if self.cand_graph is not None:
            self.cand_graph.save(zarr_path / "cand_graph")
        if self.raw is not None and "raw" not in zroot.group_keys():
            self._save_fp_array(zarr_path / "raw", self.raw)
        if self.segmentation is not None and "seg" not in zroot.group_keys():
            self._save_fp_array(zarr_path / "seg", self.segmentation)

    @classmethod
    def load(cls, path: Path):
        zroot = zarr.open(path, mode="a")
        if "cand_graph" in zroot.group_keys():
            cand_graph = CandGraph.load(path / "cand_graph")
        else:
            cand_graph = None
        metadata = zroot.attrs["project_metadata"]
        params_dict = zroot.attrs["project_params"]
        params = ProjectParams(**params_dict)
        name = metadata["name"]
        if "raw" in zroot.array_keys():
            raw = fp.open_ds(path / "raw")
        else:
            raw = None
        if "seg" in zroot.array_keys():
            seg = fp.open_ds(path / "seg")
        else:
            seg = None
        return Project(name, params, raw=raw, segmentation=seg, cand_graph=cand_graph)

    def get_pixels(self, node: int) -> tuple[np.ndarray, ...] | None:
        """Get the pixels corresponding to each node in the nodes list.

        Args:
            nodes (list[Node]): A list of node to get the values for.

        Returns:
            list[tuple[np.ndarray, ...]] | None: A list of tuples, where each tuple
            represents the pixels for one of the input nodes, or None if the segmentation
            is None. The tuple will have length equal to the number of segmentation
            dimensions, and can be used to index the segmentation.
        """
        if self.segmentation is None:
            return None
        time = self.cand_graph.get_time(node)
        loc_pixels = np.nonzero(self.segmentation[time] == node)
        time_array = np.ones_like(loc_pixels[0]) * time
        return (time_array, *loc_pixels)

    def set_pixels(self, pixels: tuple[np.ndarray, ...], value: int):
        """Set the given pixels in the segmentation to the given value.

        Args:
            pixels (Iterable[tuple[np.ndarray]]): The pixels that should be set,
                formatted like the output of np.nonzero (each element of the tuple
                represents one dimension, containing an array of indices in that
                dimension). Can be used to directly index the segmentation.
            value (Iterable[int | None]): The value to set each pixel to
        """
        if self.segmentation is None:
            raise ValueError("Cannot set pixels when segmentation is None")
        if len(self.segmentation.lazy_ops) > 0:
            raise RuntimeError(
                "Segmentation has lazy operations which is not compatible with "
                "fancy indexing: please remove lazy operations from your funlib.persistence.Array"
            )
        self.segmentation[pixels] = value

    def get_next_track_id(self) -> int:
        return self.cand_graph.get_next_track_id()

    def get_next_node_id(self) -> int:
        # TODO
        raise NotImplementedError()
