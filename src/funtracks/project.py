import json
import logging
from importlib.metadata import version

import funlib.persistence as fp
import numpy as np
from psygnal import Signal

from funtracks.params.project_params import ProjectParams

from .cand_graph_utils import nodes_from_segmentation
from .tracking_graph import TrackingGraph

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
    ):
        # one of segmentation and points
        if segmentation is None and cand_graph is None:
            raise ValueError(
                "At least one of segmentation or cand graph must be provided"
            )

        self.name = name
        self.params = project_params
        self.raw = raw
        self.segmentation = segmentation

        if cand_graph is not None:
            self.cand_graph = cand_graph
        else:
            # make a node only cand graph
            self.cand_graph = nodes_from_segmentation(self.segmentation)
        self._solution = self.cand_graph.get_solution()

    @property
    def solution(self):
        return self._solution

    def save(self, path):
        # solution is stored as an attribute on cand_graph
        if self.cand_graph is not None:
            self.cand_graph.save(path / "cand_graph")
        if self.solution is not None:
            self.solution.save(path)
        self.params.save(path / "params.json")
        self._save_metadata(path / "metadata.json")

    def _save_metadata(self, path):
        metadata = {"name": self.name, "version": version("motile_tracker")}
        with open(path, "w") as f:
            json.dump(metadata, f)

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
        self.segmentation[pixels] = value

    def get_next_track_id(self) -> int:
        # TODO
        raise NotImplementedError()

    def get_next_node_id(self) -> int:
        # TODO
        raise NotImplementedError()
