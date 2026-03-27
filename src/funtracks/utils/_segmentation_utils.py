import numpy as np
import tracksdata as td


def relabel_segmentation_with_track_id(
    solution_graph: td.graph.GraphView,
    segmentation: np.ndarray,
) -> np.ndarray:
    """Relabel a segmentation based on tracking results so that nodes in same
    track share the same id. IDs do change at division.

    Args:
        solution_graph (td.graph.GraphView): Tracksdata graph with the solution to use
            for relabeling. Nodes not in graph will be set to 0 in the output.
            Node IDs must match the segmentation labels.
        segmentation (np.ndarray): Original segmentation with dimensions (t, [z], y, x)

    Returns:
        np.ndarray: Relabeled segmentation array where nodes in same track share same
            id with shape (t,[z],y,x)
    """
    tracked_masks = np.zeros_like(segmentation)
    id_counter = 1

    # Division nodes have out_degree > 1; their outgoing edges are cut so that
    # each daughter cell starts a new tracklet
    division_nodes = {
        n for n in solution_graph.node_ids() if solution_graph.out_degree(n) > 1
    }

    visited: set = set()
    for start_node in solution_graph.node_ids():
        if start_node in visited:
            continue

        # BFS to collect the connected tracklet, cutting at division out-edges
        component: set = set()
        queue = [start_node]
        while queue:
            node = queue.pop()
            if node in visited:
                continue
            visited.add(node)
            component.add(node)
            # Traverse predecessors: only if the predecessor is not a division node
            # (because a division node's out-edges are cut)
            for pred in solution_graph.predecessors(node):
                if pred not in visited and pred not in division_nodes:
                    queue.append(pred)
            # Traverse successors: only if the current node is not a division node
            if node not in division_nodes:
                for succ in solution_graph.successors(node):
                    if succ not in visited:
                        queue.append(succ)

        for node in component:
            t = int(solution_graph.nodes[node]["t"])
            tracked_masks[t][segmentation[t] == node] = id_counter
        id_counter += 1

    return tracked_masks


def ensure_unique_labels(
    segmentation: np.ndarray,
    multiseg: bool = False,
) -> np.ndarray:
    """Relabels the segmentation in place to ensure that label ids are unique across
    time. This means that every detection will have a unique label id.
    Useful for combining predictions made in each frame independently, or multiple
    segmentation outputs that repeat label IDs.

    Args:
        segmentation (np.ndarray): Segmentation with dimensions ([h], t, [z], y, x).
        multiseg (bool, optional): Flag indicating if the segmentation contains
            multiple hypotheses in the first dimension. Defaults to False.
    """
    segmentation = segmentation.astype(np.uint64)
    orig_shape = segmentation.shape
    if multiseg:
        segmentation = segmentation.reshape((-1, *orig_shape[2:]))
    curr_max = 0
    for idx in range(segmentation.shape[0]):
        frame = segmentation[idx]
        frame[frame != 0] += curr_max
        curr_max = int(np.max(frame))
        segmentation[idx] = frame
    if multiseg:
        segmentation = segmentation.reshape(orig_shape)
    return segmentation
