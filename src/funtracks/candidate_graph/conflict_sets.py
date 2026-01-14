from itertools import combinations

import numpy as np


def compute_conflict_sets(segmentation_frame: np.ndarray) -> list[set]:
    """Compute all sets of node ids that conflict with each other.
    Note: Results might include redundant sets, for example {a, b, c} and {a, b}
    might both appear in the results.

    Args:
        segmentation_frame (np.ndarray):  One frame of the multiple hypothesis
            segmentation. Dimensions are (h, [z], y, x), where h is the number of
            hypotheses.
        time (int): Time frame, for computing node_ids.

    Returns:
        list[set]: list of sets of node ids that overlap. Might include some sets
            that are subsets of others.
    """
    flattened_segs = [seg.flatten() for seg in segmentation_frame]

    # get locations where at least two hypotheses have labels
    # This approach may be inefficient, but likely doesn't matter compared to np.unique
    conflict_indices = np.zeros(flattened_segs[0].shape, dtype=bool)
    for seg1, seg2 in combinations(flattened_segs, 2):
        non_zero_indices = np.logical_and(seg1, seg2)
        conflict_indices = np.logical_or(conflict_indices, non_zero_indices)

    flattened_stacked = np.array([seg[conflict_indices] for seg in flattened_segs])
    values = np.unique(flattened_stacked, axis=1)
    values = np.transpose(values)
    conflict_sets = []
    for conflicting_labels in values:
        id_set = {label for label in conflicting_labels if label != 0}
        conflict_sets.append(id_set)
    return conflict_sets
