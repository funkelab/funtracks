"""Test that multiple RegionpropsAnnotators with different mask_attr
respond to correct actions."""

import numpy as np
import polars as pl
import pytest
import tracksdata as td
from tracksdata.nodes import Mask

from funtracks.actions import UpdateNodeSeg
from funtracks.annotators import RegionpropsAnnotator
from funtracks.data_model import Tracks


@pytest.mark.parametrize("ndim", [3, 4])
class TestMultiAnnotatorUpdate:
    def test_update_different_masks(self, get_graph, ndim):
        """Test updates to different mask attributes trigger only
        the correct annotator."""
        graph = get_graph(ndim, with_seg=True)

        # Add nuclear mask attribute
        graph.add_node_attr_key("nuc_mask", pl.Object)
        spatial_ndim = ndim - 1
        graph.add_node_attr_key(
            "nuc_bbox",
            pl.Array(pl.Int64, 2 * spatial_ndim),
        )

        # Copy existing masks to the nuclear attribute (smaller versions)
        for node_id in graph.node_ids():
            mem_mask = graph.nodes[node_id][td.DEFAULT_ATTR_KEYS.MASK]
            # Create a smaller nuclear mask (downsample by 2)
            if spatial_ndim == 2:
                nuc_arr = mem_mask.mask[::2, ::2]
                nuc_bbox = np.array(
                    [
                        mem_mask.bbox[0] // 2,
                        mem_mask.bbox[1] // 2,
                        mem_mask.bbox[0] // 2 + nuc_arr.shape[0],
                        mem_mask.bbox[1] // 2 + nuc_arr.shape[1],
                    ]
                )
            else:
                nuc_arr = mem_mask.mask[::2, ::2, ::2]
                nuc_bbox = np.array(
                    [
                        mem_mask.bbox[0] // 2,
                        mem_mask.bbox[1] // 2,
                        mem_mask.bbox[2] // 2,
                        mem_mask.bbox[0] // 2 + nuc_arr.shape[0],
                        mem_mask.bbox[1] // 2 + nuc_arr.shape[1],
                        mem_mask.bbox[2] // 2 + nuc_arr.shape[2],
                    ]
                )
            nuc_mask = Mask(nuc_arr, bbox=nuc_bbox)
            graph.update_node_attrs(
                attrs={"nuc_mask": [nuc_mask], "nuc_bbox": [nuc_mask.bbox]},
                node_ids=[node_id],
            )

        tracks = Tracks(graph, ndim=ndim, time_attr="t", pos_attr="pos")

        # Create two RegionpropsAnnotators
        mem_annotator = RegionpropsAnnotator(tracks, mask_attr=td.DEFAULT_ATTR_KEYS.MASK)
        nuc_annotator = RegionpropsAnnotator(
            tracks, mask_attr="nuc_mask", key_prefix="nuc_"
        )

        # Add both to tracks
        for key, (feat, _) in mem_annotator.all_features.items():
            tracks.add_feature(key, feat)
        mem_annotator.activate_features(list(mem_annotator.all_features.keys()))
        tracks.annotators.append(mem_annotator)

        for key, (feat, _) in nuc_annotator.all_features.items():
            tracks.add_feature(key, feat)
        nuc_annotator.activate_features(list(nuc_annotator.all_features.keys()))
        tracks.annotators.append(nuc_annotator)

        # Compute initial areas
        mem_annotator.compute(["area"])
        nuc_annotator.compute(["nuc_area"])

        # Get a test node
        test_node = list(graph.node_ids())[0]

        initial_mem_area = tracks.get_node_attr(test_node, "area")
        initial_nuc_area = tracks.get_node_attr(test_node, "nuc_area")

        assert initial_mem_area is not None
        assert initial_nuc_area is not None
        assert initial_mem_area > initial_nuc_area  # Membrane should be bigger

        # Update the membrane mask - should only trigger mem_annotator
        mem_mask = graph.nodes[test_node][td.DEFAULT_ATTR_KEYS.MASK]
        # Create a larger mask by expanding
        if spatial_ndim == 2:
            new_shape = (mem_mask.mask.shape[0] + 4, mem_mask.mask.shape[1] + 4)
            bbox_delta = np.array([-2, -2, 2, 2])
        else:
            new_shape = (
                mem_mask.mask.shape[0] + 4,
                mem_mask.mask.shape[1] + 4,
                mem_mask.mask.shape[2] + 4,
            )
            bbox_delta = np.array([-2, -2, -2, 2, 2, 2])

        new_mem_arr = np.ones(new_shape, dtype=bool)
        new_mem_bbox = mem_mask.bbox + bbox_delta
        new_mem_mask = Mask(new_mem_arr, bbox=new_mem_bbox)

        UpdateNodeSeg(
            tracks,
            node=test_node,
            mask=new_mem_mask,
            mask_key=td.DEFAULT_ATTR_KEYS.MASK,
        )

        mem_area_after = tracks.get_node_attr(test_node, "area")
        nuc_area_after = tracks.get_node_attr(test_node, "nuc_area")

        # Membrane area should have changed, nuclear should not
        assert mem_area_after > initial_mem_area, (
            "Membrane area should increase after membrane update"
        )
        assert nuc_area_after == initial_nuc_area, (
            "Nuclear area should NOT change after membrane update"
        )

        # Update the nuclear mask - should only trigger nuc_annotator
        nuc_mask = graph.nodes[test_node]["nuc_mask"]
        if spatial_ndim == 2:
            new_shape = (nuc_mask.mask.shape[0] + 2, nuc_mask.mask.shape[1] + 2)
            bbox_delta = np.array([-1, -1, 1, 1])
        else:
            new_shape = (
                nuc_mask.mask.shape[0] + 2,
                nuc_mask.mask.shape[1] + 2,
                nuc_mask.mask.shape[2] + 2,
            )
            bbox_delta = np.array([-1, -1, -1, 1, 1, 1])

        new_nuc_arr = np.ones(new_shape, dtype=bool)
        new_nuc_bbox = nuc_mask.bbox + bbox_delta
        new_nuc_mask = Mask(new_nuc_arr, bbox=new_nuc_bbox)

        UpdateNodeSeg(
            tracks,
            node=test_node,
            mask=new_nuc_mask,
            mask_key="nuc_mask",
        )

        mem_area_final = tracks.get_node_attr(test_node, "area")
        nuc_area_final = tracks.get_node_attr(test_node, "nuc_area")

        # Membrane area should NOT have changed, nuclear should have
        assert mem_area_final == mem_area_after, (
            "Membrane area should NOT change after nuclear update"
        )
        assert nuc_area_final > nuc_area_after, (
            "Nuclear area should increase after nuclear update"
        )
