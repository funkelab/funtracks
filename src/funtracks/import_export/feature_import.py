from __future__ import annotations

import inspect
import warnings
from typing import TypedDict, cast

from funtracks.annotators._regionprops_annotator import (
    RegionpropsAnnotator,
)
from funtracks.data_model.tracks import Tracks
from funtracks.features import _regionprops_features
from funtracks.features._feature import Feature

regionprop_class_names = [
    name
    for name, func in inspect.getmembers(_regionprops_features, inspect.isfunction)
    if func.__module__ == "funtracks.features._regionprops_features"
]


default_name_map: dict[str, str] = {
    "Area": "area",
    "Circularity": "circularity",
    "Perimeter": "perimeter",
    "EllipsoidAxes": "ellipse_axis_radii",
}


class ImportedNodeFeature(TypedDict):
    """a dictionary mapping for an imported property, with the following keys:
    "prop_name", "feature", "recompute", "dtype". The prop_name is the name of the
    property in the source CSV or Geff file, feature is the name of the feature, this can
    either be the class name of a Regionprops Feature, or it can be 'Group'
    (will be imported as a boolean type to reflect a group), or 'Custom' (static feature).
     Recompute is a boolean that indicates whether to recompute the regionprops feature
    or load it as is.
    """

    prop_name: str
    feature: str
    recompute: bool
    dtype: str


def get_regionprop_dict(name: str):
    """Get the TypedDict for the requested RegionpropFeature"""

    func = getattr(_regionprops_features, name, None)
    if func is None:
        raise ValueError(f"Regionprop feature {name} not found")
    return func


def register_features(tracks: Tracks, node_features: list[ImportedNodeFeature]) -> None:
    """Function to register imported properties as Features on Tracks.

    1) Create a list of all to be recomputed regionprops features, and enable them on
        Tracks.
    2) Create a list of all features that should be interpreted as Regionprops features,
        without immediate recomputation. Since they are imported from a node property that
         can have any custom name in the source CSV or Geff file, we update the default
         key for this Regionprops Feature in the Features dictionary to the given name.
    3) Create a list of all remaining, static features (Custom/Group features) and add
     them to the Features dictionary with their respective data types.

    Args:
        tracks (Tracks): the to be modified Tracks instance.
        node_features (list[ImportedNodeFeature]): list of to be imported Features.
    """

    # 1) Recompute requested Regionprops features
    features_to_compute = [
        feature for feature in node_features if bool(feature.get("recompute", False))
    ]
    features_to_compute_names = [
        default_name_map[f["feature"]]
        for f in features_to_compute
        if f["feature"] in default_name_map
    ]
    if len(features_to_compute_names) > 0 and tracks.segmentation is not None:
        tracks.enable_features(features_to_compute_names)

    # 2) Add Regionprops features that were loaded from source but should not be
    # recomputed now.
    if tracks.segmentation is not None:
        regionprop_features = [
            feature
            for feature in node_features
            if not bool(feature.get("recompute", False))
            and feature["feature"] not in ("Group", "Custom")
        ]

        for feature in regionprop_features:
            given_name = feature["prop_name"]
            regionprop_name = feature["feature"]
            regionprop_key = default_name_map[regionprop_name]
            if regionprop_key is None:
                warnings.warn(
                    f"Unknown regionprop feature: {regionprop_name}", stacklevel=2
                )
                continue

            # Call the regionprop function (check if it needs ndim)
            regionprop_func = get_regionprop_dict(regionprop_name)
            sig = inspect.signature(regionprop_func)
            if "ndim" in sig.parameters:
                regionprop_feature = regionprop_func(ndim=tracks.ndim)
            else:
                regionprop_feature = regionprop_func()

            # Register it to the annotator, first under the default name and then update
            # it to its given name.
            tracks.features[regionprop_key] = regionprop_feature
            tracks.annotators.activate_features([regionprop_key])
            regionprops_annotator = next(
                annotator
                for annotator in tracks.annotators
                if isinstance(annotator, RegionpropsAnnotator)
            )
            regionprops_annotator.change_key(regionprop_key, given_name)
            tracks.features[given_name] = tracks.features.pop(regionprop_key)

    # 3) Add other (custom, group) features that will be static
    other_features = [f for f in node_features if f["feature"] in ("Custom", "Group")]
    for feature in other_features:
        # ensure dtype and prop_name are strings for the Feature TypedDict
        dtype_str = str(feature.get("dtype", "str"))
        display_name = str(feature.get("prop_name"))
        new_feature = {
            "feature_type": "node",
            "value_type": dtype_str,
            "num_values": 1,
            "display_name": display_name,
            "required": False,
            "default_value": None,
        }
        # cast to Feature TypedDict
        tracks.features[str(feature["prop_name"])] = cast(Feature, new_feature)
