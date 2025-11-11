from __future__ import annotations

import inspect
from typing import TypedDict, cast

from funtracks.data_model.tracks import Tracks
from funtracks.features import _regionprops_features
from funtracks.features._feature import Feature

# TODO: compute this from the actual AnnotatorRegistry of the Tracks?
default_name_map: dict[str, str] = {
    "Area": "area",
    "Circularity": "circularity",
    "Perimeter": "perimeter",
    "EllipsoidAxes": "ellipse_axis_radii",
}


class ImportedNodeFeature(TypedDict):
    """Metadata options for an imported property

    Args:
        prop_name (str): the name of the property in the source CSV or Geff file
        feature (str): the name of the feature, this can either be the DISPLAY? name of a
            computed Feature, or it can be 'Group' (will be imported as a boolean type
            to reflect a group), or 'Custom' (static feature).
        recompute (bool): indicates whether to recompute the computed feature or load it
            as is.
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

    1) Create a list of all to be recomputed features, and enable them on
        Tracks.
    2) Create a list of all features that should be interpreted as computed features,
        without immediate recomputation. Since they are imported from a node property that
        can have any custom name in the source CSV or Geff file, we update the
        key for this Feature in the FeatureDict to the given name.
    3) Create a list of all remaining, static features (Custom/Group features) and add
        them to the Features dictionary with their respective data types.

    Args:
        tracks (Tracks): the to be modified Tracks instance.
        node_features (list[ImportedNodeFeature]): list of to be imported Features.
    Raises:
        ValueError when attempting to register computed features without a segmentation.
    """

    # 1) Recompute requested Regionprops features
    features_to_compute = [
        feature for feature in node_features if bool(feature.get("recompute", False))
    ]

    features_to_compute_names = []
    for feature in features_to_compute:
        if feature["feature"] not in default_name_map:
            raise ValueError(f"Cannot compute unknown feature {feature['feature']}")
        else:
            features_to_compute_names.append(default_name_map[feature["feature"]])

    if len(features_to_compute_names) > 0:
        if tracks.segmentation is None:
            raise ValueError(
                f"Please provide a segmentation to compute Regionprops features: "
                f"{features_to_compute_names}"
            )
        tracks.enable_features(features_to_compute_names)

    # 2) Add Regionprops features that were loaded from source but should not be
    # recomputed now.

    regionprop_features = [
        feature
        for feature in node_features
        if not bool(feature.get("recompute", False))
        and feature["feature"] not in ("Group", "Custom")
    ]

    if len(regionprop_features) > 0 and tracks.segmentation is None:
        raise ValueError(
            "Please provide a segmentation to compute Regionprops features: "
            f"{[feature['feature'] for feature in regionprop_features]}"
        )

    for feature in regionprop_features:
        given_name = feature["prop_name"]
        regionprop_name = feature["feature"]
        regionprop_key = default_name_map[regionprop_name]
        if regionprop_key is None:
            raise ValueError(f"Cannot compute unknown feature {regionprop_name}")

        # Call the regionprop function (check if it needs ndim)
        regionprop_func = get_regionprop_dict(regionprop_name)
        sig = inspect.signature(regionprop_func)
        if "ndim" in sig.parameters:
            regionprop_feature = regionprop_func(ndim=tracks.ndim)
        else:
            regionprop_feature = regionprop_func()

        # Register it to the annotator, first under the default name and then update
        # it to its given name using the registry's change_key method.
        tracks.features[regionprop_key] = regionprop_feature
        tracks.annotators.activate_features([regionprop_key])
        tracks.annotators.change_key(regionprop_key, given_name)
        tracks.features[given_name] = tracks.features.pop(regionprop_key)

        # Update FeatureDict special key attributes if we renamed position or tracklet
        if tracks.features.position_key == regionprop_key:
            tracks.features.position_key = given_name
        if tracks.features.tracklet_key == regionprop_key:
            tracks.features.tracklet_key = given_name

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
