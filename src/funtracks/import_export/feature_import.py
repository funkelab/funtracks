from __future__ import annotations

from typing import TypedDict, cast

from funtracks.data_model.tracks import Tracks
from funtracks.features import _regionprops_features
from funtracks.features._feature import Feature


class ImportedNodeFeature(TypedDict):
    """Metadata options for an imported property

    Args:
        prop_name (str): the name of the property/attribute on the graph
        feature_name (str | None): the display name of a computed feature, or None if the
            feature is static.
        recompute (bool): indicates whether to recompute the computed feature or load it
            as is. Only used for computed features with a provided `feature_name`.
        dtype (str): the dtype of the feature. `bool` features will be interpreted
            as groups.
    """

    prop_name: str
    feature: str | None
    recompute: bool
    dtype: str


def get_regionprop_dict(name: str):
    """Get the TypedDict for the requested RegionpropFeature"""

    func = getattr(_regionprops_features, name, None)
    if func is None:
        raise ValueError(f"Regionprop feature {name} not found")
    return func


def register_features(tracks: Tracks, node_features: list[ImportedNodeFeature]) -> None:
    """Function to set up imported properties as Features on Tracks.

    1) Create a list of all to be recomputed features, and enable them on
        Tracks.
    2) Create a list of all features that should be interpreted as computed features,
        without immediate recomputation. Since they are imported from a node property that
        can have any custom name in the source CSV or Geff file, we update the
        key for this Feature in the FeatureDict to the given name.
    3) Create a list of all remaining, static features and add
        them to the FeatureDict with their respective data types.

    Args:
        tracks (Tracks): the Tracks instance to be modified.
        node_features (list[ImportedNodeFeature]): list of to be imported Features.
    Raises:
        ValueError when attempting to register computed features without a segmentation.
    """

    # 1) Recompute requested features
    features_to_compute = [
        feature for feature in node_features if bool(feature.get("recompute", False))
    ]
    default_name_map: dict[str | tuple, str] = {}
    for key, val in tracks.annotators.all_features.items():
        feature = val[0]
        display_name = feature["display_name"]
        if isinstance(display_name, list):
            display_name = tuple(display_name)
        default_name = key if display_name is None else display_name
        default_name_map[default_name] = key  # type: ignore

    features_to_compute_names = []
    for imported_feature in features_to_compute:
        feature_name = imported_feature["feature"]
        if (
            feature_name not in default_name_map
            or (feature_default_key := default_name_map[feature_name])
            not in tracks.annotators.all_features
        ):
            raise ValueError(
                f"Requested computation of feature {feature_name} "
                "but no such feature found in computed features. "
                "Perhaps you need to provide a segmentation?"
            )
        features_to_compute_names.append(feature_default_key)

    tracks.enable_features(features_to_compute_names)

    # 2) Add Regionprops features that were loaded from source but should not be
    # recomputed now.
    features_to_enable = [
        feature
        for feature in node_features
        if (not bool(feature.get("recompute", False)) and feature["feature"] is not None)
    ]

    for imported_feature in features_to_enable:
        new_key = imported_feature["prop_name"]
        feature_name = imported_feature["feature"]
        if feature_name not in default_name_map:
            raise ValueError(
                f"Requested activation of feature {feature_name} "
                "but no such feature found in computed features. "
                "Perhaps you need to provide a segmentation?"
            )

        feature_default_key = default_name_map[feature_name]
        if feature_default_key not in tracks.annotators.all_features:
            raise ValueError(
                f"Requested activation of feature {feature_default_key} "
                "but no such feature found in computed features. "
                "Perhaps you need to provide a segmentation?"
            )

        # Get the feature from the annotator
        feature_dict = tracks.annotators.all_features[feature_default_key][0]

        # change the annotator key and activate it to ensure recomputation on update
        tracks.annotators.change_key(feature_default_key, new_key)
        tracks.annotators.activate_features([new_key])

        # Register it to the feature dictionary, removing old key if necessary
        if feature_default_key in tracks.features:
            tracks.features.pop(feature_default_key)
        tracks.features[new_key] = feature_dict

        # Update FeatureDict special key attributes if we renamed position or tracklet
        if tracks.features.position_key == feature_default_key:
            tracks.features.position_key = new_key
        if tracks.features.tracklet_key == feature_default_key:
            tracks.features.tracklet_key = new_key

    # 3) Add other (custom, group) features that will be static
    other_features = [f for f in node_features if f["feature"] is None]
    for imported_feature in other_features:
        # ensure dtype and prop_name are strings for the Feature TypedDict
        dtype_str = str(imported_feature.get("dtype", "str"))
        display_name = str(imported_feature.get("prop_name"))
        new_feature = {
            "feature_type": "node",
            "value_type": dtype_str,
            "num_values": 1,
            "display_name": display_name,
            "required": False,
            "default_value": None,
        }
        # cast to Feature TypedDict
        tracks.features[str(imported_feature["prop_name"])] = cast(Feature, new_feature)
