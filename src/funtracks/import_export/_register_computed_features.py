from __future__ import annotations

from funtracks.data_model.tracks import Tracks
from funtracks.import_export._types import ImportedComputedFeature


def _rename_feature(tracks: Tracks, old_key: str, new_key: str) -> None:
    """Rename a feature from old_key to new_key in annotators and features dict.

    Args:
        tracks: Tracks instance to modify
        old_key: Current feature key
        new_key: New feature key
    """
    # Get the feature from the annotator
    feature_dict = tracks.annotators.all_features[old_key][0]

    # Change the annotator key and activate it to ensure recomputation on update
    tracks.annotators.change_key(old_key, new_key)
    tracks.annotators.activate_features([new_key])

    # Register it to the feature dictionary, removing old key if necessary
    if old_key in tracks.features:
        tracks.features.pop(old_key)
    tracks.features[new_key] = feature_dict

    # Update FeatureDict special key attributes if we renamed position or tracklet
    if tracks.features.position_key == old_key:
        tracks.features.position_key = new_key
    if tracks.features.tracklet_key == old_key:
        tracks.features.tracklet_key = new_key


def register_computed_features(
    tracks: Tracks, computed_features: list[ImportedComputedFeature]
) -> None:
    """Register computed features on Tracks, either by recomputing or loading.

    Features can be:
    1. Recomputed from scratch (recompute=True): Enables the feature in annotators
       to compute values for all nodes in the graph.
    2. Loaded without recomputing (recompute=False): Takes existing values from
       graph nodes and registers them as features, renaming if necessary.

    Both types support renaming from the default feature key to a custom prop_name.

    Args:
        tracks: Tracks instance to modify
        computed_features: List of features to register

    Raises:
        ValueError: If a requested feature doesn't exist in annotators or if
                   segmentation is required but not provided.
    """
    # 0) Get a mapping from display name to default key
    default_key_map: dict[str | tuple, str] = {}
    for key, val in tracks.annotators.all_features.items():
        feature = val[0]
        display_name = feature["display_name"]
        if isinstance(display_name, list):
            display_name = tuple(display_name)
        default_key_map[display_name] = key  # type: ignore

    # validate that all features exist, and rename them if they do
    features_to_compute = []
    for imported_feature in computed_features:
        feature_name = imported_feature["feature"]
        new_key = imported_feature["prop_name"]
        recompute = imported_feature["recompute"]

        if (
            feature_name not in default_key_map
            or (feature_default_key := default_key_map[feature_name])
            not in tracks.annotators.all_features
        ):
            raise ValueError(
                f"Requested activation of feature {feature_name} "
                "but no such feature found in computed features. "
                "Perhaps you need to provide a segmentation?"
            )
        _rename_feature(tracks, feature_default_key, new_key)
        if recompute:
            features_to_compute.append(new_key)

    # compute the features that should be recomputed
    tracks.enable_features(features_to_compute)
