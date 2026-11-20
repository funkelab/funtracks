"""Turn basic actions into JSON-ready dicts and back, for the history log.

Each basic action that can be logged implements ``to_dict`` and ``from_dict``.
This module holds what they share: the value codec, the registry of loggable
actions, and the summary of user-action arguments.

Value encoding (see the history log docs for the full format):

- JSON scalars stay as they are; numpy scalars become Python scalars.
- numpy arrays are always tagged, so they decode back to arrays of the same dtype:
  ``{"__a__": [...], "dtype": "float64"}`` for 1-D arrays of at most 16 values,
  ``{"__nd__": {"dtype", "shape", "data"}}`` (base64 of the zlib-compressed bytes)
  for the rest, and ``{"__objarr__": [...], "shape": [...]}`` for object arrays.
- Masks become ``{"__mask__": {"bbox", "shape", "bits"}}``, with the pixels packed
  to bits and compressed.
- Lists and tuples become JSON lists. Dict keys become strings.
"""

from __future__ import annotations

import base64
import zlib
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
from tracksdata.nodes import Mask

if TYPE_CHECKING:
    from funtracks.data_model import Tracks

    from ._base import BasicAction

# 1-D numeric arrays up to this length are written as readable lists
_SHORT_ARRAY = 16
# Longer lists in a user action's summary are shortened to their length and start
_LONG_LIST = 50


def _pack(data: bytes) -> str:
    return base64.b64encode(zlib.compress(data)).decode("ascii")


def _unpack(text: str) -> bytes:
    return zlib.decompress(base64.b64decode(text))


def encode_value(value: Any) -> Any:
    """Encode an attribute value as something ``json.dumps`` can write.

    Args:
        value: The value to encode.

    Returns:
        Any: A JSON-ready value that ``decode_value`` turns back into ``value``.
    """
    if isinstance(value, Mask):
        mask = np.asarray(value.mask, dtype=bool)
        return {
            "__mask__": {
                "bbox": [int(v) for v in np.asarray(value.bbox).ravel()],
                "shape": list(mask.shape),
                "bits": _pack(np.packbits(mask.ravel()).tobytes()),
            }
        }
    if isinstance(value, pl.Series):
        value = value.to_numpy()
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            return {
                "__objarr__": [encode_value(v) for v in value.ravel()],
                "shape": list(value.shape),
            }
        if value.ndim == 1 and value.size <= _SHORT_ARRAY:
            return {"__a__": value.tolist(), "dtype": value.dtype.name}
        return {
            "__nd__": {
                # dtype.str includes the byte order, which the raw bytes depend on
                "dtype": value.dtype.str,
                "shape": list(value.shape),
                "data": _pack(np.ascontiguousarray(value).tobytes()),
            }
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, list | tuple):
        return [encode_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): encode_value(v) for k, v in value.items()}
    return value


def decode_value(value: Any) -> Any:
    """Decode a value written by ``encode_value``.

    Args:
        value: The JSON value to decode.

    Returns:
        Any: The decoded value. Tuples come back as lists.
    """
    if isinstance(value, list):
        return [decode_value(v) for v in value]
    if not isinstance(value, dict):
        return value
    if "__mask__" in value:
        d = value["__mask__"]
        shape = tuple(d["shape"])
        n_pixels = int(np.prod(shape))
        bits = np.frombuffer(_unpack(d["bits"]), dtype=np.uint8)
        mask = np.unpackbits(bits, count=n_pixels).astype(bool).reshape(shape)
        return Mask(mask, bbox=np.asarray(d["bbox"], dtype=np.int64))
    if "__a__" in value:
        return np.asarray(value["__a__"], dtype=value["dtype"])
    if "__nd__" in value:
        d = value["__nd__"]
        array = np.frombuffer(_unpack(d["data"]), dtype=np.dtype(d["dtype"]))
        return array.reshape(d["shape"]).copy()
    if "__objarr__" in value:
        items = [decode_value(v) for v in value["__objarr__"]]
        array = np.empty(len(items), dtype=object)
        array[:] = items
        return array.reshape(value["shape"])
    return {k: decode_value(v) for k, v in value.items()}


def summarize_value(value: Any) -> Any:
    """Summarize a user-action argument for the human-readable log ``params``.

    Unlike ``encode_value``, this is lossy: masks and pixel coordinates become their
    bounding box and number of pixels, large arrays their shape and dtype, and lists
    of more than 50 items their length and first 5 items.

    Args:
        value: The argument to summarize.

    Returns:
        Any: A small JSON-ready value.
    """
    if isinstance(value, Mask):
        bbox = [int(v) for v in np.asarray(value.bbox).ravel()]
        return {"bbox": bbox, "n_voxels": int(np.count_nonzero(value.mask))}
    if _is_pixel_coordinates(value):
        coords = np.stack([np.asarray(c) for c in value])
        bbox = coords.min(axis=1).tolist() + (coords.max(axis=1) + 1).tolist()
        return {"bbox": bbox, "n_voxels": int(coords.shape[1])}
    if isinstance(value, pl.Series):
        value = value.to_numpy()
    if isinstance(value, np.ndarray):
        if value.ndim == 1 and value.size <= _SHORT_ARRAY and value.dtype != object:
            return value.tolist()
        return {"shape": list(value.shape), "dtype": value.dtype.name}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, list | tuple | set):
        items = list(value)
        if len(items) > _LONG_LIST:
            # e.g. deleting thousands of nodes: the ids are in the basic actions anyway
            return {"n": len(items), "first": [summarize_value(v) for v in items[:5]]}
        return [summarize_value(v) for v in items]
    if isinstance(value, dict):
        return {str(k): summarize_value(v) for k, v in value.items()}
    if value is None or isinstance(value, bool | int | float | str):
        return value
    return repr(value)


def _is_pixel_coordinates(value: Any) -> bool:
    """True for a tuple of equally long 1-D integer arrays, one per dimension."""
    if not isinstance(value, tuple) or len(value) < 2:
        return False
    if not all(isinstance(c, np.ndarray) and c.ndim == 1 for c in value):
        return False
    return len({c.size for c in value}) == 1 and all(
        np.issubdtype(c.dtype, np.integer) for c in value
    )


def action_registry() -> dict[str, type[BasicAction]]:
    """The basic actions that can be written to and replayed from the log.

    Keyed by class name. ``UpdateTrackIDs`` is deliberately not in it: track ids are
    not logged, but recomputed from the edges on replay.

    Returns:
        dict[str, type[BasicAction]]: The loggable basic actions by name.
    """
    from .add_delete_edge import AddEdge, DeleteEdge
    from .add_delete_feature import AddFeature, DeleteFeature
    from .add_delete_node import AddNode, DeleteNode
    from .update_node_attrs import UpdateNodeAttrs
    from .update_segmentation import UpdateNodeSeg

    classes: list[type[BasicAction]] = [
        AddNode,
        DeleteNode,
        AddEdge,
        DeleteEdge,
        UpdateNodeAttrs,
        UpdateNodeSeg,
        AddFeature,
        DeleteFeature,
    ]
    return {cls.__name__: cls for cls in classes}


def action_from_dict(tracks: Tracks, d: dict[str, Any]) -> BasicAction:
    """Apply the basic action described by ``d`` (from ``to_dict``) to the tracks.

    Args:
        tracks: The tracks to apply the action to.
        d: The action dict, with the class name under ``"a"``.

    Returns:
        BasicAction: The applied action.

    Raises:
        ValueError: If the action name is unknown.
    """
    registry = action_registry()
    name = d.get("a")
    if name not in registry:
        raise ValueError(f"Unknown action {name!r} in history log")
    return registry[name].from_dict(tracks, d)


def derived_keys(tracks: Tracks) -> set[str]:
    """Attribute keys that are recomputed on replay and so are not logged.

    These are the features computed by the registered annotators, which include the
    tracklet and lineage ids.

    Args:
        tracks: The tracks the actions belong to.

    Returns:
        set[str]: The keys to leave out of logged attributes.
    """
    keys = set(tracks.annotators.all_features)
    for key in (tracks.features.tracklet_key, tracks.features.lineage_key):
        if key is not None:
            keys.add(key)
    return keys
