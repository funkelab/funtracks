"""Deprecated: Import from funtracks.import_export instead.

This module is deprecated and will be removed in funtracks v2.0.
Use: from funtracks.import_export import import_from_geff

.. deprecated:: 1.0
    This module location is deprecated. Import from the main package instead:
    ``from funtracks.import_export import import_from_geff``
"""

from __future__ import annotations

import warnings

from .geff._import import import_from_geff, import_graph_from_geff

warnings.warn(
    "Importing from funtracks.import_export.import_from_geff is deprecated. "
    "Use 'from funtracks.import_export import import_from_geff' instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["import_from_geff", "import_graph_from_geff"]
