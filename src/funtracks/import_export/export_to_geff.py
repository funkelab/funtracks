"""Deprecated: Import from funtracks.import_export instead.

This module is deprecated and will be removed in funtracks v2.0.
Use: from funtracks.import_export import export_to_geff

.. deprecated:: 1.0
    This module location is deprecated. Import from the main package instead:
    ``from funtracks.import_export import export_to_geff``
"""

from __future__ import annotations

import warnings

from .geff._export import export_to_geff

warnings.warn(
    "Importing from funtracks.import_export.export_to_geff is deprecated. "
    "Use 'from funtracks.import_export import export_to_geff' instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["export_to_geff"]
