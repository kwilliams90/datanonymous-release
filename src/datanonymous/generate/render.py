"""Render generated values in the source's exact text forms (FOUNDATION; shared file).

:func:`render_values` dispatches by type to the renderer in the type's own
module. Level-quota types, invented labels, lorem, masks and IDs are produced
already rendered (exact raw strings: boolean tokens, whitespace variants such
as ``'Central '``, zero padding), so their renderer is the identity. Missing
slots are filled with the real NA tokens verbatim by ``generate/table.py``.

=================================================  ===================================  ====
types                                              renderer                             area
=================================================  ===================================  ====
INTEGER (continuous), FLOAT, LAT, LON              generate.numeric.render_numeric      FOUNDATION
NUMERIC_FORMATTED, NUMERIC_WITH_TOKENS, DECIMAL    generate.numeric.render_numeric_extended  B
DATE, DATE_MIXED_FORMATS, DATE_AS_INT, DATETIME,   generate.dates.render_dates          A
DATETIME_TZ, YEAR_MONTH, TIME, DURATION
everything else                                    identity                             -
=================================================  ===================================  ====
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from ..model import TYPE_INFO, ColumnProfile, ColumnType
from . import dates, numeric


def _identity(profile: ColumnProfile, values: np.ndarray) -> np.ndarray:
    return values


_B_TYPES = (ColumnType.NUMERIC_FORMATTED, ColumnType.NUMERIC_WITH_TOKENS, ColumnType.DECIMAL)


def renderer_for(profile: ColumnProfile) -> Callable[[ColumnProfile, np.ndarray], np.ndarray]:
    t = profile.type
    if t in (ColumnType.FLOAT, ColumnType.LAT, ColumnType.LON) or (
        t is ColumnType.INTEGER and profile.subtype == "continuous"
    ):
        return numeric.render_numeric
    if t in _B_TYPES:
        return numeric.render_numeric_extended
    if TYPE_INFO[t].family == "date":
        return dates.render_dates
    return _identity


def render_values(profile: ColumnProfile, values: np.ndarray) -> np.ndarray:
    """Rendered strings (object array) for generated non-missing values."""
    return renderer_for(profile)(profile, values)
