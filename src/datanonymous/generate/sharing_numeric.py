"""Exact numeric exclusions for sharing generation and final-value verification."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from typing import Any

import numpy as np

from ..errors import GenerationError
from ..model import ColumnModel, ColumnProfile, ColumnType as T
from .numeric import exact_value
from .quota import largest_remainder

TYPES = frozenset({T.INTEGER, T.INTEGER_YEAR, T.FLOAT, T.DECIMAL, T.NUMERIC_FORMATTED,
                   T.NUMERIC_WITH_TOKENS, T.LAT, T.LON})
MARKER = "_sharing_numeric_extremes"
FORBIDDEN_KEY = "_sharing_forbidden_numeric_extremes"


def discrete(profile: ColumnProfile) -> bool:
    return (profile.type is T.INTEGER and profile.subtype == "discrete" or profile.type is T.INTEGER_YEAR
            or profile.type in (T.NUMERIC_FORMATTED, T.NUMERIC_WITH_TOKENS) and bool(profile.params.get("integer_valued"))
            or profile.type is T.DECIMAL and int(profile.params.get("scale", 0)) <= 0)


def normalized_key(profile: ColumnProfile, raw: Any) -> str | None:
    # an already parsed value is not read again through the column's format ($, %, grouping), which it lacks
    value = raw if isinstance(raw, Decimal) else exact_value(raw, profile)
    if value is not None and not value.is_finite():
        return None
    if value is None:
        return None
    if not value:
        return "0"
    sign, digits, exponent = value.as_tuple()
    digits = list(digits)
    while digits[-1] == 0:
        digits.pop()
        exponent += 1
    return ("-" if sign else "") + "".join(map(str, digits)) + "e" + str(exponent)


def source_extremes(profile: ColumnProfile) -> frozenset[str]:
    model = profile.model
    if model is None:
        return frozenset()
    source = model.extra.get("_sharing_source_model", model)
    values = [value for raw in (*source.nonmissing_counts, *source.missing_counts)
              if (value := exact_value(raw, profile)) is not None]
    return frozenset(normalized_key(profile, value) for value in (min(values), max(values))) if values else frozenset()


def forbidden_extremes(profile: ColumnProfile) -> frozenset[str]:
    """Use full-column local-person evidence, including for an entity model."""
    if profile.model is not None and FORBIDDEN_KEY in profile.model.extra:
        return profile.model.extra[FORBIDDEN_KEY]
    return source_extremes(profile)


def _refuse(profile: ColumnProfile) -> None:
    raise GenerationError(f"sharing mode: column {profile.name!r} has no feasible numeric support after removing unsupported original extrema",
                          fix=f"--null {profile.name} or --exclude {profile.name}")


def protect_model(profile: ColumnProfile, model: ColumnModel, forbidden: frozenset[str]) -> ColumnModel:
    """Remove exact endpoints before either ordinary float or exact-decimal interpolation."""
    nonmissing = {raw: count for raw, count in model.nonmissing_counts.items()
                  if normalized_key(profile, raw) not in forbidden}
    missing = {raw: count for raw, count in model.missing_counts.items()
               if normalized_key(profile, raw) not in forbidden}
    numeric = {raw: value for raw in nonmissing if (value := exact_value(raw, profile)) is not None}
    had_numbers = any(exact_value(raw, profile) is not None for raw in model.nonmissing_counts)
    if had_numbers and not numeric:
        _refuse(profile)
    total = model.n_nonmissing + model.n_missing
    if not nonmissing and not missing and total:
        _refuse(profile)

    def rescale(counts: dict[Any, int], target: int) -> dict[Any, int]:
        return {raw: int(count) for raw, count in zip(counts, largest_remainder(list(counts.values()), target))
                if count} if counts else {}

    nonmissing = rescale(nonmissing, model.n_nonmissing if missing else total)
    missing = rescale(missing, model.n_missing if nonmissing else total)
    # Numeric interpolation reads this array; exact wide-number interpolation
    # reads nonmissing_counts. Both views must describe the protected support.
    xs = np.repeat(np.array([float(numeric[raw]) for raw in nonmissing if raw in numeric]),
                   [count for raw, count in nonmissing.items() if raw in numeric])
    return replace(model, nonmissing_counts=nonmissing, missing_counts=missing, sorted_numeric=np.sort(xs))


__all__ = ["TYPES", "MARKER", "FORBIDDEN_KEY", "discrete", "normalized_key", "source_extremes", "forbidden_extremes", "protect_model"]
