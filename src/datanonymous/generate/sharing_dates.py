"""Month-preserving person dates with no rendered source extrema (sharing D43).

Generation may leave the original day bounds within an observed year/month.
Whole calendar buckets hide excluded endpoints among other unobserved dates;
an exact pair of support holes must not disclose the original endpoint pair.
Calendar precision that leaves no safe bucket is refused. The ordinary date
renderer remains responsible for formats, fractions, offsets and native types.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Any, Iterable

import numpy as np

from ..detect import dates as D
from ..detect.dates_formats import US_DAY, civil_from_days, days_from_civil
from ..errors import ColumnRefusal, GenerationError
from ..model import ColumnModel, ColumnProfile
from ..profile.sharing import SOURCE_MODEL_KEY
from . import dates as G

MIN_BUCKET_DAYS = 5
MIN_ENDPOINT_PAIRS = 10
BUCKET_HOLDERS_KEY = "_sharing_date_bucket_holders"
BUCKET_MIN_KEY = "_sharing_date_bucket_min_people"


def _params(profile: ColumnProfile, model: ColumnModel) -> tuple[ColumnProfile, dict[str, Any]]:
    p = replace(profile, params=deepcopy(profile.params))
    params = G._params(p, model)
    params["_no_exact_extremes"] = True
    return p, params


def _parse_params(params: dict[str, Any]) -> dict[str, Any]:
    # A syntactically valid date labelled as a sentinel still has a normalized
    # date value and must not bypass an exact-extreme check.
    return {k: v for k, v in params.items() if k not in ("sentinels", "_other_tokens")}


def normalized_keys(profile: ColumnProfile, values: Iterable[Any]) -> list[tuple[int, ...] | None]:
    """Exact date/instant keys for final verification; no float timestamp rounding."""
    raw = list(values)
    params = profile.params
    if not params.get("formats") and profile.model is not None:
        _, params = _params(profile, profile.model)
    view = D.view(_parse_params(params), [str(v).strip() if v is not None else "" for v in raw])
    out: list[tuple[int, ...] | None] = []
    for i in range(len(raw)):
        if view.cls[i] < 0:
            out.append(None)
        elif view.kind == "datetime":
            out.append((int(view.day[i]), int(view.sod[i])))
        elif view.kind == "year_month":
            out.append((int(view.mon[i]),))
        else:
            out.append((int(view.day[i]),))
    return out


def source_extremes(profile: ColumnProfile, model: ColumnModel | None = None) -> frozenset[tuple[int, ...]]:
    """Normalized original extrema, shared by generation and on-disk verification."""
    model = model or profile.model
    if model is None:
        return frozenset()
    source = model.extra.get(SOURCE_MODEL_KEY, model)
    if not source.nonmissing_counts and not source.missing_counts:
        return frozenset()
    p, _ = _params(profile, source)
    keys = [k for k in normalized_keys(p, [*source.nonmissing_counts, *source.missing_counts]) if k is not None]
    return frozenset((min(keys), max(keys))) if keys else frozenset()


def _refuse(profile: ColumnProfile) -> None:
    raise ColumnRefusal(f"sharing mode: column {profile.name!r} has no month-preserving date cover that hides its original extrema in the required format/calendar",
                        fix=f"--null {profile.name} or --exclude {profile.name}; use a finer date precision or remove the rigid calendar constraint",
                        code="sharing_date_no_cover", column=profile.name, index=profile.index,
                        plain="its dates cannot be moved enough within their months to hide the earliest and latest real dates")


def _month_days(month: int) -> np.ndarray:
    """A public calendar domain independent of source weekdays and date grids."""
    year, mon = 1970 + month // 12, month % 12 + 1
    first = int(days_from_civil(np.array([year]), np.array([mon]), np.array([1]))[0])
    size = int(D.month_length(np.array([year]), np.array([mon]))[0])
    return np.arange(first, first + size, dtype=np.int64)


def _calendar_buckets(days: np.ndarray) -> list[np.ndarray]:
    """Endpoint-independent, balanced partitions with at least five eligible days.

    The last partial bucket is never left smaller than the minimum. Bucket
    boundaries depend only on the required calendar, never on source extrema
    or the random seed. Calendars with fewer than five days cannot provide the
    stated support-hole ambiguity.
    """
    days = np.unique(days)
    count = len(days) // MIN_BUCKET_DAYS
    return list(np.array_split(days, count)) if count else []


def calendar_bucket_keys(profile: ColumnProfile, values: Iterable[Any]) -> list[str | None]:
    """Public bucket units for distinct-person counting while source rows exist."""
    raw = list(values)
    if profile.model is None:
        return [None] * len(raw)
    _, params = _params(profile, profile.model)
    view = D.view(_parse_params(params), [str(v).strip() if v is not None else "" for v in raw])
    if view.kind not in ("date", "datetime"):
        return [None] * len(raw)
    boundaries: dict[int, np.ndarray] = {}
    result = []
    for i in range(len(raw)):
        if view.cls[i] < 0:
            result.append(None)
            continue
        month = int(view.mon[i])
        if month not in boundaries:
            boundaries[month] = np.array([bucket[-1] for bucket in _calendar_buckets(_month_days(month))])
        bucket = int(np.searchsorted(boundaries[month], int(view.local_day[i])))
        result.append(f"calendar-bucket:{month}:{bucket}")
    return result


def validate_bucket_support(profile: ColumnProfile) -> None:
    """Every visibly excluded endpoint bucket needs local distinct-person support."""
    model = profile.model
    if model is None:
        return
    p, params = _params(profile, model)
    source = model.extra.get(SOURCE_MODEL_KEY, model)
    raw = [*source.nonmissing_counts, *source.missing_counts]
    view = D.view(_parse_params(params), [str(v).strip() if v is not None else "" for v in raw])
    if view.kind not in ("date", "datetime"):
        return  # Unsupported precision is refused by the calendar generator.
    # A parsed missing sentinel outside every generated month creates no
    # calendar hole. It must not require a bucket that will never be emitted.
    months = {int(view.mon[i]) for i in range(len(source.nonmissing_counts)) if view.cls[i] >= 0}
    extrema = source_extremes(p)
    normalized = normalized_keys(p, raw)
    buckets = calendar_bucket_keys(p, raw)
    visible = {buckets[i] for i, key in enumerate(normalized)
               if key in extrema and int(view.mon[i]) in months and buckets[i] is not None}
    held = model.extra.get(BUCKET_HOLDERS_KEY) or {}
    minimum = max(10, int(model.extra.get(BUCKET_MIN_KEY, 10)))
    if any(int(held.get(key, 0)) < minimum for key in visible):
        raise ColumnRefusal(
            f"sharing mode: column {profile.name!r} has an excluded endpoint calendar bucket supported by fewer than the required distinct people",
            fix=f"--null {profile.name} or --exclude {profile.name}; use --person-id COL when a column identifies the row's person",
            code="sharing_rare_class", column=profile.name, index=profile.index, plain="too few people have dates near its earliest or latest real date to hide them")


def _covered_pool(profile: ColumnProfile, days: np.ndarray, forbidden_days: set[int], *, datetime: bool = False) -> np.ndarray:
    """Exclude entire endpoint buckets, keeping support holes ambiguous at saturation.

    For date-only data, the same excluded support represents at least ten
    distinct endpoint pairs within one bucket, or twenty-five across two buckets.
    This bound concerns the calendar support alone. It does not certify other
    statistics or the complete generator against every inference attack.

    Datetime renderers can shift a date by one day through timezone offsets.
    Excluding adjacent buckets as well preserves the same coarse day boundary
    instead of creating a new one-instant hole at a timezone boundary.
    """
    buckets = _calendar_buckets(days)
    if not buckets:
        _refuse(profile)
    blocked = {i for i, bucket in enumerate(buckets) if forbidden_days.intersection(map(int, bucket))}
    if datetime:
        blocked = {j for i in blocked for j in (i - 1, i, i + 1) if 0 <= j < len(buckets)}
    kept = [bucket for i, bucket in enumerate(buckets) if i not in blocked]
    if not kept:
        _refuse(profile)
    return np.concatenate(kept)


def _check_datetime_extreme_days(profile: ColumnProfile, params: dict[str, Any], model: ColumnModel,
                                 forbidden: frozenset[tuple[int, ...]]) -> None:
    """Refuse UTC/local endpoint day aliases outside the calendar-cover proof."""
    source = model.extra.get(SOURCE_MODEL_KEY, model)
    raw = [*source.nonmissing_counts, *source.missing_counts]
    view = D.view(_parse_params(params), [str(v).strip() if v is not None else "" for v in raw])
    for i, key in enumerate(normalized_keys(profile, raw)):
        if key in forbidden and int(view.local_day[i]) != key[0]:
            _refuse(profile)


def _calendar_cover(profile: ColumnProfile, params: dict[str, Any], model: ColumnModel,
                    forbidden: frozenset[tuple[int, ...]]) -> dict[int, np.ndarray]:
    """The full source column defines one calendar domain for every generation model."""
    source = model.extra.get(SOURCE_MODEL_KEY, model)
    view = D.view(_parse_params(params), [str(v).strip() for v in source.nonmissing_counts])
    observed = view.local_day[view.cls >= 0]
    years, months, _ = civil_from_days(observed)
    kind = params.get("kind") or D.column_kind([f["format"] for f in params["formats"]])
    if kind not in ("date", "datetime") or params.get("calendar_reason") == "--calendar":
        _refuse(profile)
    validate_bucket_support(profile)
    if kind == "datetime":
        _check_datetime_extreme_days(profile, params, model, forbidden)
    forbidden_days = {key[0] for key in forbidden}
    base = {int(m): _month_days(int(m))
            for m in np.unique((years - 1970) * 12 + months - 1)}
    pools = {month: _covered_pool(profile, days, forbidden_days, datetime=kind == "datetime")
             for month, days in base.items()}
    return pools


def cover_violations(profile: ColumnProfile, values: Iterable[Any]) -> int:
    """Count repaired/re-read dates outside the protected calendar cover."""
    if profile.model is None:
        return 0
    p, params = _params(profile, profile.model)
    raw = [str(v).strip() if v is not None else "" for v in values]
    view = D.view(_parse_params(params), raw)
    rows = np.flatnonzero(view.cls >= 0)
    if not len(rows):
        return 0
    try:
        pools = _calendar_cover(p, params, profile.model, source_extremes(p))
    except GenerationError:
        return len(rows)
    allowed = {month: frozenset(map(int, days)) for month, days in pools.items()}
    return sum(int(view.local_day[i]) not in allowed.get(int(view.mon[i]), ()) for i in rows)


def _set_days(out: np.ndarray, rows: np.ndarray, local_days: np.ndarray, local_sod: np.ndarray,
              params: dict[str, Any]) -> None:
    tpu = D.ticks_per_us(params)
    per_day = US_DAY * tpu
    if (params.get("kind") or D.column_kind([f["format"] for f in params["formats"]])) != "datetime":
        out["day"][rows] = local_days
        out["us"][rows] = 0
        return
    # Python integers detect overflow instead of silently wrapping a native
    # nanosecond timestamp into an unrelated year.
    local_big = [int(d) * per_day + int(t) for d, t in zip(local_days, local_sod)]
    info = np.iinfo(np.int64)
    if any(v < info.min or v > info.max for v in local_big):
        raise GenerationError("sharing mode: a month-preserving date exceeds the source timestamp representation",
                              code="sharing_date_out_of_range")
    local = np.array(local_big, dtype=np.int64)
    offsets = G._offset_ticks(out["tz"][rows], local, tpu)
    instant_big = [int(v) - int(offset) for v, offset in zip(local, offsets)]
    if any(v < info.min or v > info.max for v in instant_big):
        raise GenerationError("sharing mode: a month-preserving instant exceeds the source timestamp representation",
                              code="sharing_date_out_of_range")
    inst = np.array(instant_big, dtype=np.int64)
    out["day"][rows], out["us"][rows] = np.divmod(inst, per_day)


def generate_dates(profile: ColumnProfile, model: ColumnModel, n: int,
                   rng: np.random.Generator) -> tuple[np.ndarray, dict[str, bool]]:
    """Rendered person dates: preserve observed year/month shares, exclude source extrema.

    Called only for profiles marked _sharing_person_date. Source extrema come
    from the untouched row model even when generating an entity model.
    """
    if n == 0:
        return np.empty(0, dtype=object), {}
    p, params = _params(profile, model)
    source = model.extra.get(SOURCE_MODEL_KEY, model)
    # Protect the full source column when an entity generation model is supplied.
    extreme_model = profile.model or source
    forbidden = source_extremes(p, extreme_model)
    kind = params.get("kind") or D.column_kind([f["format"] for f in params["formats"]])
    pools = _calendar_cover(p, params, extreme_model, forbidden)
    raws = list(model.nonmissing_counts)
    view = D.view(_parse_params(params), [str(v).strip() for v in raws])
    real_rows = np.flatnonzero(view.cls >= 0)
    if not len(real_rows):
        _refuse(profile)
    observed = view.local_day[real_rows]
    years, months, _ = civil_from_days(observed)
    month_keys = (years - 1970) * 12 + months - 1
    unique_months, inverse = np.unique(month_keys, return_inverse=True)
    weights = np.bincount(inverse, weights=np.array([model.nonmissing_counts[raws[i]] for i in real_rows]))
    generated, _ = G.generate_dates(p, model, n, rng)
    rendered = G.render_dates(p, generated)
    parsed = D.view(_parse_params(params), [str(v).strip() for v in rendered])
    rows = np.flatnonzero(parsed.cls >= 0)
    targets = G._quota_values(unique_months, weights, len(rows), rng).astype(np.int64)
    # Convert syntactically valid sentinel dates back to ordinary date rows.
    generated["text"][rows] = None
    generated["fmt"][rows] = parsed.cls[rows]
    generated["fd"][rows] = parsed.digits[rows]
    local_sod = parsed.local_sod[rows]
    zone = D.zone_rule(params)
    generated["tz"][rows] = parsed.tz[rows]
    if zone:
        has_tz = np.array([bool(t) for t in generated["tz"][rows]])
        generated["tz"][rows[has_tz]] = zone
    picked = np.empty(len(rows), dtype=np.int64)
    for month, pool in pools.items():
        selected = targets == month
        picked[selected] = rng.choice(pool, int(selected.sum()))
    _set_days(generated, rows, picked, local_sod, params)
    rendered = G.render_dates(p, generated)
    actual = normalized_keys(p, rendered)
    actual_months = D.view(_parse_params(params), [str(v).strip() for v in rendered]).mon
    # Rendering can truncate time/fractions or change a timezone token. Verify
    # the rendered key, and search the finite day pool rather than retrying to
    # an arbitrary limit or copying a bound when a month has one source day.
    for position, row in enumerate(rows):
        if actual[row] not in forbidden and actual[row] is not None and actual_months[row] == targets[position]:
            continue
        pool = pools[int(targets[position])]
        for day in rng.permutation(pool):
            trial = generated[row:row + 1].copy()
            _set_days(trial, np.array([0]), np.array([day]), np.array([local_sod[position]]), params)
            value = G.render_dates(p, trial)[0]
            key = normalized_keys(p, [value])[0]
            month = D.view(_parse_params(params), [str(value).strip()]).mon[0]
            if key is not None and key not in forbidden and month == targets[position]:
                rendered[row] = value
                break
        else:
            _refuse(profile)
    if any(key in forbidden for key in normalized_keys(p, rendered) if key is not None):
        _refuse(profile)
    if cover_violations(profile, rendered):
        _refuse(profile)
    return rendered, {"sharing_date_extremes_removed": True, "extreme_value_adjusted": False}


__all__ = ["generate_dates", "normalized_keys", "source_extremes", "cover_violations", "calendar_bucket_keys",
           "validate_bucket_support", "BUCKET_HOLDERS_KEY", "BUCKET_MIN_KEY"]
