"""Date, time and duration detection (AREA A).

AREA A owns this module: DATE, DATE_MIXED_FORMATS, DATE_AS_INT, DATETIME,
DATETIME_TZ, YEAR_MONTH, TIME and DURATION (design 3 #12, #19-24, section 4,
section 10 "Dates", section 12). SCHOOL_YEAR stays in the FOUNDATION
(``detect/types.py``); 4-digit integer years are never claimed (INTEGER_YEAR).

Detection (:func:`detect_dates`)
--------------------------------
Works on the stripped distinct values (NA tokens removed). In order:

1. Zero dates (``0/0/0``, ``00/00/0000``, ``0000-00-00 00:00:00``) are set aside
   as sentinel tokens.
2. ``--date-format COL=FMT`` (strftime-like; ``%-m`` accepted): that format
   only; fewer than 99% of the distinct values matching FAILs the file.
3. All-digit columns (never leading-zero codes, never 4-digit years):
   DATE_AS_INT when every value is an 8-digit valid ``%Y%m%d`` (a bare ``0``
   is a zero-date token); Excel serials 10000-73050 only with a date header hint,
   ``--date-cols COL`` or ``--force-type COL=DATE_AS_INT``; Unix epoch seconds
   (9e8-2.2e9) / milliseconds (9e11-2.2e12) only with a date/time hint or those flags.
   An org-code header (``EntryCode``, ``Exit_Code``; D10) is never a date hint here.
4. Otherwise the explicit candidate formats of ``detect/dates_formats.py``
   (about 360: ISO with ``T``/``Z``/fractions/offsets, padded/unpadded m/d/Y,
   d/m/Y, ``%y``, month names in any case incl. SAS ``15MAR2024`` and Oracle
   ``14-AUG-24``, AM/PM, ``%z``/zone names, year-month, time-only, durations)
   on up to 5,000 sampled distinct values (a deterministic sample; a cheap
   character-shape pre-filter skips impossible formats). One format with >= 99%
   coverage wins; otherwise a greedy cover of <= 3 formats, each adding >= 1%,
   must reach 99% (DATE_MIXED_FORMATS; mixed datetime layouts stay DATETIME).
   The choice is then validated on all distinct values (>= 99%).
   Ties go to the earlier candidate; an m/d vs d/m tie is *ambiguous*:
   month-first by default with a warning, day-first with ``--dayfirst COL``.
   ``h:mm[:ss]`` values are DURATION when any hour is >= 24 or the header has a
   duration hint, otherwise TIME; ISO ``PT..`` and ``N days hh:mm:ss`` are DURATION.
5. ``%y`` years use the pivot 69. Values that land after the run date are moved
   back 100 years for statistics and generation (the text stays 2-digit) in
   DOB-hinted columns, and in other columns only when that makes the column's
   range narrower (near-future schedule dates stay as they are).
6. Sentinels (tokens written verbatim at their exact rates, never modelled):
   zero dates, 1900-01-01 and 9999-12-31 in any format and the other zero and
   edge days of common date systems (:data:`FIXED_SENTINELS`: Excel's day 0
   1899-12-30 / 1899-12-31, 1904-01-01, 1753-01-01, 0001-01-01), always,
   also in a DOB-hinted column, as a date (a zero day with a time of day is a
   bare time: ``1899-12-30 08:41``); and (K-07) a
   value that lies >= 50 years from the median, held by >= 1% of the rows and
   by >= ``--min-cell`` distinct entities (rows here, entities once the table's
   IDs are known: ``profile.table`` reads the column again without the far
   values fewer people hold, :data:`FAR_SENTINELS_KEY`), never in a
   person-level (DOB-hinted) column: a real far-off date (an adult learner's
   birth date, a typo) is an ordinary date then, of which only the exact
   min/max of D14 is written as it is. The <= 1% of
   distinct values no format reads (:func:`leftover_kinds`) are tokens too when
   they are out-of-range dates in a detected layout (``02/30/2024``,
   ``99/99/9999``; section 4), valid dates in a layout too rare to be a format
   (``15MAR2024`` among ISO dates), or short placeholders (<= 20 characters,
   <= 2 words, no digit other than 0, no ``@``) made only of known placeholder
   words (``PLACEHOLDER_WORDS``: ``TBD``, ``Unknown``, ``Not enrolled``) or of
   no letters at all (``0``, ``-``).
   Every other leftover (person names and other unknown words, free text, digit
   strings such as IDs or phone numbers, e-mail addresses) is never copied: its
   rows get generated dates (info message; ``--na-value TOKEN`` keeps a
   placeholder the list does not know).

``Detection.params`` (public unless the key starts with ``_``): ``kind``
(date | datetime | year_month | time | duration), ``formats`` (list of
``{format, share, render}``; ``render`` = padding per component, name case,
AM/PM tokens, fraction digits (``frac_digits``, plus ``frac_digits_pmf``
{digits: rows} when values of one format are written with different numbers
of fraction digits), duration layout), ``sentinels`` ({token: count}),
``other_tokens`` (count; values in ``_other_tokens``), ``two_digit_year``,
``tz_tokens`` ({offset/zone token: count}), ``zone_rule`` (the IANA zone whose
DST rules give every written offset, when the offsets change with it:
:func:`infer_zone`; R4-06), ``calendar`` (+ reason),
``person_level``, ``weekdays_observed``, ``granularity_seconds`` and
``grid_phase_seconds`` (the observed time grid ``phase + k * step``, any step:
:func:`time_grid`), ``day_grid`` / ``day_of_month`` / ``month_end`` /
``month_grid`` (the observed calendar grid of dates and year-months:
:func:`date_grid`), ``resolution`` (the written unit), ``whitespace``
([[lead, trail, count], ...]), ``stored_as`` (DATE_AS_INT), ``native`` (native typed storage, see below), ``time_unit``
(``'ns'`` when values carry sub-microsecond digits: times of day and durations
are then held in nanoseconds instead of microseconds). Calendar ratio,
observed weekdays and the time granularity describe the values as written
(local wall time for values with offsets).

Native typed storage (Excel date cells, Parquet date32/timestamp/time64/
duration, Stata %td/%tc): the reader renders cells to canonical strings and
describes them in ``RawColumn.native`` (contract in ``generate/dates.py``);
:func:`native_temporal` reads that description and :func:`native_detection`
(callable from the AREA E/F ``detect_native`` hooks; :func:`detect_dates` uses
it too) restricts detection to the native kind.

:func:`view` is the one reading of a column's values under its params (used by
statistics, the numeric view and the generator): per distinct value its class
(sentinel / other token / unparsed / format index) and its components in
*model space* (UTC for values with offsets, ``%y`` shift applied), next to the
local components as written.
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import re
from collections import Counter
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from .. import hints as hints_mod
from ..errors import UnsupportedColumnError, UsageError
from ..model import ColumnType, Message
from .context import DetectContext, Detection
from .dates_formats import (
    CANDIDATES,
    CLOCK_DURATIONS,
    NATIVE_DATE_EXTRA,
    NO_OFFSET,
    NS_PER_US,
    TIME_FORMATS,
    US,
    US_DAY,
    Parsed,
    civil_from_days,
    coarse_skeleton,
    day_first,
    days_from_civil,
    get_format,
    hms,
    iso_day,
    iso_instant,
    iso_month,
    kind_group,
    normalize_format,
    ordinal,
    rule_zone,
    swapped_md,
    token_offset,
    weekday,
    zone_instant_offsets,
)

T = ColumnType
DATE_TYPES = frozenset({T.DATE, T.DATE_MIXED_FORMATS, T.DATE_AS_INT, T.DATETIME, T.DATETIME_TZ, T.YEAR_MONTH,
                        T.TIME, T.DURATION})

SAMPLE_MAX = 5000
COVERAGE = 0.99
MIN_GAIN = 0.01
MAX_MIXED = 3
SENTINEL_MIN_SHARE = 0.01
SENTINEL_YEARS = 50
#: Sentinel dates in any column, whatever their count (K-07: the only unconditional ones): the zero and edge days of
#: the date systems school data passes through, which no person's real date is: 1900-01-01 and 9999-12-31; Excel's day
#: 0 as its two readings give it (1899-12-30, the OLE/Access/pandas origin; 1899-12-31, "1/0/1900"); the Mac Excel
#: 1904 system's day 0 (1904-01-01); SQL Server's ``datetime`` minimum (1753-01-01); the .NET and ISO minimum
#: (0001-01-01). A closed set of system constants, not a list of guesses: a placeholder of a column's own (2099-12-31,
#: 1901-01-01) is a sentinel by the far rule (>= --min-cell entities), and an epoch inside living memory (1970-01-01,
#: SAS/Stata's 1960-01-01) is somebody's real date.
#: A zero day is a placeholder only as a date (no time of day, or midnight): with a time of day it is how the same
#: systems write a bare time (an Access/ODBC or Excel ``Time_In`` read as 1899-12-30 08:41, SQL Server's ``time`` cast
#: to ``datetime`` as 1900-01-01 08:41, .NET's ``TimeSpan`` as 0001-01-01 08:41): an ordinary value then
#: (:data:`END_OF_TIME` is a placeholder at any time: ``9999-12-31 23:59:59``).
FIXED_SENTINELS = ((1900, 1, 1), (9999, 12, 31), (1899, 12, 30), (1899, 12, 31), (1904, 1, 1), (1753, 1, 1),
                   (1, 1, 1))
END_OF_TIME = (9999, 12, 31)
TOKEN_MAX_CHARS = 20
TOKEN_MAX_WORDS = 2
#: Words a placeholder token of a date column may be made of (case-folded; R1-02). A short leftover is
#: written verbatim only when every letter run is one of these words (``TBD``, ``Not enrolled``, ``N/A``,
#: ``MM/DD/YYYY``, ``#VALUE!``) or when it has no letter at all (``0``, ``-``, ``?``): a person's name
#: (``Zelmira Quenby``) or any other unknown word is never copied. ``--na-value TOKEN`` keeps another
#: placeholder verbatim (NA tokens are written back at their exact rates).
PLACEHOLDER_WORDS = frozenset({
    "tbd", "tba", "tbc", "tbs", "tdb", "na", "n", "a", "x", "xx", "xxx", "xxxx", "u", "unk", "unkn", "unknown",
    "nil", "null", "none", "nan", "missing", "blank", "empty", "pending", "review", "not", "no", "non", "yet",
    "applicable", "available", "avail", "provided", "reported", "recorded", "entered", "enrolled", "given",
    "known", "set", "specified", "unspecified", "undetermined", "undefined", "required", "date", "dates",
    "never", "invalid", "error", "err", "current", "currently", "present", "ongoing", "active", "inactive",
    "open", "closed", "continuing", "today", "now", "still", "in", "progress", "withdrawn", "graduated",
    "transferred", "expired", "cancelled", "canceled", "waived", "exempt", "refused", "declined",
    "incomplete", "default", "various", "multiple", "other", "same", "as", "above", "see", "notes", "note",
    "value", "ref", "div", "num", "name", "mm", "dd", "yy", "yyyy", "m", "d", "y", "hh", "ss",
})
_LETTER_RUN = re.compile(r"[^\W\d_]+")
#: PRIVATE params key: the unparsed values a date column replaces with generated dates (never written; R7-01: they
#: are real texts of the run that no generator writes, ``privacy.texts``).
REPLACED_KEY = "_unparsed_replaced_values"
#: PRIVATE params key (K-07): the sentinels the far-from-median rule found (not the fixed ones, not zero dates). A
#: sentinel is written verbatim, so each must be held by >= ``--min-cell`` entities: ``profile.table`` counts them
#: once the table's IDs are known (:func:`far_sentinel_holders`) and reads the column again with only those that pass
#: (:func:`far_sentinels_only`); the others are ordinary dates (only the exact min/max of D14 is written as it is).
FAR_SENTINELS_KEY = "_far_sentinels"
_FAR_ALLOWED: ContextVar[frozenset[str] | None] = ContextVar("datanonymous_far_sentinels", default=None)


@contextlib.contextmanager
def far_sentinels_only(values: Iterable[str]) -> Iterator[None]:
    """Inside the block, the far-from-median sentinel rule marks only ``values`` (K-07: the ones held by >=
    ``--min-cell`` entities); the fixed sentinels and zero dates are unaffected."""
    token = _FAR_ALLOWED.set(frozenset(values))
    try:
        yield
    finally:
        _FAR_ALLOWED.reset(token)


def far_sentinel_holders(p: Any, values: Any, keys: Any) -> dict[str, int]:
    """K-07: far-from-median sentinel (:data:`FAR_SENTINELS_KEY`) -> distinct entities of its table holding it, counted
    like the D35 holders of a text value (``detect.text``: the fewest over every ID column of the table and the names
    of the row's own person, else rows with an exported duplicate counted once; ``keys``:
    ``profile.entity.d35_keys``). ``values``: the column's raw values in row order."""
    from . import text as text_detect

    far = set((p.params or {}).get(FAR_SENTINELS_KEY) or ())
    if not far:
        return {}

    def units(raw: Any) -> list[str]:
        s = raw.strip() if isinstance(raw, str) else None
        return [s] if s in far else []

    rows: Counter = Counter()
    for raw, c in Counter(v for v in np.asarray(values, dtype=object).tolist() if isinstance(v, str)).items():
        for u in units(raw):
            rows[text_detect.unit_key(u)] += c
    by_key = {text_detect.unit_key(s): s for s in far}
    counted = text_detect._counted(p, values, keys, [k for k in by_key if rows[k]], rows, units)
    return {by_key[k]: int(n) for k, n in counted.items()}
QUANTILES = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
GRANULARITY_STEPS_US = (1800 * US, 900 * US, 300 * US, 60 * US, US, 1000, 1)
MAX_YEAR_PMF = 150
#: Nanosecond ticks: day ordinals whose instants fit int64 (1677-09-22 .. 2262-04-11, with a margin).
NS_SAFE_DAYS = (2**63 - 1) // (US_DAY * NS_PER_US) - 2
#: Nanosecond ticks: largest duration (us) whose nanoseconds fit int64 (about 292 years).
NS_SAFE_DUR_US = (2**63 - 1) // NS_PER_US - US_DAY

#: Classes in :class:`ColumnView` ``cls`` (>= 0 is the index of the format that reads the value).
SENTINEL, OTHER_TOKEN, UNPARSED = -3, -2, -1

ZERO_DATE_RE = re.compile(r"0+(?:[-/.]0+){1,2}(?:[ T]0+(?::0+){1,2}(?:\.0+)?)?")
_DATEISH_RE = re.compile(r"[-+]?[0-9A-Za-z][0-9A-Za-z ,./:+_\-]*")
_HAS_DIGIT = re.compile(r"[0-9]")

_KIND_TYPES = {"year_month": T.YEAR_MONTH, "time": T.TIME, "duration": T.DURATION}
_FORCED_KINDS: dict[ColumnType, tuple[str, ...]] = {
    T.DATE: ("date",), T.DATE_MIXED_FORMATS: ("date",), T.DATETIME: ("datetime",), T.DATETIME_TZ: ("datetime",),
    T.YEAR_MONTH: ("year_month",), T.TIME: ("time",), T.DURATION: ("duration",), T.DATE_AS_INT: (),
}


# --------------------------------------------------------------------------- native contract
_NATIVE_KINDS = ("date", "datetime", "time", "duration")


def native_temporal(native: dict[str, Any] | None) -> dict[str, Any] | None:
    """Temporal description of a native column, or None (see ``generate/dates.py``).

    Preferred: ``native['temporal'] = {'kind': 'date'|'datetime'|'time'|'duration',
    'tz': zone or None, 'unit': ...}``. Also recognised: an Arrow type string
    under ``arrow_type``/``type`` (``date32[day]``, ``timestamp[us, tz=...]``,
    ``time64[us]``, ``duration[s]``), a Stata format under ``stata_format``/
    ``format`` (``%td``, ``%tc``), and an Excel ``cell_type``/``cell_types`` 'd'
    with its ``number_format`` (date and time parts -> datetime, time only -> time,
    ``[h]:mm`` -> duration).
    """
    if not isinstance(native, dict):
        return None
    t = native.get("temporal")
    if isinstance(t, dict) and t.get("kind") in _NATIVE_KINDS:
        return {"kind": t["kind"], "tz": t.get("tz"), "unit": t.get("unit")}
    for key in ("arrow_type", "type", "pa_type", "dtype"):
        s = native.get(key)
        if not isinstance(s, str):
            continue
        low = s.lower()
        if low.startswith("date32") or low.startswith("date64") or low == "date":
            return {"kind": "date", "tz": None, "unit": "D"}
        if low.startswith("timestamp") or low.startswith("datetime64"):
            m = re.search(r"tz=([^\]\s,]+)", s) or re.search(r"datetime64\[\w+,\s*([^\]]+)\]", s)
            return {"kind": "datetime", "tz": m.group(1) if m else None, "unit": None}
        if low.startswith("time32") or low.startswith("time64"):
            return {"kind": "time", "tz": None, "unit": None}
        if low.startswith("duration") or low.startswith("timedelta64"):
            return {"kind": "duration", "tz": None, "unit": None}
    for key in ("stata_format", "format", "fmt"):
        s = native.get(key)
        if isinstance(s, str) and s.startswith("%t"):
            if s.startswith("%td"):
                return {"kind": "date", "tz": None, "unit": "D"}
            if s.startswith(("%tc", "%tC")):
                return {"kind": "datetime", "tz": None, "unit": "ms"}
    ct = native.get("cell_type")
    cts = native.get("cell_types")
    is_date_cell = ct == "d" or (isinstance(cts, dict) and cts and max(cts, key=lambda k: cts[k]) == "d")
    if is_date_cell:
        return {"kind": _excel_kind(str(native.get("number_format") or "")), "tz": None, "unit": None}
    return None


def _excel_kind(number_format: str) -> str:
    """Temporal kind of an Excel date cell from its number format.

    ``[h]:mm:ss`` (elapsed time) -> duration; time parts without a date part
    (``h:mm AM/PM``) -> time; date and time parts -> datetime; otherwise date.
    Quoted literals, escapes and bracketed locale/colour codes are ignored.
    """
    nf = re.sub(r'"[^"]*"|\\.', "", number_format).lower()
    if re.search(r"\[(?:h+|m+|s+)\]", nf):
        return "duration"
    bare = re.sub(r"\[[^\]]*\]", "", nf)
    has_time = bool(re.search(r"[hs]|am/pm|a/p", bare))
    has_date = bool(re.search(r"[yd]", bare)) or (not has_time and "m" in bare)
    if has_time and not has_date:
        return "time"
    return "datetime" if has_time else "date"


def native_detection(ctx: DetectContext) -> Detection | None:
    """Detection for a native temporal column (for AREA E/F ``detect_native``), or None."""
    if native_temporal(ctx.native) is None:
        return None
    return detect_dates(ctx)


# --------------------------------------------------------------------------- inputs
@dataclass
class DateInputs:
    """What the detector needs (built from a DetectContext, or from a model as a fallback)."""

    name: str
    typed: dict[str, int]
    raw_counts: dict[Any, int]
    hints: hints_mod.HeaderHints
    min_cell: int = 10
    dayfirst: bool = False
    date_cols: bool = False
    date_format: str | None = None
    calendar: bool = False
    native: dict[str, Any] | None = None
    forced: ColumnType | None = None
    digit_code: bool = False
    today: _dt.date | None = None

    @classmethod
    def from_context(cls, ctx: DetectContext) -> "DateInputs":
        o = ctx.options
        return cls(name=ctx.name, typed=ctx.typed_values, raw_counts=ctx.raw_counts, hints=ctx.hints,
                   min_cell=ctx.settings.min_cell, dayfirst=bool(getattr(o, "dayfirst", False)),
                   date_cols=bool(getattr(o, "date_cols", False)), date_format=getattr(o, "date_format", None),
                   calendar=bool(getattr(o, "calendar", False)), native=native_temporal(ctx.native),
                   forced=ctx.forced_type, digit_code=bool(ctx.digit_code))

    @property
    def dob(self) -> bool:
        return hints_mod.match_any(self.name, hints_mod.PII_HINTS["dob"]) is not None


# --------------------------------------------------------------------------- column view
@dataclass
class ColumnView:
    """Distinct values read under a column's params (see module docstring)."""

    kind: str
    values: list[str]
    cls: np.ndarray
    day: np.ndarray  # date part in model space (UTC for values with offsets)
    sod: np.ndarray  # time of day (us) in model space
    dur: np.ndarray  # durations (us)
    mon: np.ndarray  # month ordinal (months since 1970-01)
    local_day: np.ndarray  # date part as written (after the %y shift)
    local_sod: np.ndarray  # time of day as written (us)
    tz: np.ndarray  # offset/zone token per value ('' when none)
    has_time: np.ndarray  # the value's format has a time part
    parts: list[tuple[int, np.ndarray, Parsed]] = field(default_factory=list)
    #: ticks per microsecond of ``sod``/``local_sod``/``dur``: 1, or 1000 for nanosecond columns
    tpu: int = 1
    #: fraction digits as written per value (-1: none)
    digits: np.ndarray | None = None

    @property
    def dated(self) -> np.ndarray:
        return self.cls >= 0

    def numbers(self) -> np.ndarray:
        """The numeric view: days / months / seconds per value (NaN for tokens)."""
        k = self.kind
        tps = US * self.tpu
        if k == "date":
            x = self.day.astype(float)
        elif k == "year_month":
            x = self.mon.astype(float)
        elif k == "datetime":
            if self.tpu == 1:
                x = (self.day * US_DAY + self.sod).astype(float) / US
            else:  # nanosecond instants overflow float precision: seconds from the parts
                x = self.day.astype(float) * 86400.0 + self.sod.astype(float) / tps
        elif k == "time":
            x = self.sod.astype(float) / tps
        else:
            x = self.dur.astype(float) / tps
        return np.where(self.dated, x, np.nan)


def ticks_per_us(params: dict[str, Any]) -> int:
    """Ticks per microsecond of a column's model values (``params['time_unit']``)."""
    return NS_PER_US if params.get("time_unit") == "ns" else 1


def column_kind(keys: Sequence[str]) -> str:
    kinds = {get_format(k).kind for k in keys}
    if "datetime" in kinds:
        return "datetime"
    if "date" in kinds:
        return "date"
    return next(iter(kinds)) if kinds else "date"


def _format_keys(params: dict[str, Any]) -> list[str]:
    return [str(f["format"]) for f in params.get("formats") or [] if isinstance(f, dict) and f.get("format")]


def view(params: dict[str, Any], values: Sequence[str]) -> ColumnView:
    """Read stripped ``values`` under ``params`` (classes and model-space components)."""
    keys = _format_keys(params)
    kind = params.get("kind") or column_kind(keys)
    vals = values if isinstance(values, list) else [str(v) for v in values]
    n = len(vals)
    cls = np.full(n, UNPARSED, dtype=np.int64)
    sentinels = params.get("sentinels") or {}
    others = params.get("_other_tokens") or ()
    arr = np.array(vals, dtype=object)
    if sentinels or others:
        ser = pd.Series(arr, dtype=object)
        if others:
            cls[ser.isin(list(others)).to_numpy()] = OTHER_TOKEN
        if sentinels:
            cls[ser.isin(list(sentinels)).to_numpy()] = SENTINEL
    year = np.full(n, 1970, dtype=np.int64)
    month = np.ones(n, dtype=np.int64)
    dday = np.ones(n, dtype=np.int64)
    sod = np.zeros(n, dtype=np.int64)
    dur = np.zeros(n, dtype=np.int64)
    off = np.full(n, NO_OFFSET, dtype=np.int64)
    yy = np.zeros(n, dtype=bool)
    tz = np.empty(n, dtype=object)
    tz[:] = ""
    has_time = np.zeros(n, dtype=bool)
    digits = np.full(n, -1, dtype=np.int64)
    tpu = ticks_per_us(params)
    parts: list[tuple[int, np.ndarray, Parsed]] = []
    remaining = np.flatnonzero(cls == UNPARSED)
    for fi, key in enumerate(keys):
        if remaining.size == 0:
            break
        f = get_format(key)
        p = f.parse(arr[remaining])
        hit = remaining[p.ok]
        if hit.size:
            sub = p.take(p.ok)
            parts.append((fi, hit, sub))
            cls[hit] = fi
            year[hit], month[hit], dday[hit] = sub.year, sub.month, sub.day
            sod[hit] = sub.sod_us() if tpu == 1 else sub.sod_us() * tpu + sub.sub_us()
            off[hit] = sub.off_min
            yy[hit] = sub.yy
            tz[hit] = sub.tz_tok
            has_time[hit] = f.has_time
            if sub.dur_us is not None:
                dur[hit] = sub.dur_us if tpu == 1 else sub.dur_us * tpu + sub.sub_us()
            fl = sub.obs.get("f_len")
            if fl is not None:
                digits[hit] = np.where(np.isnan(fl), 0 if f.kind == "duration" else -1, fl).astype(np.int64)
        remaining = remaining[~p.ok]
    tdy = params.get("two_digit_year") or {}
    if tdy.get("shift") == "future" and yy.any():
        ref = ordinal(*[int(x) for x in str(tdy.get("reference", "2100-01-01")).split("-")])
        ld = days_from_civil(year, month, dday)
        move = yy & (cls >= 0) & (ld > ref)
        year = np.where(move, year - 100, year)
    local_day = days_from_civil(year, month, dday)
    mon = (year - 1970) * 12 + (month - 1)
    day = local_day.copy()
    sod_m = sod.copy()
    has_off = off != NO_OFFSET
    if has_off.any():
        # UTC = local - offset, carried into the day part (never a full instant: no int64 overflow in ns)
        carry, s2 = np.divmod(sod - np.where(has_off, off, 0) * 60 * US * tpu, US_DAY * tpu)
        day = np.where(has_off, local_day + carry, day)
        sod_m = np.where(has_off, s2, sod_m)
    return ColumnView(kind=kind, values=vals, cls=cls, day=day, sod=sod_m, dur=dur, mon=mon, local_day=local_day,
                      local_sod=sod, tz=tz, has_time=has_time, parts=parts, tpu=tpu, digits=digits)


# --------------------------------------------------------------------------- hooks
def detect_dates(ctx: DetectContext) -> Detection | None:
    """AREA A hook: decide whether the column holds dates/times/durations (module docstring).

    When called: by ``detect/types.py`` after EMPTY, CONSTANT/FLAG, BOOLEAN,
    ORDINAL and SCHOOL_YEAR, and before the text/PII hook, INTEGER_YEAR,
    ORG_CODE and numeric rules ("dates before string semantics"). Also called
    for ``--force-type COL=<date type>`` with ``ctx.forced_type`` set (None
    refuses: the file then FAILs with a fix).

    Inputs: ``ctx.typed_values``, ``ctx.raw_counts`` (whitespace variants),
    ``ctx.digit_code``, ``ctx.hints``, ``ctx.native`` and
    ``ctx.options.dayfirst / date_cols / date_format / calendar``.

    Returns a :class:`Detection` whose ``params`` carry everything the generator
    and renderer need, or None. Never claims 4-digit integer years or school
    years. ``--date-format`` values that do not match raise
    :class:`~datanonymous.errors.UnsupportedColumnError` (the file FAILs).
    """
    if not ctx.typed_values:
        return None
    return detect_from_inputs(DateInputs.from_context(ctx))


def detect_from_inputs(inp: DateInputs) -> Detection | None:
    """:func:`detect_dates` on prepared inputs (also the generator's fallback)."""
    det = _Detector(inp).run()
    if det is None:
        return None
    forced = inp.forced
    if forced is not None and det.type is not forced:
        # DATE and DATE_MIXED_FORMATS share one generator: the forced label wins
        if {forced, det.type} == {T.DATE, T.DATE_MIXED_FORMATS}:
            det.type = forced
        else:
            return None
    return det


def date_stats(values: list[str], profile: Any) -> dict[str, Any]:
    """AREA A hook: report-safe statistics of date-family values (design 3 #19-24).

    Called by ``profile/stats.py`` with the stripped non-missing values (real in
    pass 1, synthetic in pass 2). Day ordinals / seconds are summarised as ISO
    text (min, max, mean, median, quantiles) plus the numeric block
    ``ordinals``; dates add year/month/weekday pmfs, weekend share, rows per
    distinct date and sentinel counts; datetimes add time-of-day quantiles,
    resolution, midnight-only share and the offset/zone pmf; times an hour
    histogram; mixed formats their shares. Aggregates only.
    """
    return date_stats_counts(Counter(values), profile)


def date_stats_counts(counts: dict[str, int], profile: Any) -> dict[str, Any]:
    """AREA A statistics from counts in first-appearance order, without a row list."""
    params = getattr(profile, "params", None) or {}
    if not _format_keys(params) or not counts:
        return {}
    w = np.fromiter(counts.values(), dtype=np.int64, count=len(counts))
    values = list(counts)
    if any("\x00" in value for value in values):
        # Preserve the legacy string factorizer's NUL equivalence and weights.
        # Ordinary counts need no expanded row array.
        codes, uniques = pd.factorize(np.repeat(np.asarray(values, dtype=object), w))
        w = np.bincount(codes, minlength=len(uniques)).astype(np.int64)
        values = [str(value) for value in uniques]
    v = view(params, values)
    return _stats(v, w, params)


def date_numbers(values: list[str], profile: Any) -> np.ndarray | None:
    """AREA A hook: the numeric view of date-family values.

    Called through ``profile.stats.numeric_view`` (real values -> the sorted
    ``ColumnModel.sorted_numeric``; synthetic values -> the fidelity KS check).
    Units: days since 1970-01-01 (dates, DATE_AS_INT dates), months since
    1970-01 (YEAR_MONTH), seconds since the epoch in UTC when offsets are present
    (datetimes, epoch integers), seconds since midnight (TIME), seconds
    (DURATION). Sentinels and other tokens are NaN. None without params.
    """
    params = getattr(profile, "params", None) or {}
    if not _format_keys(params):
        return None
    if not values:
        return np.zeros(0)
    from ..profile.codes import codes_of

    codes, uniques = codes_of(np.asarray(values, dtype=object))
    nums = view(params, [str(u) for u in uniques]).numbers()
    return nums[codes]


def doc_fields(profile: Any, syn_stats: dict[str, Any]) -> dict[str, Any]:
    """AREA A hook: DATA_DICTIONARY / schema.json fields for a date-family column.

    ``format`` (or ``formats`` with shares for mixed columns), the readr letter
    (``'readr'``; replaces the type default) and an R parsing ``note``. readr's
    default parsers are used only where they read the exact format: ``D`` for
    zero-padded ``%Y-%m-%d``, ``T`` for zero-padded ISO datetimes with seconds
    (optional fraction and ``Z``/numeric offset), ``t`` for zero-padded 24-hour
    ``%H:%M[:%S]``; DATE_AS_INT stays ``i`` (``d`` for epoch milliseconds);
    everything else is ``c`` with a ``lubridate::parse_date_time`` call (or the
    matching R conversion) in ``note``. R classes come from the type
    (Date / POSIXct / hms / difftime; YEAR_MONTH is character); epoch
    integers (DATE_AS_INT, default Date) add ``r_class: 'POSIXct'``, which
    ``report/docs.py`` applies once it honours an ``r_class`` override like
    ``readr``. Shares and sentinel tokens come from ``syn_stats`` (D20).
    """
    params = getattr(profile, "params", None) or {}
    fmts = params.get("formats") or []
    keys = _format_keys(params)
    if not keys:
        return {}
    out: dict[str, Any] = {"date_kind": params.get("kind")}
    syn = syn_stats or {}
    if len(keys) == 1:
        out["format"] = keys[0]
    else:
        # D20: the out-dir docs are computed from the synthetic data only
        shares = syn.get("format_shares") or {}
        out["formats"] = {k: shares.get(k) for k in keys}
    t = getattr(profile, "type", None)
    if params.get("stored_as") in ("epoch_s", "epoch_ms"):
        out["r_class"] = "POSIXct"  # DATE_AS_INT's default class is Date; epoch numbers are instants
    readr, note = _readr(t, keys, fmts, params)
    out["readr"] = readr
    if note:
        out["note"] = note
    if syn.get("sentinels"):
        out["sentinel_tokens"] = sorted(syn["sentinels"])
    return out


# --------------------------------------------------------------------------- detector
class _Detector:
    def __init__(self, inp: DateInputs) -> None:
        self.inp = inp
        self.messages: list[Message] = []
        self.ambiguity: str | None = None
        self.dayfirst_used = False
        self.stored_as: str | None = None
        self.forced_format = False
        self._views: dict[tuple[str, ...], ColumnView] = {}

    # ------------------------------------------------------------------ driver
    def run(self) -> Detection | None:
        inp = self.inp
        typed = inp.typed
        values = list(typed)
        zero = [v for v in values if ZERO_DATE_RE.fullmatch(v)]
        rest = [v for v in values if not ZERO_DATE_RE.fullmatch(v)]
        if not rest:
            return None
        if inp.date_format:
            keys = self._explicit_format(rest)
            if keys == ["%Y%m%d"] and all(v.isascii() and v.isdigit() for v in rest):
                self.stored_as = "yyyymmdd"
        elif all(v.isascii() and v.isdigit() for v in rest):
            if inp.native is not None:
                return None
            got = self._integer_formats(rest)
            if got is None:
                return None
            keys, zero_int = got
            if zero_int:
                zero = zero + ["0"]
                rest = [v for v in rest if v != "0"]
        else:
            if inp.forced is T.DATE_AS_INT:
                return None
            keys = self._choose(rest)
            if keys is None:
                return None
        return self._build(values, zero, rest, keys)

    # ------------------------------------------------------------------ --date-format
    def _explicit_format(self, rest: list[str]) -> list[str]:
        inp = self.inp
        fmt = normalize_format(str(inp.date_format))
        try:
            f = get_format(fmt)
        except ValueError as e:
            raise UsageError(f"--date-format {inp.name}={inp.date_format}: {e}") from None
        p = f.parse(rest)
        cov = float(p.ok.mean()) if len(rest) else 0.0
        if cov < COVERAGE:
            raise UnsupportedColumnError(
                f"column {inp.name!r}: only {cov:.1%} of the distinct values match --date-format {fmt}",
                fix=f"check the format (strftime directives, e.g. %m/%d/%Y) or remove --date-format for {inp.name!r}")
        self.forced_format = True
        return [f.key]

    # ------------------------------------------------------------------ integers
    def _integer_formats(self, rest: list[str]) -> tuple[list[str], bool] | None:
        inp = self.inp
        if inp.digit_code or all(len(v) == 4 for v in rest):
            return None
        vals = [v for v in rest if v != "0"]
        zero_int = len(vals) < len(rest)
        if vals and all(len(v) == 8 for v in vals) and bool(get_format("%Y%m%d").parse(vals).ok.all()):
            self.stored_as = "yyyymmdd"
            return ["%Y%m%d"], zero_int
        if zero_int or not vals:
            return None
        allow = inp.date_cols or inp.forced is T.DATE_AS_INT
        h = inp.hints
        # an org-code header (EntryCode, Exit_Code: D10 keeps their real values) is not a date hint for
        # plain integers, even though 'entry'/'exit' are date words; --date-cols still applies
        date_hint = bool(h.date) and not h.org_code
        time_hint = bool(h.time) and not h.org_code
        if (allow or date_hint) and bool(get_format("@excel_serial").parse(vals).ok.all()):
            self.stored_as = "excel_serial"
            return ["@excel_serial"], False
        if allow or date_hint or time_hint:
            for key, label in (("@epoch_s", "epoch_s"), ("@epoch_ms", "epoch_ms")):
                if bool(get_format(key).parse(vals).ok.all()):
                    self.stored_as = label
                    return [key], False
        return None

    # ------------------------------------------------------------------ candidates
    def _eligible(self) -> list[str]:
        inp = self.inp
        kinds: tuple[str, ...] | None = None
        extra: list[str] = []
        if inp.native is not None:
            k = inp.native["kind"]
            kinds = ("date", "datetime") if k == "datetime" else (k,)
            if k == "date":
                extra = list(NATIVE_DATE_EXTRA)
        if inp.forced is not None:
            fk = _FORCED_KINDS.get(inp.forced, ())
            kinds = fk if kinds is None else tuple(k for k in kinds if k in fk)
        order = list(extra) + list(CANDIDATES)
        duration_first = (inp.hints.duration is not None or inp.forced is T.DURATION
                          or (inp.native is not None and inp.native["kind"] == "duration"))
        if duration_first:
            clock = [k for k in CLOCK_DURATIONS.values() if k in order]
            order = [k for k in order if k not in clock]
            first_time = order.index(TIME_FORMATS[0])
            order[first_time:first_time] = clock
        out = []
        for k in order:
            f = get_format(k)
            if kinds is not None and f.kind not in kinds:
                continue
            if inp.forced is T.DATETIME and f.has_tz:
                continue
            if inp.forced is T.DATETIME_TZ and f.kind == "datetime" and not f.has_tz:
                continue
            out.append(k)
        return out

    def _sample(self, rest: list[str]) -> list[str]:
        if len(rest) <= SAMPLE_MAX:
            return list(rest)
        rng = np.random.default_rng(20241004)
        idx = np.sort(rng.choice(len(rest), size=SAMPLE_MAX, replace=False))
        return [rest[i] for i in idx]

    def _choose(self, rest: list[str]) -> list[str] | None:
        inp = self.inp
        sample = self._sample(rest)
        n = len(sample)
        dateish = sum(1 for v in sample if len(v) <= 64 and _DATEISH_RE.fullmatch(v) and _HAS_DIGIT.search(v))
        if dateish < COVERAGE * n:
            return None
        groups: dict[str, list[int]] = {}
        for i, v in enumerate(sample):
            groups.setdefault(coarse_skeleton(v), []).append(i)
        need = MIN_GAIN * n
        arr = np.array(sample, dtype=object)
        res: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        order: list[str] = []
        for key in self._eligible():
            f = get_format(key)
            idx = [i for s, ids in groups.items() if f.matches_shape(s) for i in ids]
            if len(idx) < need:
                continue
            ia = np.array(idx, dtype=np.int64)
            p = f.parse(arr[ia])
            if int(p.ok.sum()) < need:
                continue
            covered = np.zeros(n, dtype=bool)
            covered[ia[p.ok]] = True
            vals = np.full(n, np.nan)
            vals[ia[p.ok]] = p.local_us()[p.ok].astype(float)
            res[key] = (covered, vals)
            order.append(key)
        if not order:
            return None
        chosen: list[str] = []
        covered = np.zeros(n, dtype=bool)
        group: str | None = None
        while len(chosen) < MAX_MIXED:
            gains = {k: int((res[k][0] & ~covered).sum()) for k in order
                     if k not in chosen and (group is None or kind_group(get_format(k).kind) == group)}
            if not gains:
                break
            g = max(gains.values())
            if g < need:
                break
            tied = [k for k in order if gains.get(k) == g]
            pick = self._tie(tied, res)
            chosen.append(pick)
            covered |= res[pick][0]
            group = kind_group(get_format(pick).kind)
            if covered.sum() >= COVERAGE * n:
                break
        if not chosen or covered.sum() < COVERAGE * n:
            return None
        chosen = self._time_or_duration(chosen, sample)
        if len(rest) > n:  # validate the choice on every distinct value
            v = self._base_view(chosen, rest)
            if float((v.cls >= 0).mean()) < COVERAGE:
                return None
        if inp.dayfirst and not self.dayfirst_used:
            self.messages.append(Message("info", "dayfirst_unused",
                                         "--dayfirst had no effect: the day/month order is not ambiguous",
                                         column=inp.name))
        return chosen

    def _tie(self, tied: list[str], res: dict[str, tuple[np.ndarray, np.ndarray]]) -> str:
        if len(tied) == 1:
            return tied[0]
        base = tied[0]

        def differs(a: str, b: str) -> bool:
            if get_format(a).kind != get_format(b).kind:  # e.g. TIME vs DURATION: decided by order and hints
                return False
            va, vb = res[a][1], res[b][1]
            both = ~np.isnan(va) & ~np.isnan(vb)
            return bool(np.any(va[both] != vb[both]))

        diff = [k for k in tied[1:] if differs(base, k)]
        pick = base
        md_pair = "%m" in base and "%d" in base and any(swapped_md(base) == k for k in tied[1:])
        if self.inp.dayfirst and md_pair:
            pick = next((k for k in tied if day_first(k)), base)
            self.dayfirst_used = True
        if diff:
            if md_pair and self.inp.dayfirst:
                self.messages.append(Message("info", "dayfirst", f"--dayfirst: day/month order read as {pick}",
                                             column=self.inp.name))
            elif md_pair:
                others = [k for k in diff if k != swapped_md(base)]
                self.ambiguity = (f"month/day order is ambiguous ({base} vs {swapped_md(base)}"
                                  + (f"; {', '.join(others)} also reads every value" if others else "")
                                  + "); month-first assumed")
            else:
                self.ambiguity = f"formats {base} and {diff[0]} both read every value; {base} assumed"
        return pick

    def _time_or_duration(self, chosen: list[str], sample: list[str]) -> list[str]:
        """Section 3 #23/#24: clock values with any hour >= 24 are durations."""
        inp = self.inp
        if any(get_format(k).kind != "time" for k in chosen) or inp.forced is T.TIME:
            return chosen
        if inp.native is not None and inp.native["kind"] == "time":
            return chosen
        if any(k not in CLOCK_DURATIONS for k in chosen):  # e.g. AM/PM times are never durations
            return chosen
        p = get_format("@dur_clock").parse(sample)
        if p.ok.mean() >= COVERAGE and p.dur_us is not None and np.any(np.abs(p.dur_us[p.ok]) >= 24 * 3600 * US):
            return self._clock_duration_cover(sample) or chosen
        return chosen

    @staticmethod
    def _clock_duration_cover(sample: list[str]) -> list[str] | None:
        """Greedy cover of the sample by the clock-duration layouts (``H:MM``, ``H:MM:SS``, ``H:MM:SS.f``)."""
        n = len(sample)
        hits = {k: get_format(k).parse(sample).ok for k in CLOCK_DURATIONS.values()}
        chosen: list[str] = []
        covered = np.zeros(n, dtype=bool)
        for k in sorted(hits, key=lambda k: -int(hits[k].sum())):  # stable: the layouts are disjoint
            if int((hits[k] & ~covered).sum()) >= MIN_GAIN * n and len(chosen) < MAX_MIXED:
                chosen.append(k)
                covered |= hits[k]
        return chosen if chosen and covered.sum() >= COVERAGE * n else None

    # ------------------------------------------------------------------ building the detection
    def _two_digit_year(self, keys: list[str], rest: list[str]) -> dict[str, Any] | None:
        if not any("%y" in k for k in keys):
            return None
        today = self.inp.today or _dt.date.today()
        ref = today.isoformat()
        info: dict[str, Any] = {"pivot": PIVOT_TEXT, "reference": ref, "shift": "none"}
        v0 = self._base_view(keys, rest)
        ok = v0.cls >= 0
        if not ok.any():
            return info
        ld = v0.local_day[ok]
        ref_ord = ordinal(today.year, today.month, today.day)
        yy_rows = np.zeros(len(rest), dtype=bool)
        for fi, rows, sub in v0.parts:
            yy_rows[rows] = sub.yy
        future = yy_rows[ok] & (ld > ref_ord)
        if not future.any():
            return info
        shifted = ld.copy()
        fy, fm, fd = civil_from_days(ld[future])
        shifted[future] = days_from_civil(fy - 100, fm, fd)
        if self.inp.dob:
            info["shift"] = "future"
            info["reason"] = "DOB-hinted column: values after the run date are 100 years earlier"
        elif (shifted.max() - shifted.min()) < (ld.max() - ld.min()):
            info["shift"] = "future"
            info["reason"] = "values after the run date are 100 years earlier (narrower range)"
        if info["shift"] == "future":
            self.messages.append(Message(
                "info", "two_digit_year",
                f"2-digit years: pivot 69; {int(future.sum())} distinct value(s) after {ref} read 100 years earlier "
                "for statistics and generation (the text keeps 2 digits)", column=self.inp.name))
        return info

    def _base_view(self, keys: list[str], rest: list[str]) -> ColumnView:
        """View of ``rest`` under bare ``keys`` (no shift, no tokens), cached per key list."""
        ck = tuple(keys)
        if ck not in self._views:
            self._views[ck] = view({"formats": [{"format": k} for k in keys]}, rest)
        return self._views[ck]

    def _build(self, values: list[str], zero: list[str], rest: list[str], keys: list[str]) -> Detection | None:
        inp = self.inp
        typed = inp.typed
        kind = column_kind(keys)
        params: dict[str, Any] = {"kind": kind, "formats": [{"format": k} for k in keys]}
        tdy = self._two_digit_year(keys, rest)
        if tdy is not None:
            params["two_digit_year"] = tdy
        if tdy is not None and tdy.get("shift") == "future":
            v = view(params, rest)
        else:
            v = self._base_view(keys, rest)
        w = np.fromiter((typed[x] for x in rest), dtype=np.int64, count=len(rest))
        parsed_share = float((v.cls >= 0).mean()) if len(rest) else 0.0
        if parsed_share < COVERAGE and not self.forced_format:
            return None
        sentinels: dict[str, int] = {z: int(typed[z]) for z in zero}
        others: list[str] = []
        replaced = 0
        gone: list[str] = []
        unparsed = np.flatnonzero(v.cls == UNPARSED)
        for i, how in zip(unparsed, leftover_kinds([rest[i] for i in unparsed], keys)):
            if how is not None:
                others.append(rest[i])
                v.cls[i] = OTHER_TOKEN
            else:
                replaced += int(w[i])
                gone.append(rest[i])
        if kind in ("date", "datetime", "year_month"):
            hit, far = self._sentinel_mask(v, w)
            for i in np.flatnonzero(hit):
                sentinels[rest[i]] = int(w[i])
            v.cls[hit] = SENTINEL
            if far.any():  # K-07: kept only when >= --min-cell entities hold each (profile.table)
                params[FAR_SENTINELS_KEY] = sorted(rest[i] for i in np.flatnonzero(far))
        if sentinels:
            params["sentinels"] = dict(sorted(sentinels.items(), key=lambda kv: (-kv[1], kv[0])))
        if others:
            params["other_tokens"] = int(sum(typed[x] for x in others))
            params["_other_tokens"] = sorted(others)
            self.messages.append(Message("info", "date_tokens",
                                         f"{len(others)} unparsed short value(s) ({params['other_tokens']} rows) "
                                         "are kept as tokens at their exact rates", column=inp.name))
        if replaced:
            params["unparsed_replaced"] = replaced
            # PRIVATE: real texts the column never writes (R7-01: no generator of the run writes one, privacy.texts)
            params[REPLACED_KEY] = sorted(gone)
            self.messages.append(Message("info", "date_unparsed_replaced",
                                         f"{replaced} row(s) hold unparsed text that is not a known placeholder "
                                         "token (names or other words, free text, digit strings, e-mail "
                                         "addresses); they get generated dates (such values are never copied)",
                                         column=inp.name,
                                         fix="--na-value TOKEN keeps a placeholder token verbatim"))
        dated = v.cls >= 0
        if not dated.any():
            return None
        wd = np.where(dated, w, 0)
        n_dated = int(wd.sum())
        # formats: shares and render specs (token rows weigh nothing)
        fmt_rows = np.zeros(len(keys))
        for fi, rows, sub in v.parts:
            ww = wd[rows]
            fmt_rows[fi] += float(ww.sum())
            params["formats"][fi]["render"] = get_format(keys[fi]).infer_render(sub, ww)
        used = [fi for fi in range(len(keys)) if fmt_rows[fi] > 0]
        if len(used) < len(keys):  # a format that only read tokens: drop it
            params["formats"] = [params["formats"][fi] for fi in used]
            keys = [keys[fi] for fi in used]
            fmt_rows = fmt_rows[used]
            v = view(params, rest)
            dated = v.cls >= 0
        for fi, f in enumerate(params["formats"]):
            f["share"] = round(float(fmt_rows[fi] / fmt_rows.sum()), 6) if fmt_rows.sum() else 0.0
        kind = column_kind(keys)
        params["kind"] = kind
        if self._time_unit(params, v, kind) == "ns":
            v = view(params, rest)
            dated = v.cls >= 0
        # time zones
        if any(get_format(k).has_tz for k in keys):
            params["tz_tokens"] = token_counts(v, w)
            params["tz_mode"] = "utc"
            zone = infer_zone(v, params)
            if zone is not None:  # R4-06: the offsets follow one zone's DST rules
                params["zone_rule"] = zone
        # calendar / person level / weekdays: the dates as written (local dates of values with offsets)
        if kind in ("date", "datetime"):
            days = v.local_day[dated]
            n_days = len(np.unique(days))
            ratio = n_dated / n_days if n_days else 0.0
            params["rows_per_distinct_date"] = round(ratio, 4)
            params["weekdays_observed"] = sorted(int(x) for x in np.unique(weekday(days)))
            params.update(date_grid(days))  # R2-13: month-precision, month-end, biweekly ... dates keep their grid
            self._calendar(params, ratio)
        elif kind == "year_month":
            mons = v.mon[dated]
            n_m = len(np.unique(mons))
            ratio = n_dated / n_m if n_m else 0.0
            params["rows_per_distinct_month"] = round(ratio, 4)
            mg = _lattice(mons)  # R2-13: quarterly / yearly months keep their grid
            if mg is not None:
                params["month_grid"] = mg
            self._calendar(params, ratio)
        if kind in ("datetime", "time", "duration"):
            if kind == "duration":
                tv = v.dur[dated]
            else:  # the time of day as written (an offset that is not a whole multiple keeps its own grid)
                tv = v.local_sod[dated & v.has_time]
            # R2-13: values are generated on the observed grid (any step, with its phase), not the nearest
            # fixed step: hourly values stay on the hour, '...:15' stays on :15
            step, phase = time_grid(tv, v.tpu)
            params["granularity_seconds"] = step / (US * v.tpu)
            if phase:
                params["grid_phase_seconds"] = phase / (US * v.tpu)
            params["resolution"] = resolution_label(granularity(np.abs(tv), v.tpu), v.tpu)
        if self.stored_as:
            params["stored_as"] = self.stored_as
        if inp.native is not None:
            params["native"] = dict(inp.native)
        ws = self._whitespace(params, int(np.where(v.cls >= 0, w, 0).sum()))
        if ws:
            params["whitespace"] = ws
        if inp.dayfirst and self.dayfirst_used:
            params["dayfirst"] = True
        # type
        typ, sub = self._type(kind, keys)
        reason = self._reason(typ, keys, params, parsed_share, len(rest))
        det = Detection(typ, subtype=sub, params=params, reason=reason, source="value")
        if inp.native is not None:
            det.source = "native"
        if self.ambiguity:
            det.ambiguous = True
            det.ambiguity = self.ambiguity
            self.messages.append(Message("warning", "ambiguous_date_order", self.ambiguity, column=inp.name,
                                         fix=(f"--dayfirst {inp.name} if the values are day-first"
                                              + (f", or --date-format {inp.name}=FMT"
                                                 if "also reads" in self.ambiguity else ""))
                                         if "month/day" in self.ambiguity
                                         else f"--date-format {inp.name}=FMT names the format"))
        if len(keys) > 1:
            shares = ", ".join(f"{f['format']} {f['share']:.1%}" for f in params["formats"])
            self.messages.append(Message("info", "date_mixed_formats", f"{len(keys)} formats: {shares}",
                                         column=inp.name))
        if params.get("sentinels"):
            self.messages.append(Message("info", "date_sentinels",
                                         f"{len(params['sentinels'])} sentinel token(s) "
                                         f"({sum(params['sentinels'].values())} rows) written verbatim at their rates",
                                         column=inp.name))
        det.messages.extend(self.messages)
        return det

    def _time_unit(self, params: dict[str, Any], v: ColumnView, kind: str) -> str:
        """``'ns'`` (sets ``params['time_unit']``) when dated values carry sub-microsecond digits.

        Fraction digits 7-9 (Arrow ``timestamp[ns]``/``time64[ns]``/``duration[ns]``,
        or text such as ``12:00:00.123456789``) are then modelled in nanoseconds
        instead of being cut to microseconds, as long as every value fits int64
        nanoseconds (instants in 1677-2262, durations under about 292 years);
        otherwise the digits beyond the microsecond are written as zeros (warning).
        """
        if kind not in ("datetime", "time", "duration"):
            return "us"
        dated = v.cls >= 0
        sub = False
        for _, rows, part in v.parts:
            if part.nano is not None and bool(np.any(part.nano[dated[rows]] != 0)):
                sub = True
                break
        if not sub:
            return "us"
        if kind == "duration":
            safe = bool(np.all(np.abs(v.dur[dated]) <= NS_SAFE_DUR_US))
        elif kind == "datetime":
            days = np.concatenate([v.local_day[dated], v.day[dated]])
            safe = bool(np.all(np.abs(days) <= NS_SAFE_DAYS))
        else:
            safe = True
        if not safe:
            self.messages.append(Message(
                "warning", "date_subus_truncated",
                "sub-microsecond digits are written as zeros: some values are outside the nanosecond range "
                "(instants 1677-2262, durations under 292 years)", column=self.inp.name))
            return "us"
        params["time_unit"] = "ns"
        return "ns"

    def _sentinel_mask(self, v: ColumnView, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(sentinel mask, its far-from-median part): the fixed sentinel dates (:data:`FIXED_SENTINELS`, as a date: a
        zero day with a time of day is a bare time) always, and,
        K-07, never in a person-level (DOB-hinted) column, values held by >= 1% of the rows and by >= ``--min-cell``
        rows that lie >= 50 years from the median (:data:`FAR_SENTINELS_KEY`: kept only when ``--min-cell`` entities
        hold them, ``profile.table``; inside :func:`far_sentinels_only`, only the values it names)."""
        dated = v.cls >= 0
        hits = np.zeros(len(v.cls), dtype=bool)
        none = np.zeros(len(v.cls), dtype=bool)
        if not dated.any():
            return hits, none
        if v.kind != "year_month":
            # a zero day with a time of day is a bare time (FIXED_SENTINELS), the end of time at any time
            timed = v.has_time & (v.local_sod != 0) & (v.local_day != ordinal(*END_OF_TIME))
            hits |= dated & ~timed & np.isin(v.local_day, [ordinal(*d) for d in FIXED_SENTINELS])
        keep = dated & ~hits
        if not keep.any() or self.inp.dob:  # a real far-off birth date is a person's date, never a token (K-07)
            return hits, none
        total = float(sum(self.inp.typed.values()))
        ld, kw = v.local_day[keep], w[keep]
        order = np.argsort(ld, kind="stable")
        cum = np.cumsum(kw[order])
        median = float(ld[order][np.searchsorted(cum, cum[-1] / 2.0)])
        far = np.abs(v.local_day.astype(float) - median) >= SENTINEL_YEARS * 365.25
        far &= keep & (w >= max(SENTINEL_MIN_SHARE * total, float(self.inp.min_cell)))
        allowed = _FAR_ALLOWED.get()
        if allowed is not None and far.any():
            far &= np.fromiter((x in allowed for x in v.values), dtype=bool, count=len(v.values))
        return hits | far, far

    def _calendar(self, params: dict[str, Any], ratio: float) -> None:
        inp = self.inp
        person = inp.dob
        params["person_level"] = person
        if person:
            params["calendar"] = False
            params["calendar_reason"] = "person-level date (DOB hint): never restricted to observed dates"
            if inp.calendar:
                self.messages.append(Message("warning", "calendar_ignored",
                                             "--calendar ignored: person-level dates (DOB) are never restricted "
                                             "to observed dates", column=inp.name))
        elif inp.calendar:
            params["calendar"] = True
            params["calendar_reason"] = "--calendar"
        elif ratio >= inp.min_cell:
            params["calendar"] = True
            params["calendar_reason"] = f"{ratio:.1f} rows per distinct date >= min-cell {inp.min_cell}"
        else:
            params["calendar"] = False
            params["calendar_reason"] = f"{ratio:.1f} rows per distinct date < min-cell {inp.min_cell}"

    def _whitespace(self, params: dict[str, Any], n_dated: int) -> list[list[Any]]:
        sentinels = params.get("sentinels") or {}
        others = set(params.get("_other_tokens") or ())
        counts: Counter = Counter()
        for raw in padded_raws(self.inp.raw_counts):
            s = raw.strip()
            if not s or s in sentinels or s in others or s not in self.inp.typed:
                continue
            counts[(raw[: len(raw) - len(raw.lstrip())], raw[len(raw.rstrip()):])] += self.inp.raw_counts[raw]
        if not counts:
            return []
        counts[("", "")] += max(n_dated - sum(counts.values()), 0)
        return [[a, b, int(c)] for (a, b), c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])) if c]

    def _type(self, kind: str, keys: list[str]) -> tuple[ColumnType, str | None]:
        inp = self.inp
        if self.stored_as:
            return T.DATE_AS_INT, self.stored_as
        if inp.native is not None:
            nk = inp.native["kind"]
            if nk == "date":
                return T.DATE, None
            if nk == "datetime":
                return (T.DATETIME_TZ if any(get_format(k).has_tz for k in keys) or inp.native.get("tz")
                        else T.DATETIME), None
        if kind == "datetime":
            return (T.DATETIME_TZ if any(get_format(k).has_tz for k in keys) else T.DATETIME), None
        if kind == "date":
            return (T.DATE_MIXED_FORMATS if len(keys) > 1 or inp.forced is T.DATE_MIXED_FORMATS else T.DATE), None
        return _KIND_TYPES[kind], None

    def _reason(self, typ: ColumnType, keys: list[str], params: dict[str, Any], share: float, n: int) -> str:
        if self.stored_as == "yyyymmdd":
            base = "8-digit integers, valid %Y%m%d dates"
        elif self.stored_as == "excel_serial":
            base = "Excel day serials 10000-73050 (date hint or --date-cols)"
        elif self.stored_as in ("epoch_s", "epoch_ms"):
            base = f"Unix epoch {'seconds' if self.stored_as == 'epoch_s' else 'milliseconds'} (hint or --date-cols)"
        elif len(keys) > 1:
            base = "mixed formats " + ", ".join(f"{f['format']} ({f['share']:.0%})" for f in params["formats"])
        else:
            base = f"format {keys[0]}"
        src = " (--date-format)" if self.forced_format else (" (native)" if self.inp.native else "")
        return f"{typ.value.lower().replace('_', ' ')}: {base}{src}; {share:.1%} of {n} distinct values read"


PIVOT_TEXT = "69 (69-99 -> 19xx, 00-68 -> 20xx)"


def leftover_kinds(values: Sequence[str], keys: Sequence[str]) -> list[str | None]:
    """How each unparsed value of a date column may be written back (module docstring, step 6).

    ``'date'``: an out-of-range date in one of the detected layouts ``keys``
    (same character shape, invalid components: ``02/30/2024``, ``99/99/9999``;
    section 4), or a valid date/time in another candidate layout too rare to
    be a format (``15MAR2024`` among ISO dates; never a bare digit string such
    as ``20230512``, which may be an ID); written verbatim as a token.
    ``'word'``: a short placeholder (<= 20 characters, <= 2 words, no digit
    other than 0, no ``@``) whose words all come from :data:`PLACEHOLDER_WORDS`
    or that has no letter at all: ``TBD``, ``Unknown``, ``Not enrolled``, ``0``,
    ``-``. None: anything else (person names and other unknown words, free
    text, digit strings such as IDs or phone numbers, e-mail addresses), which
    is never copied.
    """
    out: list[str | None] = [None] * len(values)
    if not values:
        return out
    # out-of-range values have the character shape of a strftime layout (special forms such as ISO
    # durations have shapes loose enough to fit codes like 'A1234B': they only count when valid)
    detected = [get_format(k) for k in keys if not k.startswith("@")]
    groups: dict[str, list[int]] = {}
    for i, x in enumerate(values):
        sk = coarse_skeleton(x)
        if any(f.matches_shape(sk) for f in detected):
            out[i] = "date"
        elif not x.isdigit() and len(x) <= 64 and _HAS_DIGIT.search(x) and _DATEISH_RE.fullmatch(x):
            # (a bare digit string such as 20230512 may be an ID: never read as another layout)
            groups.setdefault(sk, []).append(i)
    if groups:  # a valid value of any other candidate layout
        for key in CANDIDATES:
            f = get_format(key)
            idx = [i for sk, ids in groups.items() if f.matches_shape(sk) for i in ids if out[i] is None]
            if idx:
                ok = f.parse([values[i] for i in idx]).ok
                for i, good in zip(idx, ok.tolist()):
                    if good:
                        out[i] = "date"
    for i, x in enumerate(values):
        if out[i] is not None:
            continue
        if len(x) > TOKEN_MAX_CHARS or len(x.split()) > TOKEN_MAX_WORDS or "@" in x:
            continue
        if any(ch.isdigit() and ch != "0" for ch in x):
            continue
        if not is_placeholder_text(x):
            continue  # an unknown word (a name, a note): never copied (R1-02)
        out[i] = "word"
    return out


def is_placeholder_text(x: str) -> bool:
    """Every letter run of ``x`` is a :data:`PLACEHOLDER_WORDS` word (a value without letters qualifies)."""
    return all(w.casefold() in PLACEHOLDER_WORDS for w in _LETTER_RUN.findall(x))


def padded_raws(raw_counts: dict[Any, int]) -> list[str]:
    """Raw string values with leading or trailing whitespace (Arrow pre-filter on large columns)."""
    raws = [r for r in raw_counts if isinstance(r, str)]
    if len(raws) <= 20000:
        return [r for r in raws if r != r.strip()]
    a = pa.array(raws, type=pa.string())
    diff = pc.not_equal(pc.utf8_length(a), pc.utf8_length(pc.utf8_trim_whitespace(a)))
    return [raws[i] for i in np.flatnonzero(np.asarray(diff.to_numpy(zero_copy_only=False), dtype=bool))]


def token_counts(v: ColumnView, w: np.ndarray) -> dict[str, int]:
    """Row counts of the offset/zone tokens of dated values."""
    sel = (v.cls >= 0) & (v.tz != "")
    if not sel.any():
        return {}
    s = pd.Series(w[sel]).groupby(v.tz[sel]).sum()
    return {str(k): int(c) for k, c in sorted(s.items(), key=lambda kv: (-kv[1], str(kv[0])))}


#: Zones tried, in this order, for a column whose written UTC offsets change during the year (DST): when every
#: value's offset is the one a zone's rules give at its instant, the offsets are derived from that zone
#: (``params['zone_rule']``, :func:`infer_zone`). US zones first; zones without DST never qualify.
ZONE_CANDIDATES = (
    "America/New_York", "America/Chicago", "America/Denver", "America/Los_Angeles", "America/Anchorage",
    "America/Adak", "America/Halifax", "America/St_Johns", "America/Mexico_City", "America/Tijuana",
    "America/Havana", "America/Santiago", "America/Asuncion", "America/Sao_Paulo", "Europe/London",
    "Europe/Dublin", "Europe/Lisbon", "Europe/Paris", "Europe/Athens", "Europe/Moscow", "Atlantic/Azores",
    "Asia/Jerusalem", "Asia/Beirut", "Africa/Cairo", "Asia/Tehran", "Australia/Sydney", "Australia/Adelaide",
    "Australia/Lord_Howe", "Pacific/Auckland", "Pacific/Chatham", "Pacific/Norfolk",
)
#: Values checked against every candidate zone before a zone is verified on all of them.
_ZONE_PROBE = 256


def zone_rule(params: dict[str, Any]) -> str | None:
    """The IANA zone whose rules give the column's UTC offsets, or None (offsets drawn as observed tokens).

    A native timestamp column with a named zone (``timestamp[us, tz=America/Chicago]``: ``params['native']
    ['tz']``) is read in that zone and written back to it, so every offset is the zone's at the value's
    instant. A text column gets one from :func:`infer_zone` (``params['zone_rule']``). R4-06.
    """
    native = params.get("native")
    if isinstance(native, dict) and rule_zone(native.get("tz")) is not None:
        return str(native["tz"])
    z = params.get("zone_rule")
    return str(z) if rule_zone(z) is not None else None


def infer_zone(v: ColumnView, params: dict[str, Any]) -> str | None:
    """The zone of :data:`ZONE_CANDIDATES` whose rules give every dated value's written UTC offset, or None.

    Only for fixed offsets and abbreviations (``-06:00``/``-05:00``, ``CST``/``CDT``) taking at least two
    values: a single offset needs no rule, and zone-name tokens carry their own rules. A native named zone is
    already known (:func:`zone_rule`).
    """
    native = params.get("native")
    if isinstance(native, dict) and rule_zone(native.get("tz")) is not None:
        return None
    sel = np.flatnonzero((v.cls >= 0) & (v.tz != ""))
    if sel.size < 2:
        return None
    toks = v.tz[sel]
    uniq = [str(t) for t in pd.unique(toks)]
    fixed = {t: token_offset(t) for t in uniq}
    if any(o is None for o in fixed.values()) or len(set(fixed.values())) < 2:
        return None
    off = pd.Series(toks, dtype=object).map(fixed).to_numpy(dtype=np.int64)
    tpu = v.tpu
    inst = np.floor_divide(v.day[sel] * (US_DAY * tpu) + v.sod[sel], tpu)
    probe = np.unique(np.linspace(0, sel.size - 1, min(sel.size, _ZONE_PROBE)).astype(np.int64))
    for name in ZONE_CANDIDATES:
        if rule_zone(name) is None:
            continue
        zo = zone_instant_offsets(name, inst[probe])
        if zo is None or not np.array_equal(zo, off[probe]):
            continue
        zo = zone_instant_offsets(name, inst)
        if zo is not None and np.array_equal(zo, off):
            return name
    return None


def granularity(values_us: np.ndarray, tpu: int = 1) -> int:
    """The written unit: largest of 30/15/5/1 min, 1 s, 1 ms, 1 us (1 ns) dividing every value.

    ``tpu`` ticks per us. Only the ``resolution`` label comes from it; the grid
    values are generated on is :func:`time_grid` (R2-13).
    """
    v = np.asarray(values_us, dtype=np.int64)
    if v.size == 0:
        return 60 * US * tpu
    for step in GRANULARITY_STEPS_US:
        if not np.any(v % (step * tpu)):
            return int(step * tpu)
    return 1


def time_grid(values: np.ndarray, tpu: int = 1) -> tuple[int, int]:
    """The observed grid ``phase + k * step`` of times of day, durations or instants (ticks; R2-13).

    ``step`` is the greatest common divisor of the differences between the
    observed values, ``phase`` the values' common remainder modulo ``step``
    (0 when the grid starts at midnight / zero). Every observed value lies on
    the grid, and it is the coarsest such grid: hourly values give 1 hour,
    every 10 minutes 10 minutes, every hour at :15 a 1-hour step with a
    15-minute phase, whole-day durations 24 hours, 100 ms ticks 100 ms (not
    only the fixed 30/15/5/1-minute steps). A single distinct value keeps
    :func:`granularity` (nothing to generate between). ``tpu``: ticks per us.
    """
    v = np.unique(np.asarray(values, dtype=np.int64))
    if v.size < 2:
        return granularity(v, tpu), 0
    step = int(np.gcd.reduce(v[1:] - v[0]))
    return step, int(v[0] % step)


def _lattice(values: np.ndarray) -> list[int] | None:
    """[step, phase] of the observed integer lattice of ``values`` when its step is >= 2, else None."""
    u = np.unique(np.asarray(values, dtype=np.int64))
    if u.size < 2:
        return None
    step = int(np.gcd.reduce(u[1:] - u[0]))
    return [step, int(u[0] % step)] if step >= 2 else None


def month_length(year: np.ndarray, month: np.ndarray) -> np.ndarray:
    """Days in each (year, month) (proleptic Gregorian)."""
    y = np.asarray(year, dtype=np.int64)
    m = np.asarray(month, dtype=np.int64)
    return days_from_civil(y + (m == 12), m % 12 + 1, np.ones_like(m)) - days_from_civil(y, m, np.ones_like(m))


def date_grid(days: np.ndarray) -> dict[str, Any]:
    """The observed calendar grid of day ordinals (dates as written; R2-13), as detection params.

    ``day_grid`` [step, phase]: every value is ``phase + k * step`` days with
    step >= 2 (biweekly pay dates: 14). ``day_of_month``: every value falls on
    that day of its month (month-precision dates on the 1st, the 15th), else
    ``month_end``: every value is its month's last day. With either, a
    ``month_grid`` [step, phase] over month ordinals when step >= 2 (quarterly:
    3, yearly anniversaries: 12). Empty with fewer than 2 distinct dates or no
    such structure; observed weekdays are kept separately (``weekdays_observed``).
    """
    u = np.unique(np.asarray(days, dtype=np.int64))
    out: dict[str, Any] = {}
    if u.size < 2:
        return out
    dg = _lattice(u)
    if dg is not None:
        out["day_grid"] = dg
    y, m, d = civil_from_days(u)
    if np.all(d == d[0]):
        out["day_of_month"] = int(d[0])
    elif np.all(d == month_length(y, m)):
        out["month_end"] = True
    if "day_of_month" in out or "month_end" in out:
        mg = _lattice((y - 1970) * 12 + (m - 1))
        if mg is not None:
            out["month_grid"] = mg
    return out


def resolution_label(step_us: int, tpu: int = 1) -> str:
    if step_us >= 60 * US * tpu:
        return "minute"
    if step_us >= US * tpu:
        return "second"
    if step_us >= 1000 * tpu:
        return "millisecond"
    if step_us >= tpu:
        return "microsecond"
    return "nanosecond"


# --------------------------------------------------------------------------- statistics
def _r(x: float) -> float:
    return float(round(float(x), 6))


def _pmf(keys: np.ndarray, w: np.ndarray) -> dict[str, int]:
    u, inv = np.unique(np.asarray(keys, dtype=np.int64), return_inverse=True)
    c = np.bincount(inv, weights=w)
    return {str(int(a)): int(b) for a, b in zip(u, c)}


def _wquantiles(x: np.ndarray, w: np.ndarray, qs: Sequence[float]) -> np.ndarray:
    order = np.argsort(x, kind="stable")
    xs, ws = x[order].astype(float), w[order].astype(float)
    cum = np.cumsum(ws)
    total = cum[-1]
    # type-7 quantiles of the expanded sample, computed without expanding
    pos = np.asarray(qs) * (total - 1)
    lo = np.floor(pos)
    frac = pos - lo
    i_lo = np.searchsorted(cum, lo + 1)
    i_hi = np.searchsorted(cum, np.minimum(lo + 2, total))
    return xs[i_lo] + frac * (xs[i_hi] - xs[i_lo])


def _summary(x: np.ndarray, w: np.ndarray, fmt: Any, unit: str,
             exact: tuple[str, str] | None = None) -> dict[str, Any]:
    """Weighted summary of ``x``; ``exact`` overrides (min, max) with text from exact integers
    (nanosecond values lose digits as floats)."""
    total = float(w.sum())
    mean = float((x * w).sum() / total)
    sd = float(np.sqrt(max(((x - mean) ** 2 * w).sum() / (total - 1), 0.0))) if total > 1 else 0.0
    q = _wquantiles(x, w, QUANTILES)
    med = float(_wquantiles(x, w, (0.5,))[0])
    lo, hi = exact if exact is not None else (fmt(x.min()), fmt(x.max()))
    return {
        "min": lo, "max": hi, "mean": fmt(mean), "median": fmt(med), "sd": _r(sd),
        "sd_unit": unit,
        "quantiles": {f"p{int(round(p * 100))}": fmt(v) for p, v in zip(QUANTILES, q)},
        "ordinal_unit": unit,
    }


def _exact_extremes(v: ColumnView, dated: np.ndarray, show: Any) -> tuple[str, str] | None:
    """(min, max) text from exact integer ticks for nanosecond columns (None otherwise)."""
    if v.tpu == 1:
        return None
    if v.kind == "datetime":  # lexicographic (day, time of day): never a full int64 instant
        day, sod = v.day[dated], v.sod[dated]
        order = np.lexsort((sod, day))
        a, b = order[0], order[-1]
        per = US_DAY * v.tpu
        return show(int(day[a]) * per + int(sod[a])), show(int(day[b]) * per + int(sod[b]))
    t = v.sod[dated] if v.kind == "time" else v.dur[dated]
    return show(int(t.min())), show(int(t.max()))


def _stats(v: ColumnView, w: np.ndarray, params: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"date_kind": v.kind}
    dated = v.cls >= 0
    toks = Counter()
    for i in np.flatnonzero(v.cls == SENTINEL):
        toks[v.values[i]] += int(w[i])
    if toks:
        out["sentinels"] = dict(sorted(toks.items(), key=lambda kv: (-kv[1], kv[0])))
    n_other = int(w[v.cls == OTHER_TOKEN].sum())
    n_unparsed = int(w[v.cls == UNPARSED].sum())
    if n_other:
        out["other_tokens"] = n_other
    if n_unparsed:
        out["unparsed"] = n_unparsed
    if not dated.any():
        return out
    wd = w[dated]
    out["n_dated"] = int(wd.sum())
    keys = _format_keys(params)
    if len(keys) > 1:
        shares: dict[str, float] = {}
        tot = float(wd.sum())
        for fi, k in enumerate(keys):
            shares[k] = _r(float(w[v.cls == fi].sum()) / tot)
        out["format_shares"] = shares
    k = v.kind
    utc = bool(params.get("tz_tokens"))
    tpu = v.tpu
    tps = US * tpu
    if k in ("date", "datetime"):
        days = v.day[dated]
        if k == "date":
            out.update(_summary(days.astype(float), wd, lambda d: iso_day(int(round(d))), "days since 1970-01-01"))
        else:
            secs = v.numbers()[dated]
            out.update(_summary(secs, wd, lambda s: iso_instant(int(round(s * tps)), utc, tpu),
                                "seconds since 1970-01-01" + (" UTC" if utc else ""),
                                _exact_extremes(v, dated, lambda t: iso_instant(t, utc, tpu))))
            tod = v.sod[dated & v.has_time]
            tw = w[dated & v.has_time]
            if tod.size:
                q = _wquantiles(tod.astype(float), tw, QUANTILES)
                out["time_of_day_quantiles"] = {f"p{int(round(p * 100))}": hms(int(round(x)), tpu=tpu) for p, x in
                                                zip(QUANTILES, q)}
                out["midnight_share"] = _r(float(tw[tod == 0].sum()) / float(tw.sum()))
                out["resolution"] = resolution_label(granularity(tod, tpu), tpu)
        y, m, _ = civil_from_days(days)
        wdays = weekday(days)
        yp = _pmf(y, wd)
        if len(yp) <= MAX_YEAR_PMF:
            out["year_pmf"] = yp
        else:
            out["year_pmf_omitted"] = f"{len(yp)} distinct years"
        mp = np.bincount(m, weights=wd, minlength=13)[1:]
        out["month_pmf"] = {str(i + 1): int(c) for i, c in enumerate(mp)}
        wp = np.bincount(wdays, weights=wd, minlength=7)
        names = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
        out["weekday_pmf"] = {names[i]: int(c) for i, c in enumerate(wp)}
        out["weekend_share"] = _r(float(wp[5:].sum()) / float(wd.sum()))
        n_days = len(np.unique(days))
        out["distinct_dates"] = n_days
        out["rows_per_distinct_date"] = _r(float(wd.sum()) / n_days)
        if utc:
            out["tz_pmf"] = token_counts(v, w)
    elif k == "year_month":
        mons = v.mon[dated]
        out.update(_summary(mons.astype(float), wd, lambda x: iso_month(int(round(x))), "months since 1970-01"))
        yp = _pmf(mons // 12 + 1970, wd)
        if len(yp) <= MAX_YEAR_PMF:
            out["year_pmf"] = yp
        mp = np.bincount(mons % 12, weights=wd, minlength=12)
        out["month_pmf"] = {str(i + 1): int(c) for i, c in enumerate(mp)}
        out["rows_per_distinct_month"] = _r(float(wd.sum()) / len(np.unique(mons)))
    elif k == "time":
        sod = v.sod[dated]
        out.update(_summary(sod.astype(float) / tps, wd, lambda s: hms(int(round(s * tps)), tpu=tpu),
                            "seconds since midnight", _exact_extremes(v, dated, lambda t: hms(t, tpu=tpu))))
        hp = np.bincount(sod // (3600 * tps), weights=wd, minlength=24)
        out["hour_pmf"] = {str(i): int(c) for i, c in enumerate(hp) if c}
        out["granularity_minutes"] = _r(time_grid(sod, tpu)[0] / (60 * tps))
        out["resolution"] = resolution_label(granularity(sod, tpu), tpu)
    else:
        dur = v.dur[dated]
        out.update(_summary(dur.astype(float) / tps, wd, lambda s: hms(int(round(s * tps)), signed=True, tpu=tpu),
                            "seconds", _exact_extremes(v, dated, lambda t: hms(t, signed=True, tpu=tpu))))
        out["granularity_seconds"] = _r(time_grid(dur, tpu)[0] / tps)
        out["resolution"] = resolution_label(granularity(np.abs(dur), tpu), tpu)
        out["over_24h_share"] = _r(float(wd[np.abs(dur) >= 24 * 3600 * tps].sum()) / float(wd.sum()))
    return out


# --------------------------------------------------------------------------- R / readr
_READR_D = ("%Y-%m-%d",)
_READR_T = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S.%f")
_READR_TIME = ("%H:%M:%S", "%H:%M")


def _padded(fmt_entry: dict[str, Any]) -> bool:
    pad = (fmt_entry.get("render") or {}).get("pad") or {}
    return all(pad.values())


def _readr(t: Any, keys: list[str], fmts: list[dict[str, Any]], params: dict[str, Any]) -> tuple[str, str | None]:
    stored = params.get("stored_as")
    if t is T.DATE_AS_INT or stored:
        if stored == "excel_serial":
            return "i", "Excel day serials: as.Date(x, origin = \"1899-12-30\")"
        if stored == "epoch_s":
            return "d", "Unix seconds: as.POSIXct(x, origin = \"1970-01-01\", tz = \"UTC\")"
        if stored == "epoch_ms":
            return "d", "Unix milliseconds: as.POSIXct(x / 1000, origin = \"1970-01-01\", tz = \"UTC\")"
        return "i", "YYYYMMDD integers: as.Date(as.character(x), format = \"%Y%m%d\")"
    has_tokens = bool(params.get("sentinels") or params.get("other_tokens"))
    if len(keys) == 1 and not has_tokens and _padded(fmts[0]):
        k = keys[0]
        if t is T.DATE and k in _READR_D:
            return "D", None
        if t in (T.DATETIME, T.DATETIME_TZ):
            base = k[:-2] if k.endswith("%z") else k
            if base in _READR_T:
                return "T", "ISO 8601 datetimes; offsets are converted to UTC" if k.endswith("%z") else None
        if t is T.TIME and k in _READR_TIME:
            return "t", None
    kind = params.get("kind")
    if kind == "duration":
        return "c", "durations as text: parse with lubridate::duration(), hms::as_hms() or a split on ':'"
    special = [k for k in keys if k.startswith("@")]
    strf = [k for k in keys if not k.startswith("@")]
    if not strf:
        return "c", None
    orders = ", ".join(f'"{k}"' for k in strf)
    call = (f"lubridate::parse_date_time(x, orders = c({orders}), exact = TRUE)" if len(strf) > 1
            else f"lubridate::parse_date_time(x, orders = {orders}, exact = TRUE)")
    note = "read as character; parse with " + call
    if any("%y" in k for k in strf):
        note += "; two-digit years use the same pivot (00-68 -> 20xx)"
        if (params.get("two_digit_year") or {}).get("shift") == "future":
            note += ", except values after the run date, which are 100 years earlier"
    if any("%Z" in k for k in strf):
        note += "; R's strptime does not read zone names (%Z): drop the zone text and set tz ="
    if has_tokens:
        note += "; sentinel/other tokens become NA"
    if special:
        note += "; other forms: " + ", ".join(special)
    return "c", note


__all__ = [
    "ColumnView", "DATE_TYPES", "DateInputs", "date_grid", "date_numbers", "date_stats", "detect_dates",
    "detect_from_inputs", "doc_fields", "granularity", "infer_zone", "leftover_kinds", "month_length",
    "native_detection", "native_temporal", "ticks_per_us", "time_grid", "view", "zone_rule",
]
