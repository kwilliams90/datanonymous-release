"""Date, time and duration format engine (AREA A helper for ``detect/dates.py`` and ``generate/dates.py``).

Formats
-------
A *format key* is either a strftime-like string (``%m/%d/%Y``) or a named
special form:

=====================  ==========================================================
key                    values
=====================  ==========================================================
``%...`` strings       directives ``%Y %y %m %d %b %B %a %A %H %I %M %S %f %p %z
                       %Z`` and literal text (``%%`` is a literal percent sign).
                       ``%-m`` / ``%#m`` are accepted and mean ``%m``.
``@excel_serial``      Excel day serials 10000-73050 (1927-2099; origin 1899-12-30)
``@epoch_s``           Unix seconds 9e8-2.2e9
``@epoch_ms``          Unix milliseconds 9e11-2.2e12
``@dur_clock``         durations ``[-]H:MM[:SS[.f]]`` (hours unbounded; any layout)
``@dur_hm`` ...        one clock-duration layout each: ``@dur_hm`` ``H:MM``,
                       ``@dur_hms`` ``H:MM:SS``, ``@dur_hmsf`` ``H:MM:SS.f``
``@dur_iso``           ISO 8601 durations ``[-]P[nD][T[nH][nM][n[.f]S]]``
``@dur_days``          ``1 days 02:03:04[.f]`` (pandas) / ``1 day, 2:03:04`` (Python)
=====================  ==========================================================

Parsing never uses pandas' date inference: every format is one explicit,
anchored regular expression (RE2-compatible, run through ``Series.str.extract``)
and the components are validated with numpy (month 1-12, day within the month,
hour 0-23, minute/second 0-59, offsets within +-14:00). ``%m %d %H %I`` accept
one or two digits unless they touch another numeric field (``%Y%m%d`` then
needs exactly 8 digits); ``%M`` and ``%S`` always need two digits (``h:mm[:ss]``,
so ratios and scores such as ``15:1`` or ``3:2`` are never times); ``%y`` uses
the pivot 69 (69-99 -> 19xx, 00-68 -> 20xx); month and weekday names match in
any case. Fractions of a second (``%f`` and the duration forms) read 1-9
digits: microseconds plus the nanosecond remainder (``Parsed.nano``), which
nanosecond-precision columns keep (``detect/dates.py`` ``time_unit``).

Rendering never uses ``strftime`` (``%-m``/``%#m`` are platform-specific and
pandas silently ignores unknown directives): :meth:`DateFormat.render` builds
the text from integer components with numpy string ufuncs following a *render
spec* inferred from the real values by :meth:`DateFormat.infer_render`: zero
padding per component, month/weekday name case, AM/PM tokens, fraction digits,
duration layout. Offsets and zone names are written as observed tokens.

Every candidate key in :data:`CANDIDATES` (about 360, generated from the
explicit lists below) is tried in a fixed preference order (ISO first, then
month-first, then day-first, then month names); that order breaks ties.
"""

from __future__ import annotations

import re
import zoneinfo
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
from numpy.dtypes import StringDType

STR = StringDType()
US = 1_000_000  # microseconds per second
US_DAY = 86_400 * US
NS_PER_US = 1_000  # ticks per microsecond of a nanosecond-precision column (``params['time_unit'] == 'ns'``)
NO_OFFSET = -(1 << 30)  # off_min marker: value carries no usable UTC offset
PIVOT = 69  # %y: 69-99 -> 19xx, 00-68 -> 20xx (design section 10)

MONTHS_ABBR = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")
MONTHS_FULL = ("january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
               "november", "december")
WEEKDAYS_ABBR = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
WEEKDAYS_FULL = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
_MONTH_ABBR_IDX = {m: i + 1 for i, m in enumerate(MONTHS_ABBR)}
_MONTH_FULL_IDX = {m: i + 1 for i, m in enumerate(MONTHS_FULL)}

#: Fixed-offset zone abbreviations (minutes east of UTC). Other names are looked up as IANA zones.
TZ_ABBREVIATIONS: dict[str, int] = {
    "UTC": 0, "GMT": 0, "Z": 0, "UT": 0,
    "EST": -300, "EDT": -240, "CST": -360, "CDT": -300, "MST": -420, "MDT": -360,
    "PST": -480, "PDT": -420, "AKST": -540, "AKDT": -480, "HST": -600,
}

_DAYS_IN_MONTH = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype=np.int64)


# --------------------------------------------------------------------------- calendar arithmetic
def is_leap(y: np.ndarray) -> np.ndarray:
    return (y % 4 == 0) & ((y % 100 != 0) | (y % 400 == 0))


def days_in_month(y: np.ndarray, m: np.ndarray) -> np.ndarray:
    mm = np.clip(m, 1, 12)
    return _DAYS_IN_MONTH[mm - 1] + ((mm == 2) & is_leap(y)).astype(np.int64)


def days_from_civil(y: np.ndarray, m: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Day ordinal (days since 1970-01-01) of proleptic Gregorian dates (vectorised)."""
    y = np.asarray(y, dtype=np.int64) - (np.asarray(m) <= 2)
    m = np.asarray(m, dtype=np.int64)
    d = np.asarray(d, dtype=np.int64)
    era = np.floor_divide(y, 400)
    yoe = y - era * 400
    mp = (m + 9) % 12
    doy = (153 * mp + 2) // 5 + d - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * 146097 + doe - 719468


def civil_from_days(z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(year, month, day) of day ordinals (inverse of :func:`days_from_civil`)."""
    z = np.asarray(z, dtype=np.int64) + 719468
    era = np.floor_divide(z, 146097)
    doe = z - era * 146097
    yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
    y = yoe + era * 400
    doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
    mp = (5 * doy + 2) // 153
    d = doy - (153 * mp + 2) // 5 + 1
    m = np.where(mp < 10, mp + 3, mp - 9)
    return y + (m <= 2), m, d


def weekday(day_ord: np.ndarray) -> np.ndarray:
    """Monday = 0 ... Sunday = 6 (1970-01-01 was a Thursday)."""
    return (np.asarray(day_ord, dtype=np.int64) + 3) % 7


def ordinal(y: int, m: int, d: int) -> int:
    return int(days_from_civil(np.array([y]), np.array([m]), np.array([d]))[0])


def iso_day(day_ord: int) -> str:
    y, m, d = civil_from_days(np.array([int(day_ord)]))
    return f"{int(y[0]):04d}-{int(m[0]):02d}-{int(d[0]):02d}"


def iso_month(month_ord: int) -> str:
    y, m = divmod(int(month_ord), 12)
    return f"{y + 1970:04d}-{m + 1:02d}"


def hms(us: int, signed: bool = False, tpu: int = 1) -> str:
    """``[-]H:MM:SS[.ffffff]`` of a microsecond count (hours unbounded).

    ``tpu`` is the number of ticks per microsecond of ``us``: 1, or :data:`NS_PER_US`
    for nanosecond counts (written with 9 fraction digits).
    """
    us = int(us)
    sign = "-" if us < 0 else ""
    us = abs(us)
    s, frac = divmod(us, US * tpu)
    h, rem = divmod(s, 3600)
    mi, se = divmod(rem, 60)
    width = 6 if tpu == 1 else 9
    text = f"{h:02d}:{mi:02d}:{se:02d}" + (f".{frac:0{width}d}" if frac else "")
    return (sign if signed or sign else "") + text


def iso_instant(us: int, utc: bool, tpu: int = 1) -> str:
    day, sod = divmod(int(us), US_DAY * tpu)
    return f"{iso_day(day)}T{hms(sod, tpu=tpu)}" + ("Z" if utc else "")


# --------------------------------------------------------------------------- parse result
@dataclass
class Parsed:
    """Components of parsed values (one row per input value; ``ok`` marks valid rows).

    Date-less kinds use 1970-01-01; time-less kinds use midnight. ``off_min`` is
    the UTC offset in minutes or :data:`NO_OFFSET`; ``tz_tok`` the offset/zone
    token as written ('' when none); ``yy`` marks years read from ``%y``;
    ``dur_us`` holds durations (duration formats only). ``nano`` holds the
    sub-microsecond part (0-999 ns, fraction digits 7-9; signed like the
    duration) or None when no value has one. ``obs`` keeps raw observations for
    render-spec inference.
    """

    n: int
    ok: np.ndarray
    year: np.ndarray
    month: np.ndarray
    day: np.ndarray
    hour: np.ndarray
    minute: np.ndarray
    second: np.ndarray
    micro: np.ndarray
    off_min: np.ndarray
    tz_tok: np.ndarray
    yy: np.ndarray
    dur_us: np.ndarray | None = None
    obs: dict[str, np.ndarray] = field(default_factory=dict)
    nano: np.ndarray | None = None

    @classmethod
    def blank(cls, n: int) -> "Parsed":
        z = np.zeros(n, dtype=np.int64)
        tok = np.empty(n, dtype=object)
        tok[:] = ""
        return cls(n, np.zeros(n, dtype=bool), np.full(n, 1970, dtype=np.int64), np.ones(n, dtype=np.int64),
                   np.ones(n, dtype=np.int64), z.copy(), z.copy(), z.copy(), z.copy(),
                   np.full(n, NO_OFFSET, dtype=np.int64), tok, np.zeros(n, dtype=bool))

    def take(self, idx: np.ndarray) -> "Parsed":
        """Rows ``idx`` (boolean mask or integer index) as a new Parsed."""
        return Parsed(
            n=int(np.count_nonzero(idx)) if np.asarray(idx).dtype == bool else len(idx),
            ok=self.ok[idx], year=self.year[idx], month=self.month[idx], day=self.day[idx], hour=self.hour[idx],
            minute=self.minute[idx], second=self.second[idx], micro=self.micro[idx], off_min=self.off_min[idx],
            tz_tok=self.tz_tok[idx], yy=self.yy[idx],
            dur_us=None if self.dur_us is None else self.dur_us[idx],
            obs={k: v[idx] for k, v in self.obs.items()},
            nano=None if self.nano is None else self.nano[idx],
        )

    def sub_us(self) -> np.ndarray:
        """Sub-microsecond part in ns (zeros when the format has none)."""
        return np.zeros(self.n, dtype=np.int64) if self.nano is None else self.nano

    def day_ordinal(self) -> np.ndarray:
        return days_from_civil(self.year, self.month, self.day)

    def sod_us(self) -> np.ndarray:
        return ((self.hour * 60 + self.minute) * 60 + self.second) * US + self.micro

    def local_us(self) -> np.ndarray:
        return self.day_ordinal() * US_DAY + self.sod_us()

    def instant_us(self) -> np.ndarray:
        """UTC instant in microseconds (local time when there is no offset)."""
        off = np.where(self.off_min == NO_OFFSET, 0, self.off_min)
        return self.local_us() - off * 60 * US


@dataclass
class Components:
    """Integer components to render (local wall time), plus tokens.

    ``nano`` is the sub-microsecond part (0-999 ns; for durations of the
    absolute value) or None: only nanosecond-precision columns have one.
    """

    n: int
    year: np.ndarray
    month: np.ndarray
    day: np.ndarray
    hour: np.ndarray
    minute: np.ndarray
    second: np.ndarray
    micro: np.ndarray
    weekday: np.ndarray
    day_ord: np.ndarray
    tz: np.ndarray  # StringDType tokens ('' when none)
    dur_us: np.ndarray | None = None
    instant_us: np.ndarray | None = None
    nano: np.ndarray | None = None

    @classmethod
    def from_local(cls, day_ord: np.ndarray, sod_us: np.ndarray, tz: np.ndarray | None = None,
                   instant_us: np.ndarray | None = None, tpu: int = 1) -> "Components":
        """Components of local day ordinals and times of day (``tpu`` ticks per microsecond)."""
        day_ord = np.asarray(day_ord, dtype=np.int64)
        sod_us = np.asarray(sod_us, dtype=np.int64)
        nano = None
        if tpu > 1:
            sod_us, nano = np.divmod(sod_us, tpu)
        y, m, d = civil_from_days(day_ord)
        secs, micro = np.divmod(sod_us, US)
        h, rem = np.divmod(secs, 3600)
        mi, se = np.divmod(rem, 60)
        n = len(day_ord)
        if tz is None:
            tz = np.full(n, "", dtype=STR)
        return cls(n, y, m, d, h, mi, se, micro, weekday(day_ord), day_ord, np.asarray(tz, dtype=STR),
                   instant_us=instant_us, nano=nano)

    @classmethod
    def from_duration(cls, dur_us: np.ndarray, tpu: int = 1) -> "Components":
        """Components of durations (``tpu`` ticks per microsecond).

        With nanosecond ticks ``dur_us`` keeps the whole microseconds and ``nano``
        the remainder, both with the duration's sign (toward zero).
        """
        dur_us = np.asarray(dur_us, dtype=np.int64)
        nano = None
        if tpu > 1:
            sign = np.sign(dur_us)
            a_us, rem = np.divmod(np.abs(dur_us), tpu)
            dur_us, nano = sign * a_us, sign * rem
        c = cls.from_local(np.zeros(len(dur_us), dtype=np.int64), np.zeros(len(dur_us), dtype=np.int64))
        c.dur_us = dur_us
        c.nano = nano
        return c


def _dur_parts(comps: Components) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """(negative?, |whole microseconds|, |sub-microsecond ns| or None) of duration components."""
    dur = np.asarray(comps.dur_us, dtype=np.int64)
    if comps.nano is None:
        return dur < 0, np.abs(dur), None
    nano = np.asarray(comps.nano, dtype=np.int64)
    return (dur < 0) | (nano < 0), np.abs(dur), np.abs(nano)


def frac_value(micro: Any, nano: Any, k: int) -> Any:
    """The first ``k`` fraction digits (as an integer) of ``micro`` us plus ``nano`` ns (None = 0)."""
    if k <= 6:
        return micro // 10 ** (6 - k)
    v = micro * 10 ** (k - 6)
    return v if nano is None else v + nano // 10 ** (9 - k)


# --------------------------------------------------------------------------- helpers
_REGEX_META = set(".^$*+?()[]{}|\\")


def _escape(text: str) -> str:
    """Escape regex metacharacters only (RE2 and Python agree on this subset)."""
    return "".join("\\" + ch if ch in _REGEX_META else ch for ch in text)


_COARSE_TABLE = str.maketrans("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
                              "9" * 10 + "a" * 52)


def coarse_skeleton(value: str) -> str:
    """Digits -> '9', ASCII letters -> 'a', everything else kept (pre-filter key)."""
    return value.translate(_COARSE_TABLE)


def to_arrow(values: Any) -> pa.Array:
    """Values as an Arrow string array (None -> null)."""
    if isinstance(values, (pa.Array, pa.ChunkedArray)):
        return values
    arr = values if isinstance(values, np.ndarray) else np.asarray(list(values), dtype=object)
    try:
        return pa.array(arr, type=pa.string(), from_pandas=True)
    except (pa.ArrowInvalid, pa.ArrowTypeError):
        return pa.array([None if v is None else str(v) for v in arr.tolist()], type=pa.string())


def _np(a: Any, dtype: Any) -> np.ndarray:
    return np.array(a.to_numpy(zero_copy_only=False), dtype=dtype, copy=True)


class _Groups:
    """Named groups of an anchored regex over string values (pyarrow RE2, vectorised)."""

    _NULL = pa.scalar(None, pa.string())

    def __init__(self, values: Any, regex: str) -> None:
        a = to_arrow(values)
        self.n = len(a)
        self.st = pc.extract_regex(a, regex)
        self.ok = _np(pc.is_valid(self.st), bool)

    def given(self, g: str) -> pa.Array:
        """The group's text; null where the row did not match or the group is empty/absent."""
        f = pc.struct_field(self.st, g)
        return pc.if_else(pc.greater(pc.utf8_length(f), 0), f, self._NULL)

    def present(self, g: str) -> np.ndarray:
        return _np(pc.is_valid(self.given(g)), bool)

    def ints(self, g: str, fill: int = 0) -> np.ndarray:
        return _np(pc.fill_null(pc.cast(self.given(g), pa.int64()), fill), np.int64)

    def lens(self, g: str) -> np.ndarray:
        return _np(pc.fill_null(pc.cast(pc.utf8_length(self.given(g)), pa.float64()), np.nan), float)

    def obj(self, g: str) -> np.ndarray:
        return _np(self.given(g), object)

    def text(self, g: str) -> np.ndarray:
        """Group text with '' for nulls (object array)."""
        return _np(pc.fill_null(self.given(g), ""), object)

    def index_in(self, g: str, names: tuple[str, ...]) -> np.ndarray:
        """1-based position of the lower-cased group text in ``names`` (0 when absent)."""
        x = pc.index_in(pc.utf8_lower(self.given(g)), value_set=pa.array(names, type=pa.string()))
        return _np(pc.fill_null(x, -1), np.int64) + 1

    def micro(self, g: str) -> np.ndarray:
        """Fraction digits as microseconds (first 6 digits, right-padded)."""
        f = pc.utf8_rpad(pc.utf8_slice_codeunits(self.given(g), 0, 6), 6, padding="0")
        return _np(pc.fill_null(pc.cast(f, pa.int64()), 0), np.int64)

    def nano(self, g: str) -> np.ndarray:
        """Fraction digits 7-9 as nanoseconds (0-999; 0 for shorter fractions)."""
        f = pc.utf8_rpad(pc.utf8_slice_codeunits(self.given(g), 6, 9), 3, padding="0")
        return _np(pc.fill_null(pc.cast(f, pa.int64()), 0), np.int64)


def _digit_counts(lens: np.ndarray | None, ok: np.ndarray, w: np.ndarray,
                  bare: np.ndarray | None = None) -> dict[int, float]:
    """Row weight per observed fraction-digit count.

    ``bare`` marks values whose seconds are written without a fraction in a
    layout where the fraction is optional (ISO ``PT5S`` next to ``PT5.25S``):
    they count as 0 digits.
    """
    counts: dict[int, float] = {}
    if lens is not None:
        for ln, wt in zip(lens[ok].tolist(), w[ok].tolist()):
            if not np.isnan(ln):
                counts[int(ln)] = counts.get(int(ln), 0.0) + wt
    if bare is not None and lens is not None:
        zero = float(w[ok & bare & np.isnan(lens)].sum())
        if zero > 0:
            counts[0] = counts.get(0, 0.0) + zero
    return counts


def _frac_spec(spec: dict[str, Any], counts: dict[int, float], default: int) -> dict[str, Any]:
    """``frac_digits`` (the modal count) and, when several counts occur, ``frac_digits_pmf``.

    ``frac_digits_pmf`` ({digits: rows}, string keys) lets the generator give each
    row a digit count by exact quota, so minority precisions (and an exact extreme
    written with more digits than most values) are reproduced.
    """
    spec["frac_digits"] = int(_wmax_key(counts, default)) if counts else default
    kept = {d: c for d, c in counts.items() if c > 0}
    if len(kept) > 1:
        spec["frac_digits_pmf"] = {str(d): int(round(c)) for d, c in sorted(kept.items())}
    return spec


def _wmax_key(weights: dict[Any, float], default: Any) -> Any:
    if not weights:
        return default
    return max(weights.items(), key=lambda kv: (kv[1], str(kv[0])))[0]


def _case_style(token: str) -> str:
    if token.isupper():
        return "upper"
    if token.islower():
        return "lower"
    return "title"


def _apply_case(name: str, style: str) -> str:
    return name.upper() if style == "upper" else name.lower() if style == "lower" else name.title()


def _zf(x: np.ndarray, width: int) -> np.ndarray:
    s = np.asarray(x, dtype=np.int64).astype(STR)
    return np.strings.zfill(s, width) if width > 1 else s


def _offset_minutes(tokens: pa.Array) -> np.ndarray:
    """UTC offsets (minutes) of %z tokens; NaN when invalid."""
    out = np.full(len(tokens), np.nan)
    z = _np(pc.fill_null(pc.match_substring_regex(tokens, r"^[Zz]$"), False), bool)
    out[z] = 0.0
    g = _Groups(tokens, r"^(?P<sign>[+-])(?P<hh>[0-9]{2}):?(?P<mm>[0-9]{2})?$")
    hh = g.ints("hh", -1)
    mm = g.ints("mm", 0)
    sign = np.where(g.text("sign") == "-", -1, 1)
    good = g.ok & (hh >= 0) & (hh <= 14) & (mm < 60)
    out[good] = (sign * (hh * 60 + mm))[good]
    return out


@lru_cache(maxsize=256)
def _zone(name: str) -> zoneinfo.ZoneInfo | None:
    if "/" not in name and name.upper() not in ("UTC",):
        return None
    try:
        return zoneinfo.ZoneInfo(name)
    except (zoneinfo.ZoneInfoNotFoundError, ValueError, OSError):
        return None


_ZONE_ABBR_RE = re.compile(r"[A-Z]{3,5}")


def is_zone_token(token: str) -> bool:
    """A %Z token: a known abbreviation (any case), 3-5 capital letters, or an IANA zone name."""
    if token.upper() in TZ_ABBREVIATIONS:
        return True
    if _ZONE_ABBR_RE.fullmatch(token):
        return True
    return "/" in token and _zone(token) is not None


def zone_offsets_for_local(name: str, local_us: np.ndarray) -> np.ndarray:
    """UTC offset (minutes) of zone ``name`` for local wall times (NO_OFFSET if unknown)."""
    up = name.upper()
    if up in TZ_ABBREVIATIONS:
        return np.full(len(local_us), TZ_ABBREVIATIONS[up], dtype=np.int64)
    z = _zone(name)
    if z is None:
        return np.full(len(local_us), NO_OFFSET, dtype=np.int64)
    out = np.empty(len(local_us), dtype=np.int64)
    epoch = datetime(1970, 1, 1)
    for i, v in enumerate(local_us.tolist()):
        try:
            off = (epoch + timedelta(microseconds=v)).replace(tzinfo=z).utcoffset()
            out[i] = int(off.total_seconds() // 60) if off is not None else 0
        except (OverflowError, ValueError):
            out[i] = 0
    return out


def zone_offsets_for_instant(name: str, instant_us: np.ndarray) -> np.ndarray:
    """UTC offset (minutes) of zone ``name`` at UTC instants (NO_OFFSET if unknown)."""
    up = name.upper()
    if up in TZ_ABBREVIATIONS:
        return np.full(len(instant_us), TZ_ABBREVIATIONS[up], dtype=np.int64)
    z = _zone(name)
    if z is None:
        return np.full(len(instant_us), NO_OFFSET, dtype=np.int64)
    from datetime import timezone

    out = np.empty(len(instant_us), dtype=np.int64)
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    for i, v in enumerate(instant_us.tolist()):
        try:
            off = (epoch + timedelta(microseconds=v)).astimezone(z).utcoffset()
            out[i] = int(off.total_seconds() // 60) if off is not None else 0
        except (OverflowError, ValueError):
            out[i] = 0
    return out


@lru_cache(maxsize=256)
def rule_zone(name: Any) -> zoneinfo.ZoneInfo | None:
    """The IANA zone ``name`` names when its offset follows rules (``America/Chicago``, ``CST6CDT``), else None.

    A fixed offset (``+05:30``), ``UTC``/``Z`` or a fixed abbreviation (``CST``) is not a rule zone: its
    offset is the token itself.
    """
    if not isinstance(name, str) or not name.strip() or token_offset(name) is not None:
        return None
    try:
        return zoneinfo.ZoneInfo(name)
    except (zoneinfo.ZoneInfoNotFoundError, ValueError, OSError):
        return None


def zone_instant_offsets(name: str, instant_us: np.ndarray) -> np.ndarray | None:
    """UTC offsets (minutes) of IANA zone ``name`` at UTC instants (microseconds), vectorised.

    None when the zone is unknown or a value cannot be converted (callers then fall back to
    :func:`zone_offsets_for_instant`).
    """
    inst = np.asarray(instant_us, dtype=np.int64)
    if inst.size == 0:
        return np.zeros(0, dtype=np.int64)
    try:
        idx = pd.DatetimeIndex(inst.astype("datetime64[us]")).tz_localize("UTC").tz_convert(name)
        wall = np.asarray(idx.tz_localize(None).as_unit("us").asi8, dtype=np.int64)
    except Exception:  # noqa: BLE001 - unknown zone, out-of-range value
        return None
    return np.floor_divide(wall - inst, 60 * US)


def zone_local_offsets(name: str, local_us: np.ndarray) -> np.ndarray | None:
    """UTC offsets (minutes) of IANA zone ``name`` for local wall times (microseconds), vectorised.

    The semantics of Python's ``fold=0`` (:func:`zone_offsets_for_local`): a wall time a DST change repeats
    takes its first (daylight-time) offset; a wall time a DST change skips takes the offset before the
    change, so it is written one DST step later. None when the zone is unknown or a value cannot be converted.
    """
    loc = np.asarray(local_us, dtype=np.int64)
    if loc.size == 0:
        return np.zeros(0, dtype=np.int64)
    try:
        idx = pd.DatetimeIndex(loc.astype("datetime64[us]"))
        aware = idx.tz_localize(name, ambiguous=np.ones(loc.size, dtype=bool), nonexistent="NaT")
        utc = aware.tz_convert("UTC").tz_localize(None).as_unit("us")
        skipped = np.asarray(utc.isna(), dtype=bool)
        u = np.asarray(utc.asi8, dtype=np.int64)
    except Exception:  # noqa: BLE001 - unknown zone, out-of-range value
        return None
    off = np.floor_divide(loc - np.where(skipped, loc, u), 60 * US)
    if skipped.any():
        z = zoneinfo.ZoneInfo(name)
        epoch = datetime(1970, 1, 1)
        for i in np.flatnonzero(skipped).tolist():  # fold=0: the offset before the change
            o = (epoch + timedelta(microseconds=int(loc[i]))).replace(tzinfo=z).utcoffset()
            off[i] = int(o.total_seconds() // 60) if o is not None else 0
    return off


def token_offset(token: str) -> int | None:
    """Fixed UTC offset (minutes) of a %z token or zone abbreviation; None for zone names/unknown."""
    if token in ("Z", "z"):
        return 0
    m = re.fullmatch(r"([+-])([0-9]{2}):?([0-9]{2})?", token)
    if m:
        v = int(m.group(2)) * 60 + int(m.group(3) or 0)
        return -v if m.group(1) == "-" else v
    return TZ_ABBREVIATIONS.get(token.upper())


# --------------------------------------------------------------------------- formats
class DateFormat:
    """Base class: one explicit format (see module docstring)."""

    key: str
    kind: str  # 'date' | 'datetime' | 'year_month' | 'time' | 'duration'
    has_tz: bool = False
    has_time: bool = False
    is_integer: bool = False  # written as a plain integer (DATE_AS_INT forms)
    regex: str
    coarse: re.Pattern[str]

    def parse(self, values: Any) -> Parsed:  # pragma: no cover - abstract
        raise NotImplementedError

    def infer_render(self, parsed: Parsed, weights: np.ndarray) -> dict[str, Any]:
        return {}

    def render(self, comps: Components, spec: dict[str, Any]) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def resolution_us(self, spec: dict[str, Any], tpu: int = 1) -> int:
        """Step (microseconds; ``tpu`` ticks per microsecond) of the values this format writes under ``spec``.

        :meth:`render` truncates finer values to it (dates to the day, ``%H:%M``
        to the minute, ``%S`` to the second, ``%f`` to its digits), so the
        generator snaps values to this grid before rendering.
        """
        return US_DAY * tpu if self.kind in ("date", "year_month") else 1

    def matches_shape(self, skeleton: str) -> bool:
        return self.coarse.fullmatch(skeleton) is not None


def _frac_resolution(digits: int, tpu: int = 1) -> int:
    """Step (us, or ``tpu`` ticks per us) of a seconds value written with ``digits`` fraction digits."""
    if digits <= 0:
        return US * tpu
    return max(10 ** (9 - min(digits, 9)) * tpu // NS_PER_US, 1)


_DIR_RE = re.compile(r"%([-#]?)(.)|([^%]+)", re.S)
NUMERIC_CODES = frozenset("YymdHIMSf")
FLEX_CODES = frozenset("mdHIMS")
#: Always exactly two digits when parsing (minutes and seconds: ``h:mm[:ss]``).
TWO_DIGIT_CODES = frozenset("MS")
SUPPORTED_CODES = frozenset("YymdbBaAHIMSfpzZ")


class StrfFormat(DateFormat):
    """A strftime-like format string parsed with an explicit regex (see module docstring)."""

    def __init__(self, fmt: str) -> None:
        if fmt.endswith("%") and not fmt.endswith("%%"):
            raise ValueError(f"format {fmt!r} ends with a lone '%'")
        tokens: list[tuple[str, str]] = []
        for m in _DIR_RE.finditer(fmt):
            flag, code, lit = m.group(1), m.group(2), m.group(3)
            if lit is not None:
                tokens.append(("l", lit))
            elif code == "%":
                tokens.append(("l", "%"))
            elif code not in SUPPORTED_CODES:
                raise ValueError(f"unsupported directive %{flag}{code} in {fmt!r}")
            else:
                tokens.append(("d", code))
        merged: list[tuple[str, str]] = []
        for t in tokens:
            if merged and t[0] == "l" and merged[-1][0] == "l":
                merged[-1] = ("l", merged[-1][1] + t[1])
            else:
                merged.append(t)
        self.tokens = merged
        codes = [c for k, c in merged if k == "d"]
        if len(codes) != len(set(codes)):
            raise ValueError(f"format {fmt!r} repeats a directive")
        cs = set(codes)
        self.codes = cs
        if len(cs & {"Y", "y"}) > 1 or len(cs & {"m", "b", "B"}) > 1 or len(cs & {"H", "I"}) > 1 \
                or len(cs & {"a", "A"}) > 1 or len(cs & {"z", "Z"}) > 1:
            raise ValueError(f"format {fmt!r} has conflicting directives")
        if ("I" in cs) != ("p" in cs):
            raise ValueError(f"format {fmt!r}: %I needs %p (and %p needs %I)")
        has_hour = bool(cs & {"H", "I"})
        if ("M" in cs) != has_hour or ("S" in cs and "M" not in cs) or ("f" in cs and "S" not in cs):
            raise ValueError(f"format {fmt!r}: incomplete time (use %H:%M[:%S[.%f]])")
        has_year, has_month, has_day = bool(cs & {"Y", "y"}), bool(cs & {"m", "b", "B"}), "d" in cs
        date_part = has_year and has_month and has_day
        if has_hour and date_part:
            kind = "datetime"
        elif has_hour and not (has_year or has_month or has_day):
            kind = "time"
        elif date_part:
            kind = "date"
        elif has_year and has_month and not has_day:
            kind = "year_month"
        else:
            raise ValueError(f"format {fmt!r} is not a date, datetime, year-month or time format")
        if cs & {"z", "Z"} and kind != "datetime":
            raise ValueError(f"format {fmt!r}: offsets/zones need a date and a time")
        if cs & {"a", "A"} and not date_part:
            raise ValueError(f"format {fmt!r}: weekday names need a full date")
        self.kind = kind
        self.has_tz = bool(cs & {"z", "Z"})
        self.has_time = has_hour
        self.key = "".join(("%%" if v == "%" else v.replace("%", "%%")) if k == "l" else "%" + v for k, v in merged)
        self.compact: set[str] = set()
        pieces: list[str] = []
        coarse: list[str] = []
        for i, (k, v) in enumerate(merged):
            if k == "l":
                pieces.append(_escape(v))
                coarse.append(_escape(coarse_skeleton(v)))
                continue
            prev_num = i > 0 and merged[i - 1][0] == "d" and merged[i - 1][1] in NUMERIC_CODES
            next_num = i + 1 < len(merged) and merged[i + 1][0] == "d" and merged[i + 1][1] in NUMERIC_CODES
            if v in TWO_DIGIT_CODES or (v in FLEX_CODES and (prev_num or next_num)):
                self.compact.add(v)
            p, c = self._piece(v, v in self.compact)
            pieces.append(p)
            coarse.append(c)
        self.regex = "^" + "".join(pieces) + "$"
        self.coarse = re.compile("".join(coarse))
        self.is_integer = all(k == "d" and v in NUMERIC_CODES for k, v in merged)

    @staticmethod
    def _piece(code: str, compact: bool) -> tuple[str, str]:
        if code == "Y":
            return r"(?P<Y>[0-9]{4})", r"9{4}"
        if code == "y":
            return r"(?P<y>[0-9]{2})", r"9{2}"
        if code in FLEX_CODES:
            q = "{2}" if compact else "{1,2}"
            return rf"(?P<{code}>[0-9]{q})", rf"9{q}"
        if code == "f":
            return r"(?P<f>[0-9]{1,9})", r"9{1,9}"
        if code == "b":
            return r"(?P<b>[A-Za-z]{3})", r"a{3}"
        if code == "B":
            return r"(?P<B>[A-Za-z]{3,9})", r"a{3,9}"
        if code == "a":
            return r"(?P<a>[A-Za-z]{3})", r"a{3}"
        if code == "A":
            return r"(?P<A>[A-Za-z]{6,9})", r"a{6,9}"
        if code == "p":
            return r"(?P<p>[AaPp]\.?[Mm]\.?)", r"a\.?a\.?"
        if code == "z":
            return r"(?P<z>[Zz]|[+-][0-9]{2}(?::?[0-9]{2})?)", r"(?:a|[+-]9{2}(?::?9{2})?)"
        if code == "Z":
            return r"(?P<Z>[A-Za-z][A-Za-z0-9_+/-]{0,39})", r"a[a9_+/-]{0,39}"
        raise ValueError(code)  # pragma: no cover

    # ------------------------------------------------------------------ parse
    def parse(self, values: Any) -> Parsed:
        n = len(values)
        p = Parsed.blank(n)
        if n == 0:
            return p
        g = _Groups(values, self.regex)
        ok = g.ok.copy()
        cs = self.codes
        if "Y" in cs:
            year = g.ints("Y")
            ok &= year >= 1
        elif "y" in cs:
            yy = g.ints("y")
            year = np.where(yy >= PIVOT, 1900 + yy, 2000 + yy)
            p.yy[:] = True
        else:
            year = np.full(n, 1970, dtype=np.int64)
        if "m" in cs:
            month = g.ints("m")
            p.obs["m_len"] = g.lens("m")
        elif "b" in cs:
            month = g.index_in("b", MONTHS_ABBR)
            p.obs["b"] = g.obj("b")
        elif "B" in cs:
            month = g.index_in("B", MONTHS_FULL)
            p.obs["B"] = g.obj("B")
        else:
            month = np.ones(n, dtype=np.int64)
        ok &= (month >= 1) & (month <= 12)
        if "d" in cs:
            day = g.ints("d")
            p.obs["d_len"] = g.lens("d")
            ok &= (day >= 1) & (day <= days_in_month(year, month))
        else:
            day = np.ones(n, dtype=np.int64)
        for code, names in (("a", WEEKDAYS_ABBR), ("A", WEEKDAYS_FULL)):
            if code in cs:
                ok &= g.index_in(code, names) > 0
                p.obs[code] = g.obj(code)
        hour = np.zeros(n, dtype=np.int64)
        if "H" in cs:
            hour = g.ints("H")
            p.obs["H_len"] = g.lens("H")
            ok &= hour <= 23
        elif "I" in cs:
            h12 = g.ints("I")
            p.obs["I_len"] = g.lens("I")
            p.obs["I_val"] = h12
            ok &= (h12 >= 1) & (h12 <= 12)
            pm = _np(pc.fill_null(pc.starts_with(pc.utf8_lower(g.given("p")), "p"), False), bool)
            hour = h12 % 12 + 12 * pm
            p.obs["p"] = g.obj("p")
        minute = second = micro = np.zeros(n, dtype=np.int64)
        if "M" in cs:
            minute = g.ints("M")
            p.obs["M_len"] = g.lens("M")
            ok &= minute <= 59
        if "S" in cs:
            second = g.ints("S")
            p.obs["S_len"] = g.lens("S")
            ok &= second <= 59
        if "f" in cs:
            micro = g.micro("f")
            p.nano = g.nano("f")
            p.obs["f_len"] = g.lens("f")
        p.year, p.month, p.day, p.hour, p.minute, p.second, p.micro = year, month, day, hour, minute, second, micro
        if "z" in cs:
            ztok = g.given("z")
            off = _offset_minutes(ztok)
            ok &= ~np.isnan(off)
            p.off_min = np.where(np.isnan(off), NO_OFFSET, off).astype(np.int64)
            p.tz_tok = g.text("z")
        elif "Z" in cs:
            toks = g.text("Z")
            p.tz_tok = toks
            uniq = pd.unique(toks)
            valid = {t: is_zone_token(str(t)) for t in uniq}
            if not all(valid.values()):
                ok &= np.fromiter((valid[t] for t in toks.tolist()), dtype=bool, count=n)
            off = np.full(n, NO_OFFSET, dtype=np.int64)
            good = np.flatnonzero(ok)
            if good.size:
                local = days_from_civil(year[good], month[good], day[good]) * US_DAY + (
                    ((hour[good] * 60 + minute[good]) * 60 + second[good]) * US + micro[good])
                gt = toks[good]
                for name in pd.unique(gt):
                    sel = gt == name
                    off[good[sel]] = zone_offsets_for_local(str(name), local[sel])
            p.off_min = off
        p.ok = ok
        return p

    # ------------------------------------------------------------------ render spec
    def infer_render(self, parsed: Parsed, weights: np.ndarray) -> dict[str, Any]:
        w = np.asarray(weights, dtype=float)
        ok = parsed.ok
        spec: dict[str, Any] = {}
        pad: dict[str, bool] = {}
        values = {"m": parsed.month, "d": parsed.day, "H": parsed.hour, "M": parsed.minute, "S": parsed.second,
                  "I": parsed.obs.get("I_val")}
        for code in sorted(self.codes & FLEX_CODES):
            if code in self.compact:
                pad[code] = True
                continue
            lens = parsed.obs.get(code + "_len")
            vals = values[code]
            if lens is None or vals is None:
                pad[code] = True
                continue
            small = ok & (vals < 10)
            w2 = float(w[small & (lens == 2)].sum())
            w1 = float(w[small & (lens == 1)].sum())
            pad[code] = bool(w2 >= w1) if (w1 + w2) > 0 else True
        if pad:
            spec["pad"] = pad
        case: dict[str, str] = {}
        for code in ("b", "B", "a", "A"):
            toks = parsed.obs.get(code)
            if toks is None:
                continue
            styles: dict[str, float] = {}
            for t, wt in zip(toks[ok].tolist(), w[ok].tolist()):
                if t:
                    st = _case_style(t)
                    styles[st] = styles.get(st, 0.0) + wt
            case[code] = _wmax_key(styles, "title")
        if case:
            spec["case"] = case
        if "p" in self.codes:
            spec["ampm"] = _ampm_tokens(parsed.obs.get("p"), ok, w)
        if "f" in self.codes:
            _frac_spec(spec, _digit_counts(parsed.obs.get("f_len"), ok, w), 6)
        return spec

    def resolution_us(self, spec: dict[str, Any], tpu: int = 1) -> int:
        if not self.has_time:
            return US_DAY * tpu
        if "f" in self.codes:
            return _frac_resolution(int(spec.get("frac_digits", 6)), tpu)
        return (US if "S" in self.codes else 60 * US) * tpu

    # ------------------------------------------------------------------ render
    def render(self, comps: Components, spec: dict[str, Any]) -> np.ndarray:
        n = comps.n
        out = np.full(n, "", dtype=STR)
        pad = spec.get("pad", {})
        case = spec.get("case", {})
        for k, v in self.tokens:
            if k == "l":
                out = np.strings.add(out, v)
                continue
            out = np.strings.add(out, self._render_code(v, comps, pad, case, spec))
        return out

    def _render_code(self, code: str, c: Components, pad: dict[str, bool], case: dict[str, str],
                     spec: dict[str, Any]) -> np.ndarray:
        def flex(x: np.ndarray, key: str) -> np.ndarray:
            return _zf(x, 2 if pad.get(key, True) or key in self.compact else 1)

        if code == "Y":
            return _zf(c.year, 4)
        if code == "y":
            return _zf(c.year % 100, 2)
        if code == "m":
            return flex(c.month, "m")
        if code == "d":
            return flex(c.day, "d")
        if code in ("b", "B"):
            names = MONTHS_ABBR if code == "b" else MONTHS_FULL
            arr = np.array([_apply_case(x, case.get(code, "title")) for x in names], dtype=STR)
            return arr[np.clip(c.month, 1, 12) - 1]
        if code in ("a", "A"):
            names = WEEKDAYS_ABBR if code == "a" else WEEKDAYS_FULL
            arr = np.array([_apply_case(x, case.get(code, "title")) for x in names], dtype=STR)
            return arr[c.weekday]
        if code == "H":
            return flex(c.hour, "H")
        if code == "I":
            h12 = c.hour % 12
            h12 = np.where(h12 == 0, 12, h12)
            return flex(h12, "I")
        if code == "M":
            return flex(c.minute, "M")
        if code == "S":
            return flex(c.second, "S")
        if code == "f":
            k = int(spec.get("frac_digits", 6))
            return _zf(frac_value(c.micro, c.nano, k), k)
        if code == "p":
            am_pm = spec.get("ampm") or {"am": "AM", "pm": "PM"}
            arr = np.array([am_pm["am"], am_pm["pm"]], dtype=STR)
            return arr[(c.hour >= 12).astype(np.int64)]
        if code in ("z", "Z"):
            return np.asarray(c.tz, dtype=STR)
        raise ValueError(code)  # pragma: no cover


def _ampm_tokens(tokens: np.ndarray | None, ok: np.ndarray, w: np.ndarray) -> dict[str, str]:
    am: dict[str, float] = {}
    pm: dict[str, float] = {}
    if tokens is not None:
        for t, wt in zip(tokens[ok].tolist(), w[ok].tolist()):
            if not t:
                continue
            (pm if t[0] in "Pp" else am)[t] = (pm if t[0] in "Pp" else am).get(t, 0.0) + wt
    a = _wmax_key(am, None)
    p = _wmax_key(pm, None)

    def like(ref: str, letter: str) -> str:
        out = letter + ref[1:]
        return out.lower() if ref.islower() else out.upper() if ref[0].isupper() else out

    if a is None and p is None:
        a, p = "AM", "PM"
    elif a is None:
        a = like(p, "A" if p[0].isupper() else "a")
    elif p is None:
        p = like(a, "P" if a[0].isupper() else "p")
    return {"am": a, "pm": p}


# --------------------------------------------------------------------------- special formats
class _IntegerFormat(DateFormat):
    is_integer = True

    def __init__(self, key: str, kind: str, digits: str, lo: float, hi: float) -> None:
        self.key = key
        self.kind = kind
        self.has_time = kind == "datetime"
        self.lo, self.hi = lo, hi
        self.regex = rf"^(?P<n>[0-9]{{{digits}}})$"
        self.coarse = re.compile(rf"9{{{digits}}}")

    def _numbers(self, values: Any) -> tuple[np.ndarray, np.ndarray]:
        g = _Groups(values, self.regex)
        x = g.ints("n", -1)
        ok = g.ok & (x >= self.lo) & (x <= self.hi)
        return x, ok


class ExcelSerialFormat(_IntegerFormat):
    """Excel day serials (1900 date system, origin 1899-12-30): 10000-73050."""

    ORIGIN = -25569  # day ordinal of 1899-12-30

    def __init__(self) -> None:
        super().__init__("@excel_serial", "date", "5", 10000, 73050)

    def parse(self, values: Any) -> Parsed:
        p = Parsed.blank(len(values))
        if not len(values):
            return p
        x, ok = self._numbers(values)
        day = np.where(ok, x, 25569) + self.ORIGIN
        p.year, p.month, p.day = civil_from_days(day)
        p.ok = ok
        return p

    def render(self, comps: Components, spec: dict[str, Any]) -> np.ndarray:
        return _zf(comps.day_ord - self.ORIGIN, 1)


class EpochFormat(_IntegerFormat):
    """Unix epoch seconds (``@epoch_s``) or milliseconds (``@epoch_ms``), UTC."""

    def __init__(self, unit: str) -> None:
        self.scale = US if unit == "s" else 1000
        if unit == "s":
            super().__init__("@epoch_s", "datetime", "9,10", 9e8, 2.2e9)
        else:
            super().__init__("@epoch_ms", "datetime", "12,13", 9e11, 2.2e12)

    def parse(self, values: Any) -> Parsed:
        p = Parsed.blank(len(values))
        if not len(values):
            return p
        x, ok = self._numbers(values)
        inst = np.where(ok, x, 0) * self.scale
        day, sod = np.divmod(inst, US_DAY)
        p.year, p.month, p.day = civil_from_days(day)
        secs, p.micro = np.divmod(sod, US)
        p.hour, rem = np.divmod(secs, 3600)
        p.minute, p.second = np.divmod(rem, 60)
        p.ok = ok
        return p

    def resolution_us(self, spec: dict[str, Any], tpu: int = 1) -> int:
        return int(self.scale) * tpu

    def render(self, comps: Components, spec: dict[str, Any]) -> np.ndarray:
        inst = comps.instant_us if comps.instant_us is not None else (
            comps.day_ord * US_DAY + ((comps.hour * 60 + comps.minute) * 60 + comps.second) * US + comps.micro)
        return _zf(np.floor_divide(inst, self.scale), 1)


class _DurationFormat(DateFormat):
    kind = "duration"
    has_time = True

    def resolution_us(self, spec: dict[str, Any], tpu: int = 1) -> int:
        """Durations are written truncated toward zero to this step."""
        return _frac_resolution(int(spec.get("frac_digits", 0)), tpu)

    def _finish(self, n: int, ok: np.ndarray, dur: np.ndarray, neg: np.ndarray | None = None,
                nano: np.ndarray | None = None) -> Parsed:
        p = Parsed.blank(n)
        p.ok = ok
        p.dur_us = np.where(neg, -dur, dur) if neg is not None else dur
        if nano is not None:
            p.nano = np.where(neg, -nano, nano) if neg is not None else nano
        return p


class DurationClockFormat(_DurationFormat):
    """``[-]H:MM[:SS[.f]]`` with unbounded hours (pandas ``to_timedelta`` rejects ``hh:mm``).

    ``@dur_clock`` reads every layout (``--date-format``); the value
    candidates are its exact layouts ``@dur_hm`` (``H:MM``), ``@dur_hms``
    (``H:MM:SS``) and ``@dur_hmsf`` (``H:MM:SS.f``), the counterparts of the
    TIME formats ``%H:%M``, ``%H:%M:%S`` and ``%H:%M:%S.%f``, so a column that
    mixes layouts is a greedy cover with per-layout shares.
    """

    _LAYOUTS = {
        "any": (r"(?::(?P<S>[0-9]{2})(?:\.(?P<f>[0-9]{1,9}))?)?", r"(?::9{2}(?:\.9{1,9})?)?"),
        "hm": (r"(?P<S>)(?P<f>)", r""),
        "hms": (r":(?P<S>[0-9]{2})(?P<f>)", r":9{2}"),
        "hmsf": (r":(?P<S>[0-9]{2})\.(?P<f>[0-9]{1,9})", r":9{2}\.9{1,9}"),
    }

    def __init__(self, layout: str = "any") -> None:
        tail, coarse_tail = self._LAYOUTS[layout]
        self.layout = layout
        self.key = "@dur_clock" if layout == "any" else "@dur_" + layout
        self.regex = r"^(?P<neg>-?)(?P<h>[0-9]{1,7}):(?P<M>[0-9]{2})" + tail + "$"
        self.coarse = re.compile(r"-?9{1,7}:9{2}" + coarse_tail)

    def parse(self, values: Any) -> Parsed:
        n = len(values)
        if not n:
            return Parsed.blank(0)
        g = _Groups(values, self.regex)
        ok = g.ok.copy()
        h = g.ints("h")
        mi = g.ints("M")
        se = g.ints("S")
        micro = g.micro("f")
        ok &= (mi <= 59) & (se <= 59)
        neg = g.text("neg") == "-"
        dur = ((h * 60 + mi) * 60 + se) * US + micro
        p = self._finish(n, ok, dur, neg, g.nano("f"))
        p.obs["has_s"] = g.present("S")
        p.obs["h_len"] = g.lens("h")
        p.obs["h_val"] = h
        p.obs["f_len"] = g.lens("f")
        return p

    def infer_render(self, parsed: Parsed, weights: np.ndarray) -> dict[str, Any]:
        w = np.asarray(weights, dtype=float)
        ok = parsed.ok
        has_s = parsed.obs.get("has_s", np.zeros(parsed.n, dtype=bool))
        ws = float(w[ok & has_s].sum())
        wn = float(w[ok & ~has_s].sum())
        h = parsed.obs.get("h_val", np.zeros(parsed.n, dtype=np.int64))
        lens = parsed.obs.get("h_len", np.ones(parsed.n))
        small = ok & (h < 10)
        w2 = float(w[small & (lens >= 2)].sum())
        w1 = float(w[small & (lens == 1)].sum())
        seconds = {"hm": False, "hms": True, "hmsf": True}.get(self.layout, ws >= wn)
        spec: dict[str, Any] = {"seconds": seconds, "hour_width": 2 if (w2 >= w1 and w1 + w2 > 0) else 1}
        return _frac_spec(spec, _digit_counts(parsed.obs.get("f_len"), ok, w, has_s if self.layout == "any" else None),
                          0)

    def resolution_us(self, spec: dict[str, Any], tpu: int = 1) -> int:
        if not spec.get("seconds", True):
            return 60 * US * tpu
        return _frac_resolution(int(spec.get("frac_digits", 0)), tpu)

    def render(self, comps: Components, spec: dict[str, Any]) -> np.ndarray:
        neg, a, nano = _dur_parts(comps)
        secs, micro = np.divmod(a, US)
        h, rem = np.divmod(secs, 3600)
        mi, se = np.divmod(rem, 60)
        out = np.strings.add(np.where(neg, "-", "").astype(STR), _zf(h, int(spec.get("hour_width", 1))))
        out = np.strings.add(np.strings.add(out, ":"), _zf(mi, 2))
        if spec.get("seconds", True):
            out = np.strings.add(np.strings.add(out, ":"), _zf(se, 2))
            k = int(spec.get("frac_digits", 0))
            if k:
                out = np.strings.add(np.strings.add(out, "."), _zf(frac_value(micro, nano, k), k))
        return out


class DurationIsoFormat(_DurationFormat):
    """ISO 8601 durations ``[-]P[nD][T[nH][nM][n[.f]S]]`` (no years, months or weeks)."""

    key = "@dur_iso"
    regex = (r"^(?P<neg>-?)P(?:(?P<D>[0-9]+)D)?(?:(?P<T>T)(?:(?P<H>[0-9]+)H)?(?:(?P<M>[0-9]+)M)?"
             r"(?:(?P<S>[0-9]+)(?:\.(?P<f>[0-9]{1,9}))?S)?)?$")
    coarse = re.compile(r"-?a(?:9+a)?(?:a(?:9+a)?(?:9+a)?(?:9+(?:\.9{1,9})?a)?)?")

    def parse(self, values: Any) -> Parsed:
        n = len(values)
        if not n:
            return Parsed.blank(0)
        g = _Groups(values, self.regex)
        ok = g.ok.copy()
        parts = {c: g.ints(c) for c in ("D", "H", "M", "S")}
        present = {c: g.present(c) for c in ("D", "H", "M", "S")}
        has_t = g.present("T")
        any_time = present["H"] | present["M"] | present["S"]
        ok &= (present["D"] | any_time) & (~has_t | any_time)
        micro = g.micro("f")
        dur = (((parts["D"] * 24 + parts["H"]) * 60 + parts["M"]) * 60 + parts["S"]) * US + micro
        neg = g.text("neg") == "-"
        p = self._finish(n, ok, dur, neg, g.nano("f"))
        for c in ("D", "H", "M", "S"):
            p.obs["has_" + c] = present[c]
        p.obs["f_len"] = g.lens("f")
        return p

    def infer_render(self, parsed: Parsed, weights: np.ndarray) -> dict[str, Any]:
        w = np.asarray(weights, dtype=float)
        ok = parsed.ok
        total = float(w[ok].sum()) or 1.0
        share = {c: float(w[ok & parsed.obs.get("has_" + c, np.zeros(parsed.n, dtype=bool))].sum()) / total
                 for c in ("D", "H", "M", "S")}
        spec: dict[str, Any] = {"days": share["D"] > 0,
                                "always": [c for c in ("H", "M", "S") if share[c] >= 0.999]}
        return _frac_spec(spec, _digit_counts(parsed.obs.get("f_len"), ok, w, parsed.obs.get("has_S")), 0)

    def render(self, comps: Components, spec: dict[str, Any]) -> np.ndarray:
        neg_a, abs_a, nano_a = _dur_parts(comps)
        use_days = bool(spec.get("days"))
        always = set(spec.get("always", []))
        k = int(spec.get("frac_digits", 0))
        out = []
        nanos = [0] * len(abs_a) if nano_a is None else nano_a.tolist()
        for neg, v, nano in zip(neg_a.tolist(), abs_a.tolist(), nanos):
            sign = "-" if neg else ""
            secs, micro = divmod(v, US)
            d, rem = divmod(secs, 86400) if use_days else (0, secs)
            h, rem = divmod(rem, 3600)
            mi, se = divmod(rem, 60)
            text = sign + "P" + (f"{d}D" if d else "")
            t = ""
            if h or "H" in always:
                t += f"{h}H"
            if mi or "M" in always:
                t += f"{mi}M"
            if se or micro or nano or "S" in always or (not t and not d):
                frac = ""
                if k:
                    frac = "." + str(frac_value(micro, nano, k)).zfill(k)
                t += f"{se}{frac}S"
            out.append(text + ("T" + t if t else ""))
        return np.array(out, dtype=STR) if out else np.zeros(0, dtype=STR)


class DurationDaysFormat(_DurationFormat):
    """``N days HH:MM:SS[.f]`` (pandas ``str(Timedelta)``) or ``N day[s], H:MM:SS`` (Python ``timedelta``)."""

    key = "@dur_days"
    regex = (r"^(?P<D>[0-9]+) (?P<unit>days?)(?P<comma>,?) (?P<h>[0-9]{1,2}):(?P<M>[0-9]{2}):(?P<S>[0-9]{2})"
             r"(?:\.(?P<f>[0-9]{1,9}))?$")
    coarse = re.compile(r"9+ a{3,4},? 9{1,2}:9{2}:9{2}(?:\.9{1,9})?")

    def parse(self, values: Any) -> Parsed:
        n = len(values)
        if not n:
            return Parsed.blank(0)
        g = _Groups(values, self.regex)
        ok = g.ok.copy()
        d, h, mi, se = (g.ints(c) for c in ("D", "h", "M", "S"))
        ok &= (h <= 23) & (mi <= 59) & (se <= 59)
        dur = (((d * 24 + h) * 60 + mi) * 60 + se) * US + g.micro("f")
        p = self._finish(n, ok, dur, None, g.nano("f"))
        p.obs["comma"] = g.text("comma") == ","
        p.obs["h_len"] = g.lens("h")
        p.obs["f_len"] = g.lens("f")
        return p

    def infer_render(self, parsed: Parsed, weights: np.ndarray) -> dict[str, Any]:
        w = np.asarray(weights, dtype=float)
        ok = parsed.ok
        comma = parsed.obs.get("comma", np.zeros(parsed.n, dtype=bool))
        python = float(w[ok & comma].sum()) > float(w[ok & ~comma].sum())
        spec: dict[str, Any] = {"style": "python" if python else "pandas"}
        return _frac_spec(spec, _digit_counts(parsed.obs.get("f_len"), ok, w, np.ones(parsed.n, dtype=bool)), 0)

    def render(self, comps: Components, spec: dict[str, Any]) -> np.ndarray:
        python = spec.get("style") == "python"
        k = int(spec.get("frac_digits", 0))
        out = []
        dur = np.asarray(comps.dur_us, dtype=np.int64)
        nanos = [0] * len(dur) if comps.nano is None else np.asarray(comps.nano, dtype=np.int64).tolist()
        for v, nano in zip(dur.tolist(), nanos):
            if v < 0 or nano < 0:
                v, nano = 0, 0
            secs, micro = divmod(v, US)
            d, rem = divmod(secs, 86400)
            h, rem = divmod(rem, 3600)
            mi, se = divmod(rem, 60)
            frac = ""
            if k:
                frac = "." + str(frac_value(micro, nano, k)).zfill(k)
            if python:
                out.append(f"{d} day{'s' if d != 1 else ''}, {h}:{mi:02d}:{se:02d}{frac}")
            else:
                out.append(f"{d} days {h:02d}:{mi:02d}:{se:02d}{frac}")
        return np.array(out, dtype=STR) if out else np.zeros(0, dtype=STR)


_SPECIALS = {
    "@excel_serial": ExcelSerialFormat,
    "@epoch_s": lambda: EpochFormat("s"),
    "@epoch_ms": lambda: EpochFormat("ms"),
    "@dur_clock": DurationClockFormat,
    "@dur_hm": lambda: DurationClockFormat("hm"),
    "@dur_hms": lambda: DurationClockFormat("hms"),
    "@dur_hmsf": lambda: DurationClockFormat("hmsf"),
    "@dur_iso": DurationIsoFormat,
    "@dur_days": DurationDaysFormat,
}


def normalize_format(fmt: str) -> str:
    """``%-m``/``%#m`` -> ``%m`` (flexible width is the default)."""
    return re.sub(r"%[-#]([A-Za-z])", r"%\1", fmt)


@lru_cache(maxsize=None)
def get_format(key: str) -> DateFormat:
    """The format object for a key (raises ValueError for an invalid key)."""
    if key.startswith("@"):
        if key not in _SPECIALS:
            raise ValueError(f"unknown special format {key!r}")
        return _SPECIALS[key]()
    return StrfFormat(normalize_format(key))


# --------------------------------------------------------------------------- candidates
DATE_FORMATS: tuple[str, ...] = (
    "%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d",
    "%m/%d/%Y", "%d/%m/%Y", "%m-%d-%Y", "%d-%m-%Y", "%m.%d.%Y", "%d.%m.%Y",
    "%m/%d/%y", "%d/%m/%y", "%m-%d-%y", "%d-%m-%y", "%m.%d.%y", "%d.%m.%y", "%y-%m-%d", "%y/%m/%d",
    "%d-%b-%Y", "%d-%b-%y", "%d%b%Y", "%d%b%y", "%d %b %Y", "%d %b %y", "%d %B %Y", "%d-%B-%Y", "%d/%b/%Y",
    "%b %d, %Y", "%B %d, %Y", "%b %d %Y", "%B %d %Y", "%b-%d-%Y", "%b. %d, %Y", "%Y-%b-%d",
    "%a, %d %b %Y", "%a %b %d %Y", "%A, %B %d, %Y", "%a, %b %d, %Y", "%Y%m%d",
)
YEAR_MONTH_FORMATS: tuple[str, ...] = (
    "%Y-%m", "%Y/%m", "%m/%Y", "%m-%Y", "%b %Y", "%B %Y", "%b-%y", "%b-%Y", "%B-%Y", "%b%Y", "%b %y",
)
TIME_FORMATS: tuple[str, ...] = (
    "%H:%M", "%H:%M:%S", "%H:%M:%S.%f", "%I:%M %p", "%I:%M:%S %p", "%I:%M%p", "%I:%M:%S%p", "%I:%M:%S.%f %p",
)
_DT_DATES: tuple[str, ...] = (
    "%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%d/%m/%Y", "%m-%d-%Y", "%d-%m-%Y", "%m/%d/%y", "%d/%m/%y",
    "%d.%m.%Y", "%d-%b-%Y", "%d-%b-%y", "%b %d, %Y", "%d %b %Y", "%Y%m%d",
)
_DT_TIMES: tuple[str, ...] = (
    "%H:%M:%S", "%H:%M", "%H:%M:%S.%f", "%I:%M %p", "%I:%M:%S %p", "%I:%M%p", "%I:%M:%S%p",
)
_DT_SPECIAL: tuple[str, ...] = (
    "%d%b%Y:%H:%M:%S", "%d%b%Y:%H:%M:%S.%f", "%d%b%Y:%H:%M",  # SAS DATETIME
    "%d-%b-%y %I.%M.%S.%f %p", "%d-%b-%y %I.%M.%S %p", "%d-%b-%Y %H.%M.%S",  # Oracle TIMESTAMP
    "%a %b %d %H:%M:%S %Y", "%a %b %d %H:%M:%S %Z %Y",  # ctime / Unix date
    "%a, %d %b %Y %H:%M:%S %z", "%a, %d %b %Y %H:%M:%S %Z",  # RFC 2822
    "%Y-%m-%d-%H.%M.%S.%f",  # DB2
    "%Y%m%dT%H%M%S", "%Y%m%dT%H%M%SZ",
)
DURATION_FORMATS: tuple[str, ...] = ("@dur_iso", "@dur_days", "@dur_hm", "@dur_hms", "@dur_hmsf")
#: Clock durations, each the counterpart of a TIME format (section 3 #23/#24: hours >= 24 make a duration).
CLOCK_DURATIONS: dict[str, str] = {"%H:%M": "@dur_hm", "%H:%M:%S": "@dur_hms", "%H:%M:%S.%f": "@dur_hmsf"}
INTEGER_FORMATS: tuple[str, ...] = ("%Y%m%d", "@excel_serial", "@epoch_s", "@epoch_ms")
#: Extra candidates for native date columns whose readers render a midnight time.
NATIVE_DATE_EXTRA: tuple[str, ...] = ("%Y-%m-%d 00:00:00", "%Y-%m-%dT00:00:00", "%Y-%m-%d 00:00:00.000000")


def _datetime_candidates() -> list[str]:
    out: list[str] = []
    for d in _DT_DATES:
        seps = (" ", "T") if d.startswith("%Y-") else (" ",)
        for t in _DT_TIMES:
            for sep in seps:
                if sep == "T" and "%p" in t:
                    continue
                base = d + sep + t
                out.append(base)
                if "%p" not in t:
                    out += [base + "%z", base + " %z"]
                out.append(base + " %Z")
    return out + list(_DT_SPECIAL)


def _build_candidates() -> tuple[str, ...]:
    seen: dict[str, None] = {}
    for k in (*DATE_FORMATS, *YEAR_MONTH_FORMATS, *_datetime_candidates(), *TIME_FORMATS, *DURATION_FORMATS):
        seen.setdefault(k, None)
    return tuple(seen)


#: Every value-detection candidate, in preference order (ties go to the earlier one).
CANDIDATES: tuple[str, ...] = _build_candidates()


def swapped_md(key: str) -> str:
    """The format with %m and %d exchanged (m/d vs d/m ambiguity)."""
    return key.replace("%m", "\0").replace("%d", "%m").replace("\0", "%d")


def day_first(key: str) -> bool:
    return "%d" in key and "%m" in key and key.index("%d") < key.index("%m")


def kind_group(kind: str) -> str:
    """Kinds that may share one column (greedy cover): date and datetime mix."""
    return "date" if kind in ("date", "datetime") else kind


__all__ = [
    "CANDIDATES", "CLOCK_DURATIONS", "Components", "DateFormat", "NO_OFFSET", "NS_PER_US", "Parsed", "STR", "US",
    "US_DAY", "civil_from_days", "coarse_skeleton", "days_from_civil", "frac_value", "get_format", "iso_day",
    "iso_instant", "iso_month",
    "hms", "normalize_format", "ordinal", "rule_zone", "token_offset", "weekday", "zone_instant_offsets",
    "zone_local_offsets", "zone_offsets_for_instant", "zone_offsets_for_local",
]
