"""Date, datetime, time and duration generation and rendering (AREA A).

AREA A owns this module (design 3 #12, #19-24, section 4, section 10 "Dates",
D14). :func:`generate_dates` never sees a real row: it reads the column's
detection params and its private model (raw value counts), through
``detect.dates.view``.

Generation (per column, ``n`` non-missing slots)
------------------------------------------------
1. Tokens: sentinels (``1900-01-01``, ``9999-12-31``, ``0/0/0``, far-off
   values held by >= 1% of rows) and short unparsed tokens get an exact
   largest-remainder quota against one "date" bucket, at least one row each
   (``rare_category_adjusted`` when that floor changed a count). They are
   written verbatim (raw text, whitespace included). Long unparsed text is never
   copied: its rows are in the date bucket.
2. Dates (DATE, DATE_MIXED_FORMATS, DATE_AS_INT, the date part of DATETIME):
   *calendar-like* columns (detection: >= ``--min-cell`` rows per distinct
   date, or ``--calendar``) draw only observed dates by exact quota; every
   other column - always person-level dates (DOB hint, or a date constant
   within an ID, D4) - uses a stratified interpolated inverse-CDF over day
   ordinals, rounded to the day, snapped to the nearest observed weekday when
   some weekday never occurs (ties go to the earlier day), clipped to
   [min, max]. Dates with an observed calendar grid (``day_grid``,
   ``day_of_month``, ``month_end``, ``month_grid``: month-precision dates on
   the 1st, month ends, biweekly or quarterly dates; R2-13) snap to the
   nearest grid date on an observed weekday instead (:func:`grid_days`);
   YEAR_MONTH values to their ``month_grid``.
3. Time of day (DATETIME), TIME and DURATION: stratified inverse-CDF over
   microseconds (nanoseconds when ``params['time_unit'] == 'ns'``), rounded to
   the observed grid ``phase + k * step`` (``granularity_seconds`` and
   ``grid_phase_seconds``: any step, e.g. hourly, every 10 minutes, every hour
   at :15, whole days of duration; R2-13), clipped. A time-of-day grid whose
   step does not divide a day restarts at each midnight (:func:`fit_to_grid`).
   Each value is then snapped to its row's format grid (:func:`fit_to_formats`:
   a date-only or ``%H:%M`` layout in a mixed column truncates when written,
   and so do a row's fraction digits), staying inside the real [min, max].
   YEAR_MONTH: month ordinals (calendar quota or inverse-CDF). Epoch integers
   are datetimes in UTC.
4. D14: the exact min and max (day, month, instant, time or duration) are
   each emitted at least once, in a row whose format writes them exactly
   (``extreme_value_adjusted``; off with ``--no-exact-extremes COL``). When no
   generated row can write an extreme (it has more fraction digits than any
   row got, or its format got no rows), the nearest row takes the layout the
   source wrote it in.
5. Offsets (DATETIME_TZ): the date part and the time of day are modelled as
   written (local wall time: calendar dates, observed weekdays and the time
   grid hold in the written text); each row gets an offset/zone token by exact
   quota over the observed tokens (fixed offsets, US abbreviations, IANA zone
   names), and the bounds and exact extremes are UTC instants (a value clipped
   to a bound written with another offset moves inward onto its local grid).
   When the offsets follow one zone's rules (``detect.dates.zone_rule``: a
   native ``timestamp[tz=America/Chicago]`` column, or text offsets such as
   ``-06:00``/``-05:00`` or ``CST``/``CDT`` that match a zone's DST changes)
   they are never drawn: each value's local wall time is placed in that zone
   and its offset is the zone's (R4-06), so the local hour/minute grid and the
   weekdays hold in the written file across DST changes. A wall time a DST
   change skips is written one DST step later (Python's ``fold=0``), or moves
   to the nearest grid time that exists; :func:`render_dates` writes every
   such value with the zone's offset at its instant, in the observed token
   style.
6. Mixed formats: each row's format is drawn by exact quota at the observed
   row shares; leading/trailing whitespace variants likewise, and so are the
   fraction-digit counts within one format (``frac_digits_pmf``).

:func:`render_dates` writes every value with the component renderer of
``detect/dates_formats.py`` (never ``strftime``), following each format's
render spec (zero padding, month-name case, AM/PM tokens, fraction digits, %y
as two digits even when the year was moved back 100 years).

Internal values
---------------
:func:`generate_dates` returns a structured array of dtype :data:`GEN_DTYPE`:
``text`` (the verbatim token, or None for a generated value), ``fmt`` (index
into ``params['formats']``), ``day`` (day ordinal, days since 1970-01-01),
``us`` (time of day, or the duration, in microseconds; in nanoseconds for
``time_unit`` 'ns'), ``tz`` (offset/zone token), ``lead``/``trail``
(whitespace), ``fd`` (the row's fraction digits, -1 for its format's render
spec). For DATETIME_TZ, ``day``/``us`` are UTC.

Native typed storage contract (AREAS E/F rely on this)
------------------------------------------------------
Readers of typed formats (Excel date cells, Parquet/Feather ``date32``,
``timestamp[unit, tz]``, ``time64``, ``duration``, Stata ``%td``/``%tc``)
render each cell to a **canonical string** in ``RawColumn.values`` (``None``
for a null) and describe the column in ``RawColumn.native``::

    native["temporal"] = {"kind": "date" | "datetime" | "time" | "duration",
                          "tz": None | "UTC" | "America/Chicago" | "+05:30",
                          "unit": None | "s" | "ms" | "us" | "ns" | "D"}

Canonical strings (any form the value detector reads also works, but these
are recommended because they round-trip exactly):

=========  ===============================================================
kind       canonical string
=========  ===============================================================
date       ``YYYY-MM-DD``
datetime   ``YYYY-MM-DD HH:MM:SS`` plus ``.ffffff`` when the column has
           sub-second values (6 digits; 9 for nanosecond units, which are
           kept exactly); tz-aware columns append the offset as ``+HH:MM``
           (local time in the column's zone)
time       ``HH:MM:SS`` plus ``.ffffff`` when needed
duration   ``[-]H:MM:SS`` plus ``.ffffff`` when needed (hours unbounded)
=========  ===============================================================

Detection: ``detect.dates.native_temporal(native)`` reads that description
(also an Arrow type string under ``arrow_type``, a Stata format under
``stata_format``, or an Excel ``cell_type`` 'd' with its ``number_format``);
the AREA E/F ``detect_native`` hooks may return
``detect.dates.native_detection(ctx)`` (or None and let the value rules call
``detect_dates``, which also checks ``ctx.native``). The Detection has a date
type, ``params['native']`` and formats detected from the canonical strings.

Generation output: ``OutTable.columns`` hold :func:`render_dates` strings in
**exactly the canonical form the reader produced** (same detected format, same
padding and fraction digits; tz-aware values carry an observed offset, and in a
column with a named zone the zone's offset at the value's instant, R4-06), or
``None`` for a null. Writers convert them with :func:`to_native` (the source
column's ``ColumnProfile`` is ``OutTable.profile.columns[source_index[k]]``):

* date -> ``datetime.date``; datetime -> ``datetime.datetime`` (naive, or
  aware with the written offset; :func:`to_native` converts aware values to
  ``native['temporal']['tz']`` when that is a zone name);
  time -> ``datetime.time``; duration -> ``datetime.timedelta`` (these hold
  microseconds: nanosecond digits need the text, as the Arrow writer parses it);
* ``None``, NA tokens and unparseable tokens -> ``None``.

Values stay within the source's [min, max], so they fit the source unit
(``timestamp[ns]`` included).
"""

from __future__ import annotations

import datetime as _dt
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from ..detect import dates as D
from ..detect.dates_formats import (
    NO_OFFSET,
    STR,
    US,
    US_DAY,
    Components,
    civil_from_days,
    days_from_civil,
    get_format,
    rule_zone,
    token_offset,
    weekday,
    zone_instant_offsets,
    zone_local_offsets,
    zone_offsets_for_instant,
    zone_offsets_for_local,
)
from ..errors import UnsupportedColumnError
from ..hints import header_hints
from ..model import ColumnModel, ColumnProfile
from .numeric import interp_icdf, stratified_uniform
from .quota import allocate

GEN_DTYPE = np.dtype([("text", object), ("fmt", np.int16), ("day", np.int64), ("us", np.int64), ("tz", object),
                      ("lead", object), ("trail", object), ("fd", np.int8)])


# --------------------------------------------------------------------------- hooks
def generate_dates(profile: ColumnProfile, model: ColumnModel, n: int,
                   rng: np.random.Generator) -> tuple[np.ndarray, dict[str, bool]]:
    """AREA A hook: ``n`` values for a date-family column (see module docstring).

    Called by ``generate/table.py`` with ``n`` = number of non-missing slots and
    the column's model (the per-entity model for within-ID-constant columns).
    Returns (structured array of :data:`GEN_DTYPE`, flags
    ``{'rare_category_adjusted', 'extreme_value_adjusted'}``); :func:`render_dates`
    makes the text.
    """
    flags: dict[str, bool] = {}
    out = _empty(n)
    if n == 0:
        return out, flags
    params = _params(profile, model)
    raws = [r for r in model.nonmissing_counts if r is not None]
    if not raws:
        raise UnsupportedColumnError(f"column {profile.name!r}: no values to sample")
    cnts = np.fromiter((int(model.nonmissing_counts[r]) for r in raws), dtype=np.int64, count=len(raws))
    padded = set(D.padded_raws({r: 1 for r in raws}))
    stripped = [str(r).strip() for r in raws] if padded else [str(r) for r in raws]
    rid, uniques = pd.factorize(np.asarray(stripped, dtype=object))
    v = D.view(params, [str(u) for u in uniques])
    w = np.bincount(rid, weights=cnts, minlength=len(uniques)).astype(np.int64)
    cls_raw = v.cls[rid]
    tok = cls_raw <= D.OTHER_TOKEN
    tok_idx = np.flatnonzero(tok)
    bucket = int(cnts[~tok].sum()) if np.any(v.cls >= 0) else 0
    if bucket == 0 and tok_idx.size == 0:
        raise UnsupportedColumnError(f"column {profile.name!r}: no date values to model",
                                     fix=f"--force-text {profile.name} or --force-factor {profile.name}")
    if tok_idx.size:
        counts = np.concatenate([cnts[tok_idx], [bucket]])
        alloc, adjusted = allocate(counts, n, floor=True)
        flags["rare_category_adjusted"] = bool(adjusted)
        m = int(alloc[-1])
        texts = np.repeat(np.array([raws[i] for i in tok_idx] + [None], dtype=object), alloc)
    else:
        m = n
        texts = np.full(n, None, dtype=object)
    exact = not profile.params.get("_no_exact_extremes", False)
    gen, adj = _generate(profile, params, v, w, m, rng, exact)
    flags["extreme_value_adjusted"] = adj
    k = n - m
    out["text"][:k] = texts[:k]
    out["text"][k:] = None
    out["fmt"][k:] = gen["fmt"]
    out["day"][k:] = gen["day"]
    out["us"][k:] = gen["us"]
    out["tz"][k:] = gen["tz"]
    out["fd"][k:] = gen["fd"]
    lead, trail = _whitespace(padded, raws, cnts, cls_raw, m, rng)
    out["lead"][k:] = lead
    out["trail"][k:] = trail
    return out[rng.permutation(n)], flags


def render_dates(profile: ColumnProfile, values: np.ndarray) -> np.ndarray:
    """AREA A hook: render :func:`generate_dates` values as source text (object array of str).

    Tokens are written verbatim; generated values use their row's format and
    render spec (``detect/dates_formats.py`` component renderer, never
    ``strftime``). Plain string arrays are returned unchanged.
    """
    values = np.asarray(values)
    n = len(values)
    if values.dtype != GEN_DTYPE:
        out = np.empty(n, dtype=object)
        out[:] = [None if x is None else str(x) for x in values.tolist()]
        return out
    params = profile.params
    fmts = params.get("formats") or []
    kind = params.get("kind") or D.column_kind([f["format"] for f in fmts])
    tpu = D.ticks_per_us(params)
    out = np.empty(n, dtype=object)
    text = values["text"]
    is_tok = ~np.equal(text, None).astype(bool)
    if kind == "datetime":
        zone = D.zone_rule(params)
        if zone is not None:
            values = zone_tokens(params, values, zone, ~is_tok, tpu)
    out[is_tok] = text[is_tok]
    fd = values["fd"]
    for fi, entry in enumerate(fmts):
        in_fmt = ~is_tok & (values["fmt"] == fi)
        if not in_fmt.any():
            continue
        f = get_format(str(entry["format"]))
        spec = entry.get("render") or {}
        for d in np.unique(fd[in_fmt]).tolist():  # each row's fraction digits (-1: the format's own)
            rows = np.flatnonzero(in_fmt & (fd == d))
            sub = values[rows]
            comps = _components(kind, f, sub, params, tpu)
            s = f.render(comps, spec if d < 0 else {**spec, "frac_digits": int(d)})
            lead, trail = sub["lead"], sub["trail"]
            if np.any(lead != "") or np.any(trail != ""):
                s = np.strings.add(np.strings.add(np.asarray(lead, dtype=STR), s), np.asarray(trail, dtype=STR))
            out[rows] = s.astype(object)
    return out


def to_native(profile: ColumnProfile, values: Any) -> list[Any]:
    """Python temporal objects for rendered strings (writers of typed formats; contract above).

    ``None``, blanks, NA tokens and text no format reads give ``None``.
    """
    params = dict(profile.params)
    params.pop("sentinels", None)
    params.pop("_other_tokens", None)
    vals = list(values)
    keys = [i for i, x in enumerate(vals) if isinstance(x, str) and x.strip()]
    out: list[Any] = [None] * len(vals)
    if not keys or not params.get("formats"):
        return out
    uniq: dict[str, int] = {}
    rid = [uniq.setdefault(vals[i].strip(), len(uniq)) for i in keys]
    v = D.view(params, list(uniq))
    native_tz = (params.get("native") or {}).get("tz")
    zone = None
    if isinstance(native_tz, str) and token_offset(native_tz) is None:
        try:
            import zoneinfo

            zone = zoneinfo.ZoneInfo(native_tz)
        except Exception:  # noqa: BLE001 - unknown zone: keep the written offset
            zone = None
    kind = v.kind
    tpu = v.tpu
    conv: list[Any] = []
    for j in range(len(uniq)):
        if v.cls[j] < 0:
            conv.append(None)
            continue
        if kind == "duration":  # Python timedeltas hold microseconds: nanoseconds truncate toward zero
            dur = int(v.dur[j])
            conv.append(_dt.timedelta(microseconds=(abs(dur) // tpu) * (-1 if dur < 0 else 1)))
            continue
        y, mo, d = (int(a[0]) for a in civil_from_days(np.array([v.local_day[j]])))
        if kind in ("date", "year_month"):
            conv.append(_dt.date(y, mo, d))
            continue
        sod = int(v.local_sod[j]) // tpu
        h, rem = divmod(sod // US, 3600)
        mi, se = divmod(rem, 60)
        micro = sod % US
        if kind == "time":
            conv.append(_dt.time(h, mi, se, micro))
            continue
        tok = str(v.tz[j] or "")
        stored = params.get("stored_as")
        if stored in ("epoch_s", "epoch_ms"):
            conv.append(_dt.datetime(y, mo, d, h, mi, se, micro, tzinfo=_dt.timezone.utc))
            continue
        off = token_offset(tok) if tok else None
        tzinfo: _dt.tzinfo | None = None
        if off is not None:
            tzinfo = _dt.timezone(_dt.timedelta(minutes=off))
        elif tok:
            try:
                import zoneinfo

                tzinfo = zoneinfo.ZoneInfo(tok)
            except Exception:  # noqa: BLE001
                tzinfo = None
        value = _dt.datetime(y, mo, d, h, mi, se, micro, tzinfo=tzinfo)
        if zone is not None and value.tzinfo is not None:
            value = value.astimezone(zone)
        conv.append(value)
    for i, r in zip(keys, rid):
        out[i] = conv[r]
    return out


# --------------------------------------------------------------------------- zone rules (R4-06)
def _offset_style(params: dict[str, Any]) -> tuple[dict[int, str], str]:
    """(offset minutes -> the observed token most rows wrote it with, numeric style of new offsets)."""
    by_off: dict[int, tuple[int, str]] = {}
    numeric: Counter = Counter()
    for tok, c in (params.get("tz_tokens") or {}).items():
        off = token_offset(str(tok))
        if off is None:
            continue
        if off not in by_off or int(c) > by_off[off][0]:
            by_off[off] = (int(c), str(tok))
        t = str(tok)
        style = "colon" if len(t) == 6 and t[3] == ":" else "plain" if len(t) == 5 and t[1:].isdigit() else \
            "hours" if len(t) == 3 and t[1:].isdigit() else "abbr" if t.isalpha() and t.upper() != "Z" else None
        if style is not None:
            numeric[style] += int(c)
    style = numeric.most_common(1)[0][0] if numeric else "colon"
    return {k: v[1] for k, v in by_off.items()}, style


def _format_offset(minutes: int, style: str) -> str:
    sign = "-" if minutes < 0 else "+"
    h, m = divmod(abs(int(minutes)), 60)
    if style == "plain":
        return f"{sign}{h:02d}{m:02d}"
    if style == "hours" and m == 0:
        return f"{sign}{h:02d}"
    return f"{sign}{h:02d}:{m:02d}"


def zone_tokens(params: dict[str, Any], values: np.ndarray, zone: str, rows: np.ndarray, tpu: int) -> np.ndarray:
    """``values`` (:data:`GEN_DTYPE`) with the offset token of every generated value in a format with an offset
    set to the one ``zone``'s rules give at its instant (R4-06), written as the source wrote that offset
    (``-06:00``, ``-0600``, ``CST``); an offset the source never wrote takes the source's style (a zone
    abbreviation when the source wrote abbreviations)."""
    fmts = params.get("formats") or []
    has_tz = np.array([get_format(str(f["format"])).has_tz for f in fmts] or [False], dtype=bool)
    fmt = np.asarray(values["fmt"], dtype=np.int64)
    sel = np.flatnonzero(rows & has_tz[np.clip(fmt, 0, len(has_tz) - 1)] & (fmt < len(has_tz)))
    if sel.size == 0:
        return values
    inst = values["day"][sel].astype(np.int64) * (US_DAY * tpu) + values["us"][sel].astype(np.int64)
    inst_us = np.floor_divide(inst, tpu)
    off = _zone_offsets_at(zone, inst_us)
    seen, style = _offset_style(params)
    toks: dict[int, str] = {}
    out = values.copy()
    tz = out["tz"]
    for o in np.unique(off).tolist():
        tok = seen.get(int(o))
        if tok is None and style == "abbr":
            hit = np.flatnonzero(off == o)[0]
            name = _zone_name_at(zone, int(inst_us[hit]))
            tok = name if name and token_offset(name) == int(o) else None
        toks[int(o)] = tok or _format_offset(int(o), "colon" if style == "abbr" else style)
    for o, tok in toks.items():
        tz[sel[off == o]] = tok
    return out


def _zone_name_at(zone: str, instant_us: int) -> str | None:
    """The zone's abbreviation at a UTC instant (``CDT``), or None."""
    try:
        import zoneinfo

        t = _dt.datetime(1970, 1, 1, tzinfo=_dt.timezone.utc) + _dt.timedelta(microseconds=instant_us)
        return t.astimezone(zoneinfo.ZoneInfo(zone)).tzname()
    except Exception:  # noqa: BLE001
        return None


def _zone_offsets_at(zone: str, inst_us: np.ndarray) -> np.ndarray:
    """The zone's UTC offsets (minutes) at instants (microseconds): vectorised, else one by one."""
    off = zone_instant_offsets(zone, inst_us)
    if off is not None:
        return off
    import zoneinfo

    z = zoneinfo.ZoneInfo(zone)  # a rule zone (``detect.dates.zone_rule``) always loads
    epoch = _dt.datetime(1970, 1, 1, tzinfo=_dt.timezone.utc)
    out = np.zeros(len(inst_us), dtype=np.int64)
    for i, v in enumerate(np.asarray(inst_us, dtype=np.int64).tolist()):
        try:
            o = (epoch + _dt.timedelta(microseconds=v)).astimezone(z).utcoffset()
            out[i] = int(o.total_seconds() // 60) if o is not None else 0
        except (OverflowError, ValueError):
            out[i] = 0
    return out


def _zone_offset_ticks(zone: str, inst: np.ndarray, tpu: int) -> np.ndarray:
    """The zone's UTC offset (ticks) at instants (ticks)."""
    return _zone_offsets_at(zone, np.floor_divide(inst, tpu)) * 60 * US * tpu


def _on_grid(y: np.ndarray, step: int, phase: int, period: int) -> np.ndarray:
    big = np.iinfo(np.int64)
    return fit_to_grid(y, step, phase, big.min // 4, big.max // 4, period) == y


def zone_settle(inst: np.ndarray, rows: np.ndarray, zone: str, step: int, phase: int, lo: int, hi: int,
                tpu: int, keep: np.ndarray) -> np.ndarray:
    """Values (instants, ticks) of a zone-rule column whose written local time left the observed time grid
    move to the nearest grid time that exists in the zone, inside [lo, hi] (R4-06).

    That happens to a wall time a DST change skips (written one DST step later) when the grid step does not
    divide the DST step, and to a value clipped across a DST change. ``rows`` are the values written with an
    offset; ``keep`` (the exact extremes, real instants) never move.
    """
    day_t = US_DAY * tpu
    if step <= 1 and phase == 0:
        return inst
    sel = np.flatnonzero(rows & ~keep)
    if sel.size == 0:
        return inst
    local = inst[sel] + _zone_offset_ticks(zone, inst[sel], tpu)
    bad = sel[~_on_grid(local, step, phase, day_t)]
    if bad.size == 0:
        return inst
    big = np.iinfo(np.int64)
    out = inst.copy()
    for i in bad.tolist():
        t = int(out[i] + _zone_offset_ticks(zone, out[i:i + 1], tpu)[0])
        cands: list[int] = []
        for k in range(0, 9):
            for c in (t - k * step, t + k * step):
                g = int(fit_to_grid(np.array([c], dtype=np.int64), step, phase, big.min // 4, big.max // 4,
                                    day_t)[0])
                if g not in cands:
                    cands.append(g)
        best = None
        for g in sorted(cands, key=lambda g: (abs(g - t), g)):
            lo_us = zone_local_offsets(zone, np.array([g // tpu], dtype=np.int64))
            if lo_us is None:
                break
            cand = g - int(lo_us[0]) * 60 * US * tpu
            back = cand + int(_zone_offset_ticks(zone, np.array([cand], dtype=np.int64), tpu)[0])
            if back == g and lo <= cand <= hi:
                best = cand
                break
        if best is not None:
            out[i] = best
    return out


# --------------------------------------------------------------------------- internals
def _empty(n: int) -> np.ndarray:
    out = np.zeros(n, dtype=GEN_DTYPE)
    for f in ("text", "tz", "lead", "trail"):
        out[f] = np.full(n, None if f == "text" else "", dtype=object)
    out["fd"] = -1
    return out


def _params(profile: ColumnProfile, model: ColumnModel) -> dict[str, Any]:
    """The column's AREA A params; re-detected from the model when a foreign detection has none."""
    params = profile.params
    if params.get("formats"):
        return params
    typed: Counter = Counter()
    for raw, c in model.nonmissing_counts.items():
        if isinstance(raw, str) and raw.strip():
            typed[raw.strip()] += c
    for forced in ((profile.type if profile.type in D.DATE_TYPES else None), None):
        inp = D.DateInputs(name=profile.name, typed=dict(typed), raw_counts=dict(model.nonmissing_counts),
                           hints=header_hints(profile.name), forced=forced,
                           native=D.native_temporal(profile.native))
        det = D.detect_from_inputs(inp)
        if det is not None:
            for key, val in det.params.items():
                params.setdefault(key, val)
            return params
    raise UnsupportedColumnError(f"column {profile.name!r}: {profile.type.value} values could not be read as dates",
                                 fix=f"--date-format {profile.name}=FMT, or --force-text {profile.name}")


def _quota_values(values: np.ndarray, weights: np.ndarray, m: int, rng: np.random.Generator) -> np.ndarray:
    alloc, _ = allocate(np.asarray(weights, dtype=float), m, floor=False)
    return rng.permutation(np.repeat(values, alloc))


def _icdf(values: np.ndarray, weights: np.ndarray, m: int, rng: np.random.Generator) -> np.ndarray:
    xs = np.sort(np.repeat(np.asarray(values, dtype=float), np.asarray(weights, dtype=np.int64)))
    return interp_icdf(xs, stratified_uniform(m, rng))


def snap_weekdays(y: np.ndarray, allowed: list[int] | set[int], lo: int, hi: int) -> np.ndarray:
    """Move day ordinals on never-observed weekdays to the nearest observed one (earlier on ties)."""
    ok = np.zeros(7, dtype=bool)
    ok[list(allowed)] = True
    if ok.all() or not ok.any():
        return y
    delta = np.zeros(7, dtype=np.int64)
    for wd in range(7):
        if ok[wd]:
            continue
        for k in range(1, 4):
            if ok[(wd - k) % 7]:
                delta[wd] = -k
                break
            if ok[(wd + k) % 7]:
                delta[wd] = k
                break
    y = y + delta[weekday(y)]
    return np.clip(y, lo, hi)


def ensure_extremes(y: np.ndarray, lo: int, hi: int, res: np.ndarray | None = None,
                    prefer: dict[int, np.ndarray] | None = None) -> bool:
    """D14: put the exact ``lo`` and ``hi`` in ``y`` (in place) by replacing the nearest values.

    ``res`` (per-row render step, see :func:`fit_to_formats`) restricts each
    extreme to rows whose format can write it exactly (a time-of-day minimum
    never lands in a date-only row, where rendering would drop the time).
    ``prefer`` ({extreme: row mask}) narrows that choice further when some of
    those rows qualify (rows with the offset token the source wrote the extreme
    with, so its local time is the source's).
    """
    if len(y) == 0 or lo == hi:
        return False
    need = [t for t in (lo, hi) if not np.any(y == t)]
    if not need:
        return False
    rows = np.arange(len(y))
    cand = {t: rows if res is None else np.flatnonzero(t % res == 0) for t in need}
    if prefer:
        for t in need:
            pm = prefer.get(t)
            if pm is not None and np.any(pm[cand[t]]):
                cand[t] = cand[t][pm[cand[t]]]
    protected = np.zeros(len(y), dtype=bool)  # the only copy of an extreme is never replaced
    for t in (lo, hi):
        hit = np.flatnonzero(y == t)
        if hit.size == 1:
            protected[hit] = True
    adjusted = False
    for t in sorted(need, key=lambda t: len(cand[t])):  # the more constrained extreme first
        c = cand[t][~protected[cand[t]]]
        if c.size == 0:
            continue
        i = int(c[np.argmin(y[c])] if t == lo else c[np.argmax(y[c])])
        y[i] = t
        protected[i] = True
        adjusted = True
    return adjusted


def row_resolution(params: dict[str, Any], fmt: np.ndarray, fd: np.ndarray | None = None,
                   tpu: int = 1) -> np.ndarray:
    """Per-row render step (us; ``tpu`` ticks per us) of each row's format (``DateFormat.resolution_us``).

    ``fd`` gives a row's own fraction digits (-1: the format's render spec).
    """
    fmts = params.get("formats") or []
    steps = np.array([max(int(get_format(str(f["format"])).resolution_us(f.get("render") or {}, tpu)), 1)
                      for f in fmts] or [1], dtype=np.int64)
    fmt = np.asarray(fmt, dtype=np.int64)
    res = steps[fmt]
    if fd is not None and np.any(fd >= 0):
        for fi, f in enumerate(fmts):
            sel = (fmt == fi) & (fd >= 0)
            if not sel.any():
                continue
            fobj = get_format(str(f["format"]))
            spec = f.get("render") or {}
            for d in np.unique(fd[sel]).tolist():
                res[sel & (fd == d)] = max(int(fobj.resolution_us({**spec, "frac_digits": int(d)}, tpu)), 1)
    return res


def _digit_variants(params: dict[str, Any], fmt: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Per-row fraction digits by exact quota over each format's observed digit counts (-1: one count only)."""
    fd = np.full(len(fmt), -1, dtype=np.int8)
    for fi, f in enumerate(params.get("formats") or []):
        pmf = (f.get("render") or {}).get("frac_digits_pmf")
        if not pmf:
            continue
        rows = np.flatnonzero(fmt == fi)
        if rows.size == 0:
            continue
        digits = np.array([int(k) for k in pmf], dtype=np.int8)
        fd[rows] = _quota_values(digits, np.array([float(c) for c in pmf.values()]), rows.size, rng)
    return fd


def _source_layouts(v: D.ColumnView, w: np.ndarray, idx: np.ndarray, vals: np.ndarray,
                    targets: tuple[int, int]) -> dict[int, tuple[int, int, str]]:
    """Per target value: the (format index, fraction digits, offset token) most real rows wrote it in."""
    out: dict[int, tuple[int, int, str]] = {}
    digits = v.digits if v.digits is not None else np.full(len(v.cls), -1, dtype=np.int64)
    for t in targets:
        at = idx[vals == t]
        if at.size == 0:
            continue
        best: Counter = Counter()
        for i in at.tolist():
            best[(int(v.cls[i]), int(digits[i]), str(v.tz[i] or ""))] += int(w[i])
        out[t] = max(best.items(), key=lambda kv: (kv[1], -kv[0][0], kv[0][1], kv[0][2]))[0]
    return out


def _writable_extremes(y: np.ndarray, lo: int, hi: int, res: np.ndarray, layouts: dict[int, tuple[int, int, str]],
                       params: dict[str, Any], fmt: np.ndarray, fd: np.ndarray, tz: np.ndarray, tpu: int,
                       by_token: bool = False) -> bool:
    """D14 for times of day, datetimes and durations (in place on ``y``, ``fmt``, ``fd``, ``tz``, ``res``).

    Each exact extreme goes to a row whose layout writes it exactly. When no row
    can (the source wrote it with more fraction digits than any generated row
    has, or in a format that got no rows), the row nearest to it first takes the
    layout the source wrote it in. With ``by_token``, rows carrying the source
    extreme's offset token are preferred.
    """
    if len(y) == 0 or lo == hi:
        return False
    fmts = params.get("formats") or []
    changed = False
    for t in (lo, hi):
        if np.any(y == t) or np.any(t % res == 0) or t not in layouts:
            continue
        fi, d, tok = layouts[t]
        free = np.flatnonzero((y != lo) & (y != hi))
        if free.size == 0 or not 0 <= fi < len(fmts):
            continue
        i = int(free[np.argmin(np.abs(y[free] - t))])
        fobj = get_format(str(fmts[fi]["format"]))
        spec = fmts[fi].get("render") or {}
        fmt[i], fd[i] = fi, d
        tz[i] = tok if fobj.has_tz else ""
        res[i] = max(int(fobj.resolution_us(spec if d < 0 else {**spec, "frac_digits": int(d)}, tpu)), 1)
        changed = True
    prefer = None
    if by_token:
        prefer = {t: np.asarray(tz == layouts[t][2], dtype=bool) for t in (lo, hi) if t in layouts}
    return bool(ensure_extremes(y, lo, hi, res, prefer) or changed)


def _offset_ticks(tz: np.ndarray, local: np.ndarray, tpu: int) -> np.ndarray:
    """UTC offset per row in ticks for local wall times ``local`` (0 where a row has no token)."""
    off = np.zeros(len(local), dtype=np.int64)
    for name in [str(x) for x in pd.unique(tz) if x]:
        sel = tz == name
        fixed = token_offset(name)
        if fixed is not None:
            off[sel] = fixed
            continue
        loc = np.floor_divide(local[sel], tpu)
        uniq, inv = np.unique(loc, return_inverse=True)
        o = zone_local_offsets(name, uniq) if rule_zone(name) is not None else None
        if o is None:
            o = zone_offsets_for_local(name, uniq)
        o = o[inv]
        off[sel] = np.where(o == NO_OFFSET, 0, o)
    return off * 60 * US * tpu


def fit_to_formats(y: np.ndarray, res: np.ndarray, lo: int, hi: int, toward_zero: bool = False) -> np.ndarray:
    """Snap values to each row's render grid (what the renderer writes) without leaving [lo, hi].

    Renderers truncate (dates to the day, ``%H:%M`` to the minute ...; durations
    toward zero), so a value just above ``lo`` in a coarser format would be
    written below the real minimum: such values move one step inward.
    """
    y = np.asarray(y, dtype=np.int64)
    if y.size == 0 or not np.any(res > 1):
        return y
    if toward_zero:
        q = np.sign(y) * ((np.abs(y) // res) * res)
    else:
        q = (y // res) * res
    q = np.where(q < lo, q + res, q)
    q = np.where(q > hi, q - res, q)
    # no grid point of that format inside [lo, hi]: keep the clipped value
    return np.where((q < lo) | (q > hi), np.clip(y, lo, hi), q)


def fit_to_grid(y: np.ndarray, step: int, phase: int, lo: Any, hi: Any, period: int | None = None) -> np.ndarray:
    """Snap values down onto the observed grid ``phase + k * step`` without leaving [lo, hi] (R2-13).

    With ``period`` (a day, for wall-clock times) the grid restarts every
    period, so a step that does not divide a day (every 7 hours: 00, 07, 14,
    21) keeps the observed times of day on every date. A value whose grid
    point falls outside [lo, hi] moves one grid point inward (like
    :func:`fit_to_formats`); with no grid point inside, the value is clipped.
    ``lo``/``hi`` may be per-row arrays.
    """
    y = np.asarray(y, dtype=np.int64)
    if y.size == 0 or (step <= 1 and phase == 0):
        return y
    if period is None or period % step == 0:  # one global grid
        q = phase + np.floor_divide(y - phase, step) * step
        up, down = q + step, q - step
    else:
        last = (period - 1 - phase) // step  # index of a period's last grid point
        d, s = np.divmod(y, period)
        k = np.floor_divide(s - phase, step)
        before = k < 0  # before the period's first grid point: the previous period's last one
        d = np.where(before, d - 1, d)
        k = np.where(before, last, k)
        q = d * period + phase + k * step
        up = np.where(k < last, q + step, (d + 1) * period + phase)
        down = np.where(k > 0, q - step, (d - 1) * period + phase + last * step)
    q = np.where(q < lo, up, q)
    q = np.where(q > hi, down, q)
    return np.where((q < lo) | (q > hi), np.clip(y, lo, hi), q)


def _round_step(y: np.ndarray, step: int, lo: int, hi: int, phase: int = 0) -> np.ndarray:
    """Round to the nearest point of the grid ``phase + k * step`` and clip to [lo, hi] (observed grid points)."""
    r = np.rint((np.asarray(y, dtype=float) - phase) / step).astype(np.int64) * step + phase
    return np.clip(r, lo, hi)


def _grid(params: dict[str, Any], tpu: int) -> tuple[int, int]:
    """The column's observed time grid (step, phase) in ticks (``detect.dates.time_grid``)."""
    step = int(round(float(params.get("granularity_seconds", 60.0)) * US * tpu)) or 1
    phase = int(round(float(params.get("grid_phase_seconds", 0.0)) * US * tpu)) % step
    return step, phase


def grid_days(params: dict[str, Any], lo: int, hi: int, weekdays: list[int] | set[int]) -> np.ndarray | None:
    """Every day ordinal in [lo, hi] on the column's observed calendar grid (R2-13), or None without one.

    The grid is ``detect.dates.date_grid``: a day lattice (``day_grid``), a fixed
    day of the month (``day_of_month``) or the month's last day (``month_end``),
    with a month lattice (``month_grid``); dates on never-observed weekdays are
    left out. ``lo`` and ``hi`` (observed values) are always on it.
    """
    dg, dom, me, mg = (params.get(k) for k in ("day_grid", "day_of_month", "month_end", "month_grid"))
    if not (dg or dom or me):
        return None
    if dom or me:
        (y0,), (m0,), _ = civil_from_days(np.array([lo]))
        (y1,), (m1,), _ = civil_from_days(np.array([hi]))
        mo = np.arange((y0 - 1970) * 12 + m0 - 1, (y1 - 1970) * 12 + m1, dtype=np.int64)
        if mg:
            mo = mo[(mo - int(mg[1])) % int(mg[0]) == 0]
        yy, mm = 1970 + mo // 12, mo % 12 + 1
        first = days_from_civil(yy, mm, np.ones_like(mm))
        length = D.month_length(yy, mm)
        if me:
            cand = first + length - 1
        else:
            cand = (first + int(dom) - 1)[int(dom) <= length]  # months without that day have no grid date
    else:
        step, phase = int(dg[0]), int(dg[1])
        cand = np.arange(lo + (phase - lo) % step, hi + 1, step, dtype=np.int64)
    cand = cand[(cand >= lo) & (cand <= hi)]
    if dg:
        cand = cand[(cand - int(dg[1])) % int(dg[0]) == 0]
    if 0 < len(set(weekdays)) < 7:
        cand = cand[np.isin(weekday(cand), list(weekdays))]
    return cand


def snap_to(y: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    """Each value moved to the nearest of the sorted ``allowed`` values (the earlier one on ties)."""
    if allowed.size == 0:
        return y
    j = np.searchsorted(allowed, y, side="left")
    up = allowed[np.minimum(j, allowed.size - 1)]
    dn = allowed[np.maximum(j - 1, 0)]
    return np.where(np.abs(y - dn) <= np.abs(up - y), dn, up)


def _gen_days(profile: ColumnProfile, params: dict[str, Any], days: np.ndarray, w: np.ndarray, m: int,
              rng: np.random.Generator) -> np.ndarray:
    lo, hi = int(days.min()), int(days.max())
    if _use_calendar(profile, params):
        uv, inv = np.unique(days, return_inverse=True)
        uw = np.bincount(inv, weights=w)
        return _quota_values(uv, uw, m, rng).astype(np.int64)
    y = np.clip(np.rint(_icdf(days, w, m, rng)).astype(np.int64), lo, hi)
    allowed = sorted({int(x) for x in np.unique(weekday(days))})
    grid = grid_days(params, lo, hi, allowed)
    if grid is not None:  # R2-13: month-precision, month-end, biweekly ... dates stay on their observed grid
        return snap_to(y, grid)
    return snap_weekdays(y, allowed, lo, hi)


def _use_calendar(profile: ColumnProfile, params: dict[str, Any]) -> bool:
    if not params.get("calendar"):
        return False
    if params.get("person_level"):
        return False
    if profile.role == "entity":
        params["calendar"] = False
        params["calendar_reason"] = (str(params.get("calendar_reason", "")) +
                                     "; constant within each ID (person-level): never restricted to observed dates")
        return False
    return True


def _generate(profile: ColumnProfile, params: dict[str, Any], v: D.ColumnView, w: np.ndarray, m: int,
              rng: np.random.Generator, exact: bool) -> tuple[dict[str, np.ndarray], bool]:
    kind = v.kind
    tpu = v.tpu  # ticks per microsecond of v.sod / v.dur (1000 for nanosecond columns)
    day_t = US_DAY * tpu
    idx = np.flatnonzero(v.cls >= 0)
    wv = w[idx]
    n_fmt = len(params.get("formats") or [])
    day = np.zeros(m, dtype=np.int64)
    us = np.zeros(m, dtype=np.int64)
    tz = np.full(m, "", dtype=object)
    fd = np.full(m, -1, dtype=np.int8)
    if m == 0 or idx.size == 0:
        return {"fmt": np.zeros(m, dtype=np.int16), "day": day, "us": us, "tz": tz, "fd": fd}, False
    fmt_w = np.bincount(v.cls[idx], weights=wv, minlength=n_fmt)
    fmt = _quota_values(np.arange(n_fmt, dtype=np.int16), fmt_w, m, rng).astype(np.int16)
    if kind in ("datetime", "time", "duration"):
        fd = _digit_variants(params, fmt, rng)
    adjusted = False
    step, phase = _grid(params, tpu)
    if kind == "date":
        days = v.day[idx]
        day = _gen_days(profile, params, days, wv, m, rng)
        if exact:
            adjusted = ensure_extremes(day, int(days.min()), int(days.max()))
    elif kind == "year_month":
        mons = v.mon[idx]
        lo, hi = int(mons.min()), int(mons.max())
        if _use_calendar(profile, params):
            uv, inv = np.unique(mons, return_inverse=True)
            mo = _quota_values(uv, np.bincount(inv, weights=wv), m, rng).astype(np.int64)
        else:
            mo = np.clip(np.rint(_icdf(mons, wv, m, rng)).astype(np.int64), lo, hi)
            mg = params.get("month_grid")
            if mg:  # R2-13: quarterly / yearly months stay on their observed grid
                mo = _round_step(mo, int(mg[0]), lo, hi, int(mg[1]))
        if exact:
            adjusted = ensure_extremes(mo, lo, hi)
        day = days_from_civil(1970 + mo // 12, mo % 12 + 1, np.ones(m, dtype=np.int64))
    elif kind == "datetime":
        toks = params.get("tz_tokens") or {}
        tz_fmt = np.array([get_format(str(f["format"])).has_tz for f in params["formats"]], dtype=bool)
        # values with offsets: the date part and the time of day are modelled as written (local wall
        # time), so observed weekdays, calendar dates and the time grid hold in the written text; the
        # bounds and the exact extremes are UTC instants
        local_model = bool(toks) and bool(tz_fmt.any())
        # R4-06: offsets that follow a zone's rules are derived from the zone, never drawn
        zone = D.zone_rule(params) if local_model else None
        days = (v.local_day if local_model else v.day)[idx]
        day = _gen_days(profile, params, days, wv, m, rng)
        timed = idx[v.has_time[idx]]
        if timed.size:
            sod = (v.local_sod if local_model else v.sod)[timed]
            tw = w[timed]
            us = _round_step(_icdf(sod, tw, m, rng), step, int(sod.min()), int(sod.max()), phase)
        inst_real = v.day[idx] * day_t + v.sod[idx]
        lo, hi = int(inst_real.min()), int(inst_real.max())
        if toks:
            tz_rows = np.flatnonzero(tz_fmt[fmt])
            if tz_rows.size and zone is not None:
                tz[tz_rows] = zone  # each value's offset is the zone's at its local wall time
            elif tz_rows.size:
                real_tok: Counter = Counter()
                for i in idx:
                    if v.tz[i]:
                        real_tok[v.tz[i]] += int(w[i])
                names = list(real_tok) or list(toks)
                weights = [real_tok[t] for t in names] if real_tok else [toks[t] for t in names]
                tz[tz_rows] = _quota_values(np.array(names, dtype=object), np.array(weights, dtype=float),
                                            tz_rows.size, rng)
        res = row_resolution(params, fmt, fd, tpu)
        # the date part and the time of day are drawn independently: clip the instant to [min, max],
        # then snap it to its row's format (a date-only or minute layout in a mixed column truncates)
        if local_model:
            local = day * day_t + us
            off = _offset_ticks(tz, local, tpu)
            # a value clipped to a bound written with another offset moves inward onto its local time grid
            local = fit_to_grid(np.clip(local - off, lo, hi) + off, step, phase, lo + off, hi + off, day_t)
            inst = fit_to_formats(local - off, res, lo, hi)
        else:
            inst = fit_to_formats(np.clip(day * day_t + us, lo, hi), res, lo, hi)
        if exact:
            layouts = _source_layouts(v, w, idx, inst_real, (lo, hi))
            adjusted = _writable_extremes(inst, lo, hi, res, layouts, params, fmt, fd, tz, tpu,
                                          by_token=local_model and zone is None)
        if zone is not None:
            zoned = tz_fmt[fmt]
            tz[zoned] = zone
            inst = zone_settle(inst, zoned, zone, step, phase, lo, hi, tpu, (inst == lo) | (inst == hi))
        day, us = np.divmod(inst, day_t)
    elif kind == "time":
        sod = v.sod[idx]
        lo, hi = int(sod.min()), int(sod.max())
        res = row_resolution(params, fmt, fd, tpu)
        us = fit_to_formats(_round_step(_icdf(sod, wv, m, rng), step, lo, hi, phase), res, lo, hi)
        if exact:
            layouts = _source_layouts(v, w, idx, sod, (lo, hi))
            adjusted = _writable_extremes(us, lo, hi, res, layouts, params, fmt, fd, tz, tpu)
    else:  # duration
        dur = v.dur[idx]
        lo, hi = int(dur.min()), int(dur.max())
        res = row_resolution(params, fmt, fd, tpu)
        us = fit_to_formats(_round_step(_icdf(dur, wv, m, rng), step, lo, hi, phase), res, lo, hi, toward_zero=True)
        if exact:
            layouts = _source_layouts(v, w, idx, dur, (lo, hi))
            adjusted = _writable_extremes(us, lo, hi, res, layouts, params, fmt, fd, tz, tpu)
    return {"fmt": fmt, "day": day, "us": us, "tz": tz, "fd": fd}, bool(adjusted)


def _whitespace(padded: set[str], raws: list[str], cnts: np.ndarray, cls_raw: np.ndarray, m: int,
                rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Leading/trailing whitespace per generated value, by exact quota over the dated raw values."""
    lead = np.full(m, "", dtype=object)
    trail = np.full(m, "", dtype=object)
    if m == 0 or not padded:
        return lead, trail
    combos: Counter = Counter()
    plain = int(cnts[cls_raw >= 0].sum())
    for i, r in enumerate(raws):
        if r in padded and cls_raw[i] >= 0:
            combos[(r[: len(r) - len(r.lstrip())], r[len(r.rstrip()):])] += int(cnts[i])
            plain -= int(cnts[i])
    if not combos:
        return lead, trail
    combos[("", "")] += max(plain, 0)
    keys = list(combos)
    pick = _quota_values(np.arange(len(keys)), np.array([combos[k] for k in keys], dtype=float), m, rng)
    for j, (a, b) in enumerate(keys):
        sel = pick == j
        lead[sel] = a
        trail[sel] = b
    return lead, trail


def _components(kind: str, f: Any, sub: np.ndarray, params: dict[str, Any], tpu: int = 1) -> Components:
    day = sub["day"].astype(np.int64)
    us = sub["us"].astype(np.int64)
    if kind == "duration":
        return Components.from_duration(us, tpu)
    if kind == "time":
        return Components.from_local(np.zeros(len(us), dtype=np.int64), us, tpu=tpu)
    if kind in ("date", "year_month"):
        return Components.from_local(day, np.zeros(len(day), dtype=np.int64))
    day_t = US_DAY * tpu
    inst = day * day_t + us
    toks = sub["tz"]
    local = inst.copy()
    names = [str(x) for x in pd.unique(toks) if x]
    if names:
        off = np.zeros(len(inst), dtype=np.int64)
        for name in names:
            sel = toks == name
            fixed = token_offset(name)
            if fixed is not None:
                off[sel] = fixed
            else:
                o = zone_offsets_for_instant(name, np.floor_divide(inst[sel], tpu))
                off[sel] = np.where(o == NO_OFFSET, 0, o)
        local = inst + off * 60 * US * tpu
    ld, ls = np.divmod(local, day_t)
    return Components.from_local(ld, ls, tz=np.asarray(toks, dtype=STR),
                                 instant_us=inst if tpu == 1 else np.floor_divide(inst, tpu), tpu=tpu)


__all__ = ["GEN_DTYPE", "ensure_extremes", "fit_to_formats", "fit_to_grid", "generate_dates", "render_dates",
           "row_resolution", "snap_weekdays", "to_native"]
