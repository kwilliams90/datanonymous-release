"""Parquet and Feather reader/writer (Arrow schema preserved).

AREA F owns this module (design 6 "Parquet/Feather", section 4, D17).

Reading
-------
The file is read with pyarrow (the Arrow-native reading that pandas'
``dtype_backend='pyarrow'`` wraps), so the exact schema survives: field types,
nullability, field metadata and schema (file) metadata are kept in
``FileLayout.extra['_arrow_schema']`` (private: metadata may quote anything);
codec per column, row-group size, Parquet format version and whether the Arrow
schema was stored (Parquet) / IPC codec and record-batch size (Feather) are kept
as public layout facts. Each cell is rendered to a canonical string in
``RawColumn.values`` (``None`` = Arrow null) and the column's Arrow details go to
``RawColumn.native`` (``family='arrow'``, ``kind``, the source ``pa.Field``):

==============  ====================================================================
Arrow type      canonical text (and detection)
==============  ====================================================================
bool            ``true`` / ``false`` (value rules: BOOLEAN / FLAG ...)
int*/uint*      ``123`` (value rules)
float*          shortest round-trip positional text of that width (``2.0``, ``0.1``),
                ``NaN``, ``inf``, ``-inf`` (value rules; the non-finite ones are
                numbers the column stores natively, ``model.NUMERIC_SPECIALS_KEY``:
                kept verbatim at their exact counts like ``NaN``, out of the
                statistics, never text tokens, K-20)
decimal         ``12.50`` with the type's scale (value rules: header rules such as D10 org
                codes first, then DECIMAL from the Arrow type at the numeric rule; R4-05)
string          the string itself (value rules: IDs, PII, text ...)
dictionary      the value's text -> FACTOR, or ORDINAL for ``ordered=True`` (native;
                the labels some row holds kept in the dictionary order, a label
                the D35 allowlist did not keep written as its pseudonym in its
                place; an entry no row holds was never checked and is written
                only when it is a vocabulary word such as a grade level:
                vocabulary_label)
date32/64       ``YYYY-MM-DD`` -> DATE through AREA A (``ctx.forced_type``)
timestamp       ``YYYY-MM-DD HH:MM:SS[.f]`` -> DATETIME; with a zone the local wall
                time plus ``+HH:MM`` offset -> DATETIME_TZ (AREA A)
time32/64       ``HH:MM:SS[.f]`` -> TIME (AREA A)
duration        ``[-]HH:MM:SS[.f]`` (hours may exceed 23) -> DURATION (AREA A)
==============  ====================================================================

Fractions of a second are written with the unit's digits (3/6/9) on every value
of a column when any value has one, else omitted. These canonical forms are the
contract with AREA A: :func:`native_date_detection` calls ``detect_dates`` with
``ctx.forced_type`` set to the type implied by the Arrow type, and the writer
parses whatever AREA A renders in the same forms (``T`` separator, ``Z`` and
``+HHMM`` offsets are accepted too). If no date generator accepts the column
the file FAILs with the fix ``--null COL`` / ``--exclude COL``.

Binary, nested (list/struct/map/union), interval and extension columns are never
stringified or copied: their values are read as ``None`` and, unless every value
is null or the column is ``--null`` / ``--exclude``, the file FAILs before
writing with that exact fix (design 4). A Parquet file inside a hive partition
folder (``key=value``) FAILs: partitioned datasets are not supported in v1.

Metadata (privacy)
------------------
Arrow schema and field metadata are free-form key/value text: a pipeline can put
anything there (a data owner's email, a dataset description quoting a patient,
``DataFrame.attrs``). Only **structural** metadata is copied to the output
(:func:`safe_schema_metadata`, :func:`safe_field_metadata`): the ``pandas`` key
rebuilt from its structural parts (column/index layout, dtype names, creator
library and version; ``attributes`` and index or column-level names that are not
column names are left out) and the ``PARQUET:field_id`` of each field (an
integer). Every other key is dropped and the file gets the warning
``arrow_metadata_dropped`` (counts only, never a key or value).

Writing
-------
Each output column is converted back to the source field's exact type (strings
parsed per type; dictionaries rebuilt on the source entries the synthetic rows
hold, plus unused tool-vocabulary entries, in the source order, for label-safe
types, else on the synthetic values; the pandas ``num_categories`` follows the
written dictionary), the table is assembled on the source
fields and schema with their structural metadata only (see above), cast to that
schema, and written with the source codec(s), row-group size, Parquet version,
dictionary/statistics settings, stored-schema flag and legacy INT96 timestamp
storage (Spark/Hive files; Feather: IPC version, codec and record-batch size).
``--null`` keeps the field type with all values null (a dictionary column gets
an empty dictionary: its real labels are never copied; a NOT NULL field cannot
be nulled: the file FAILs naming ``--exclude``). A synthetic value the field's
type cannot hold (e.g. synthetic IDs wider than an ``int8``) FAILs the file with
``--null`` / ``--exclude`` (the value itself is never quoted).
``write_parquet`` also writes SAS input (D17; ``io/statio.py`` builds the
layout). :func:`layout_diff` compares the re-read schema field by field
(type, nullability, metadata), the schema metadata (both against the source's
structural metadata: anything else in the output is a difference), dictionary
entries no row holds (never written unless tool vocabulary), codecs, row-group/batch
size and versions.
"""

from __future__ import annotations

import datetime as _dt
import json
import re
import struct
import warnings
from decimal import ROUND_HALF_EVEN, Context, Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

import numpy as np

from ..errors import ReadError, UnsupportedColumnError, WriteError
from ..model import (LABEL_SAFE_TYPES, NUMERIC_SPECIALS_KEY, TYPE_INFO, ColumnType, FileLayout, Message, RawColumn,
                     RawTable, ReadResult)

if TYPE_CHECKING:  # pragma: no cover
    from .registry import ReadRequest, WriteRequest

SCHEMA_KEY = "_arrow_schema"
TEMPORAL_KINDS = ("date", "timestamp", "time", "duration")
_SCALAR_KINDS = frozenset({"null", "bool", "int", "float", "decimal", "string", *TEMPORAL_KINDS})
UNIT_PER = {"s": 1, "ms": 10**3, "us": 10**6, "ns": 10**9}
UNIT_DIGITS = {"s": 0, "ms": 3, "us": 6, "ns": 9}
_DAY = 86400
_EPOCH_ORDINAL = _dt.date(1970, 1, 1).toordinal()
_HIVE_RE = re.compile(r"^[^=/\\]+=[^/\\]*$")
#: Decimal arithmetic for the writer's cast back (decimal256 holds 76 digits; the default context 28).
_DEC_CONTEXT = Context(prec=400, Emax=999999, Emin=-999999)
_PARQUET_CODEC_OUT = {"UNCOMPRESSED": "none", "SNAPPY": "snappy", "GZIP": "gzip", "BROTLI": "brotli",
                      "LZ4": "lz4", "LZ4_RAW": "lz4", "ZSTD": "zstd"}


class _CannotRender(Exception):
    """A column's values cannot be rendered to canonical text (treated as unsupported)."""


# --------------------------------------------------------------------------- kinds
def arrow_kind(t: Any) -> str:
    """Rendering kind of an Arrow type ('unsupported' for binary/nested/extension/interval)."""
    import pyarrow.types as pt

    if pt.is_null(t):
        return "null"
    if pt.is_boolean(t):
        return "bool"
    if pt.is_integer(t):
        return "int"
    if pt.is_floating(t):
        return "float"
    if pt.is_decimal(t):
        return "decimal"
    if pt.is_string(t) or pt.is_large_string(t) or (hasattr(pt, "is_string_view") and pt.is_string_view(t)):
        return "string"
    if pt.is_date(t):
        return "date"
    if pt.is_timestamp(t):
        return "timestamp"
    if pt.is_time(t):
        return "time"
    if pt.is_duration(t):
        return "duration"
    if pt.is_dictionary(t):
        return "dictionary" if arrow_kind(t.value_type) in _SCALAR_KINDS - {"null"} else "unsupported"
    return "unsupported"


def _unit_of(t: Any) -> str:
    return str(getattr(t, "unit", "s"))


# --------------------------------------------------------------------------- canonical temporal text
def format_dates(days: np.ndarray) -> list[str]:
    """``YYYY-MM-DD`` for integer days since 1970-01-01."""
    return np.datetime_as_string(np.asarray(days, dtype=np.int64).astype("datetime64[D]"), unit="D").tolist()


def format_datetimes(secs: np.ndarray, frac: np.ndarray, digits: int, offsets: np.ndarray | None = None) -> list[str]:
    """``YYYY-MM-DD HH:MM:SS[.f][+HH:MM]`` for wall-clock seconds since 1970 (``frac`` in units of 10**-digits s)."""
    base = np.datetime_as_string(np.asarray(secs, dtype=np.int64).astype("datetime64[s]"), unit="s")
    out = [s.replace("T", " ") for s in base.tolist()]
    if digits:
        out = [f"{s}.{int(f):0{digits}d}" for s, f in zip(out, np.asarray(frac).tolist())]
    if offsets is not None:
        out = [s + format_offset(int(o)) for s, o in zip(out, np.asarray(offsets).tolist())]
    return out


def format_offset(seconds: int) -> str:
    sign = "-" if seconds < 0 else "+"
    s = abs(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{sign}{h:02d}:{m:02d}" + (f":{sec:02d}" if sec else "")


def format_clock(units: np.ndarray, per: int, digits: int) -> list[str]:
    """``[-]HH:MM:SS[.f]`` for a count of ``1/per`` seconds (hours may exceed 23)."""
    out: list[str] = []
    for v in np.asarray(units, dtype=np.int64).tolist():
        sign = "-" if v < 0 else ""
        s, f = divmod(abs(v), per)
        h, rem = divmod(s, 3600)
        m, sec = divmod(rem, 60)
        text = f"{sign}{h:02d}:{m:02d}:{sec:02d}"
        if digits:
            text += f".{f * 10**digits // per:0{digits}d}"
        out.append(text)
    return out


_DT_RE = re.compile(r"\s*([+-]?\d{4,})-(\d{1,2})-(\d{1,2})(?:[ T](\d{1,2}):(\d{2})(?::(\d{2})(?:[.,](\d{1,9}))?)?)?"
                    r"\s*(Z|[+-]\d{2}(?::?\d{2}(?::?\d{2})?)?)?\s*")
_CLOCK_RE = re.compile(r"\s*([+-])?(\d+):(\d{2})(?::(\d{2})(?:[.,](\d{1,9}))?)?\s*")


def _days_from_civil(y: int, m: int, d: int) -> int:
    return _dt.date(y, m, d).toordinal() - _EPOCH_ORDINAL


def _frac_ns(text: str | None) -> int:
    if not text:
        return 0
    return int((text + "000000000")[:9])


def _offset_seconds(text: str | None) -> int | None:
    if text is None:
        return None
    if text == "Z":
        return 0
    sign = -1 if text[0] == "-" else 1
    digits = text[1:].replace(":", "")
    h = int(digits[:2])
    m = int(digits[2:4] or 0)
    s = int(digits[4:6] or 0)
    return sign * (h * 3600 + m * 60 + s)


def parse_datetime(text: str) -> tuple[int, int, int | None, bool] | None:
    """(wall seconds since 1970, fraction in ns, offset seconds or None, has_time) of a canonical
    date/datetime text; None if it does not parse."""
    m = _DT_RE.fullmatch(text)
    if m is None:
        return None
    y, mo, d, hh, mi, ss, fr, off = m.groups()
    try:
        days = _days_from_civil(int(y), int(mo), int(d))
    except ValueError:
        return None
    h, mn, s = int(hh or 0), int(mi or 0), int(ss or 0)
    if h > 23 or mn > 59 or s > 60:
        return None
    return days * _DAY + h * 3600 + mn * 60 + s, _frac_ns(fr), _offset_seconds(off), hh is not None


def parse_clock(text: str) -> int | None:
    """Signed nanoseconds of ``[-]H:MM[:SS[.f]]``; None if it does not parse."""
    m = _CLOCK_RE.fullmatch(text)
    if m is None:
        return None
    sign, h, mi, s, fr = m.groups()
    if int(mi) > 59 or int(s or 0) > 59:
        return None
    total = (int(h) * 3600 + int(mi) * 60 + int(s or 0)) * 10**9 + _frac_ns(fr)
    return -total if sign == "-" else total


# --------------------------------------------------------------------------- rendering (read)
#: The canonical texts of a float column's non-finite values (:func:`_float_text`).
NONFINITE_TEXTS = frozenset({"NaN", "inf", "-inf"})


def _float_text(v: float, width: int) -> str:
    if v != v:
        return "NaN"
    if v in (float("inf"), float("-inf")):
        return "inf" if v > 0 else "-inf"
    typ = {16: np.float16, 32: np.float32}.get(width, np.float64)
    return np.format_float_positional(typ(v), unique=True, trim="0")


def _zone_offsets(utc_secs: np.ndarray, tz: str) -> np.ndarray:
    """Offset (seconds) of zone ``tz`` at each UTC instant."""
    m = re.fullmatch(r"([+-])(\d{2}):?(\d{2})", tz)
    if m:
        off = (int(m.group(2)) * 3600 + int(m.group(3)) * 60) * (-1 if m.group(1) == "-" else 1)
        return np.full(len(utc_secs), off, dtype=np.int64)
    if tz.upper() in ("UTC", "Z", "ETC/UTC", "GMT"):
        return np.zeros(len(utc_secs), dtype=np.int64)
    import pandas as pd

    utc = np.asarray(utc_secs, dtype=np.int64)
    try:
        idx = pd.DatetimeIndex(utc.astype("datetime64[s]")).tz_localize("UTC")
        wall = idx.tz_convert(tz).tz_localize(None).as_unit("s")
    except Exception as e:  # unknown zone name, out of range
        raise _CannotRender(f"time zone {tz!r}: {type(e).__name__}") from None
    return np.asarray(wall.asi8, dtype=np.int64) - utc


def _storage_ints(arr: Any) -> np.ndarray:
    """The integer storage of a temporal Arrow array (days, units since epoch/midnight)."""
    import pyarrow as pa

    storage = pa.int32() if getattr(arr.type, "bit_width", 64) == 32 else pa.int64()
    return np.asarray(arr.view(storage).to_numpy(zero_copy_only=False), dtype=np.int64)


def _render_scalar(arr: Any, kind: str, t: Any) -> tuple[list[str], dict[str, Any]]:
    """Canonical text of every value of a null-free Arrow array, plus column-level details."""
    import pyarrow as pa
    import pyarrow.compute as pc

    info: dict[str, Any] = {}
    if len(arr) == 0:
        return [], info
    if kind == "bool":
        return ["true" if v else "false" for v in arr.to_pylist()], info
    if kind == "int":
        return [str(v) for v in arr.to_pylist()], info
    if kind == "float":
        width = getattr(t, "bit_width", 64)
        texts = [_float_text(v, width) for v in arr.to_pylist()]
        specials = sorted({s for s in texts if s in NONFINITE_TEXTS})
        if specials:  # K-20: native non-finite numbers, kept as such (model.NUMERIC_SPECIALS_KEY)
            info[NUMERIC_SPECIALS_KEY] = specials
        return texts, info
    if kind == "decimal":
        return [format(v, "f") for v in arr.to_pylist()], info
    if kind == "string":
        return [str(v) for v in arr.to_pylist()], info
    if kind == "date":
        ints = _storage_ints(arr)
        if str(t) == "date64[ms]":
            ints = np.floor_divide(ints, _DAY * 1000)
        return format_dates(ints), info
    unit = _unit_of(t)
    per = UNIT_PER[unit]
    ints = _storage_ints(arr)
    if kind == "timestamp":
        secs = np.floor_divide(ints, per)
        frac = ints - secs * per
        digits = UNIT_DIGITS[unit] if np.any(frac) else 0
        info["frac_digits"] = digits
        offsets = None
        if t.tz is not None:
            offsets = _zone_offsets(secs, str(t.tz))
            secs = secs + offsets
        return format_datetimes(secs, frac, digits, offsets), info
    if kind in ("time", "duration"):
        digits = UNIT_DIGITS[unit] if np.any(np.mod(ints, per)) else 0
        info["frac_digits"] = digits
        return format_clock(ints, per, digits), info
    raise _CannotRender(kind)


def render_array(arr: Any, kind: str) -> tuple[np.ndarray, dict[str, Any]]:
    """Object array of canonical strings (None for nulls) for one Arrow array of a supported kind."""
    import pyarrow as pa
    import pyarrow.compute as pc

    n = len(arr)
    out = np.empty(n, dtype=object)
    info: dict[str, Any] = {}
    t = arr.type
    if kind == "dictionary":
        dict_kind = arrow_kind(t.value_type)
        dvals = arr.dictionary
        rendered = np.empty(len(dvals), dtype=object)
        valid_d = np.asarray(dvals.is_valid().to_numpy(zero_copy_only=False), dtype=bool)
        if valid_d.any():
            texts, dinfo = _render_scalar(pc.drop_null(dvals), dict_kind, t.value_type)
            rendered[valid_d] = texts
            info.update(dinfo)
        info["dictionary_values"] = rendered.tolist()
        info["dictionary"] = dvals
        info["value_kind"] = dict_kind
        if n:
            valid = np.asarray(arr.is_valid().to_numpy(zero_copy_only=False), dtype=bool)
            idx = np.asarray(pc.fill_null(arr.indices, 0).to_numpy(zero_copy_only=False), dtype=np.int64)
            out[valid] = rendered[idx[valid]]
        return out, info
    if n == 0 or arr.null_count == n:
        return out, info
    valid = np.asarray(arr.is_valid().to_numpy(zero_copy_only=False), dtype=bool)
    if kind == "string":
        out[:] = arr.to_pylist()
        return out, info
    try:
        enc = pc.dictionary_encode(arr)
        uniq, idx_arr = enc.dictionary, enc.indices
    except (pa.ArrowNotImplementedError, pa.ArrowInvalid):
        uniq, idx_arr = pc.drop_null(arr), None
    texts, info = _render_scalar(uniq, kind, t)
    rendered = np.empty(len(texts), dtype=object)
    rendered[:] = texts
    if idx_arr is None:
        out[valid] = rendered
    else:
        idx = np.asarray(pc.fill_null(idx_arr, 0).to_numpy(zero_copy_only=False), dtype=np.int64)
        out[valid] = rendered[idx[valid]]
    return out, info


# --------------------------------------------------------------------------- metadata policy (privacy)
#: The only field metadata copied to the output: the Parquet field id (an integer; Iceberg/Spark schemas).
FIELD_ID_KEY = b"PARQUET:field_id"
#: The only schema metadata key copied to the output (rebuilt from its structural parts).
PANDAS_KEY = b"pandas"
_FIELD_ID_RE = re.compile(rb"-?[0-9]{1,10}")
#: dtype names, versions and zone names of the pandas metadata ('datetime64[ns]', 'Int64', '3.0.6',
#: 'America/Chicago', 'timestamp[us, tz=UTC][pyarrow]'): no free text (a space only after a comma).
_TOKEN_RE = re.compile(r"[A-Za-z0-9_.+\-/:<>()\[\]=,]+(?:, [A-Za-z0-9_.+\-/:<>()\[\]=,]+)*")
_TOKEN_MAX = 120
#: Keys of a pandas column entry's ``metadata`` that describe its dtype (categorical, tz, decimal, encoding).
_PANDAS_ENTRY_META = ("encoding", "timezone", "num_categories", "ordered", "precision", "scale", "unit")


def _token(v: Any) -> bool:
    return isinstance(v, str) and len(v) <= _TOKEN_MAX and _TOKEN_RE.fullmatch(v) is not None


class _NotStructural(Exception):
    """The pandas metadata has a part that is not a known structural form (the key is dropped whole)."""


def _pandas_metadata(raw: bytes, names: list[str]) -> tuple[bytes | None, int]:
    """The pandas schema metadata rebuilt from its structural parts, and how many free-form parts were left out.

    Kept: ``index_columns`` (column names and range descriptions), ``column_indexes``,
    ``columns`` (entries of the columns in ``names``), ``creator`` and ``pandas_version``.
    Left out: ``attributes`` (``DataFrame.attrs``) and any other key, and an index or
    column-level name that is not a column name (written as null). A dtype that is not
    a plain token drops the whole key (returns None). The original bytes are returned
    when nothing changed.
    """
    try:
        meta = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        return None, 1
    if not isinstance(meta, dict):
        return None, 1
    fields = set(names)
    lost = 0

    def label(v: Any) -> Any:  # an index / column-level name: kept when it is a column name (in the file anyway)
        nonlocal lost
        if v is None or (isinstance(v, str) and v in fields):
            return v
        lost += 1
        return None

    def entry_meta(m: Any) -> Any:
        nonlocal lost
        if m is None:
            return None
        if not isinstance(m, dict):
            lost += 1
            return None
        out = {}
        for k, v in m.items():
            ok = k in _PANDAS_ENTRY_META and (v is None or isinstance(v, (bool, int)) or _token(v))
            if ok:
                out[k] = v
            else:
                lost += 1
        return out

    def entry(e: Any, column: bool) -> dict[str, Any] | None:
        if not isinstance(e, dict):
            raise _NotStructural
        for k in ("pandas_type", "numpy_type"):
            if e.get(k) is not None and not _token(e.get(k)):
                raise _NotStructural
        out: dict[str, Any] = {}
        for k, v in e.items():
            if k in ("name", "field_name"):
                out[k] = label(v) if not (column and k == "field_name") else v
            elif k in ("pandas_type", "numpy_type"):
                out[k] = v
            elif k == "metadata":
                out[k] = entry_meta(v)
            else:
                raise _NotStructural
        return out

    clean: dict[str, Any] = {}
    try:
        for key, value in meta.items():
            if key == "columns" and isinstance(value, list):
                kept = []
                for e in value:
                    if not isinstance(e, dict) or e.get("field_name") not in fields:
                        if not isinstance(e, dict):
                            raise _NotStructural
                        continue  # a column not in the file (--exclude): its entry goes too
                    kept.append(entry(e, True))
                clean[key] = kept
            elif key == "column_indexes" and isinstance(value, list):
                clean[key] = [entry(e, False) for e in value]
            elif key == "index_columns" and isinstance(value, list):
                idx: list[Any] = []
                for e in value:
                    if isinstance(e, str):
                        if e in fields:
                            idx.append(e)
                    elif (isinstance(e, dict) and e.get("kind") == "range"
                          and all(isinstance(e.get(k), int) for k in ("start", "stop", "step"))
                          and set(e) <= {"kind", "name", "start", "stop", "step"}):
                        idx.append({k: (label(v) if k == "name" else v) for k, v in e.items()})
                    else:
                        raise _NotStructural
                clean[key] = idx
            elif key == "creator" and isinstance(value, dict) and set(value) <= {"library", "version"} \
                    and all(_token(v) for v in value.values()):
                clean[key] = value
            elif key == "pandas_version" and _token(value):
                clean[key] = value
            elif value not in (None, {}, []):  # 'attributes' (DataFrame.attrs) and anything unknown
                lost += 1
    except _NotStructural:
        return None, lost + 1
    if clean == meta:
        return raw, lost
    return json.dumps(clean).encode("utf-8"), lost


def safe_schema_metadata(md: dict[bytes, bytes] | None, names: list[str]) -> tuple[dict[bytes, bytes] | None, int]:
    """(schema metadata copied to an output whose columns are ``names``, number of entries/parts left out)."""
    if not md:
        return None, 0
    keep: dict[bytes, bytes] = {}
    lost = 0
    for k, v in md.items():
        if k == PANDAS_KEY:
            clean, n = _pandas_metadata(v, names)
            lost += n
            if clean is not None:
                keep[k] = clean
        else:
            lost += 1
    return (keep or None), lost


def safe_field_metadata(md: dict[bytes, bytes] | None) -> tuple[dict[bytes, bytes] | None, int]:
    """(field metadata copied to the output: the Parquet field id only, number of entries left out)."""
    if not md:
        return None, 0
    keep = {k: v for k, v in md.items() if k == FIELD_ID_KEY and _FIELD_ID_RE.fullmatch(v)}
    return (keep or None), len(md) - len(keep)


def _metadata_strings(raw: Any) -> list[str]:
    """The text of one metadata key or value: the string leaves (and object keys) of a JSON value, else the decoded
    text."""
    s = raw.decode("utf-8", "replace") if isinstance(raw, (bytes, bytearray)) else str(raw)
    try:
        obj = json.loads(s)
    except ValueError:
        return [s]
    out: list[str] = []

    def walk(x: Any) -> None:
        if isinstance(x, str):
            out.append(x)
        elif isinstance(x, dict):
            for k, v in x.items():
                out.append(str(k))
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(obj)
    return out


def dropped_metadata_texts(schema: Any) -> list[str]:
    """R8-04 audit: the free-form text of a source schema's metadata that the writers leave out
    (:func:`safe_schema_metadata`, :func:`safe_field_metadata`): every key and value but the structural pandas parts
    and Parquet field ids, of the schema and of every field (nested ones too). Real text no output holds (PRIVATE):
    the run's real-text registry (``privacy.texts``) keeps generators from writing it."""
    import pyarrow.types as pt

    out: list[str] = []
    md = schema.metadata or {}
    clean, _ = safe_schema_metadata(md, list(schema.names))
    kept = set(_metadata_strings(clean[PANDAS_KEY])) if clean and PANDAS_KEY in clean else set()
    for k, v in md.items():
        out.extend(x for x in ([] if k == PANDAS_KEY else _metadata_strings(k)) + _metadata_strings(v)
                   if x not in kept)

    def fields(t: Any, items: list[Any]) -> None:
        for f in items:
            fmd = f.metadata or {}
            safe, _ = safe_field_metadata(fmd)
            for k, v in fmd.items():
                if not (safe and k in safe):
                    out.extend(_metadata_strings(k) + _metadata_strings(v))
            ft = f.type
            if pt.is_struct(ft):
                fields(ft, [ft.field(i) for i in range(ft.num_fields)])
            elif pt.is_list(ft) or pt.is_large_list(ft) or pt.is_fixed_size_list(ft):
                fields(ft, [ft.value_field])

    fields(None, list(schema))
    return out


def _with_metadata(field: Any, md: dict[bytes, bytes] | None) -> Any:
    return field.with_metadata(md) if md else field.remove_metadata()


def metadata_message(schema: Any) -> Message | None:
    """The ``arrow_metadata_dropped`` warning for a source schema (counts only), or None."""
    _, n_schema = safe_schema_metadata(schema.metadata, list(schema.names))
    n_field = sum(safe_field_metadata(f.metadata)[1] for f in schema)
    if not (n_schema or n_field):
        return None
    parts = []
    if n_schema:
        parts.append(f"{n_schema} schema (file) metadata item(s)")
    if n_field:
        parts.append(f"{n_field} column (field) metadata item(s)")
    return Message("warning", "arrow_metadata_dropped",
                   "free-form Arrow metadata is not copied to the synthetic file (it can quote real data): "
                   + " and ".join(parts) + " left out; structural metadata (the pandas column/index layout, "
                   "Parquet field ids) is kept")


# --------------------------------------------------------------------------- reading
def type_text(t: Any) -> str:
    """An Arrow type as printed in messages and the layout, without the names of its child fields: a struct
    field can be named after a person (D34), so ``struct<Okonkwo, Chinedu: int64>`` is ``struct<1 field>``."""
    import pyarrow.types as pt

    if not _named_children(t):
        return str(t)
    try:
        if pt.is_struct(t):
            return f"struct<{t.num_fields} field{'' if t.num_fields == 1 else 's'}>"
        views = [getattr(pt, name, None) for name in ("is_list_view", "is_large_list_view")]
        if pt.is_list(t) or pt.is_large_list(t) or pt.is_fixed_size_list(t) or any(
                v is not None and v(t) for v in views):
            kind = str(t).split("<", 1)[0]
            return f"{kind}<{type_text(t.value_type)}>"
        if pt.is_map(t):
            return f"map<{type_text(t.key_type)}, {type_text(t.item_type)}>"
        if pt.is_union(t):
            return f"union<{t.num_fields} member(s)>"
        if pt.is_dictionary(t):
            return f"dictionary<values={type_text(t.value_type)}, indices={t.index_type}>"
    except (AttributeError, TypeError):  # pragma: no cover - an Arrow build without one of the predicates
        pass
    return str(t)


def _named_children(t: Any) -> bool:
    """The type holds a struct or union (children with free names) at any depth."""
    import pyarrow.types as pt

    if pt.is_struct(t) or pt.is_union(t):
        return True
    for attr in ("value_type", "key_type", "item_type"):
        child = getattr(t, attr, None)
        if child is not None and child is not t and _named_children(child):
            return True
    return False


def check_unsupported(req: "ReadRequest", table: str | None, unsupported: list[tuple[str, str]]) -> None:
    """FAIL (before writing) when a binary/nested column holds values and is neither --null nor --exclude."""
    bad = []
    for name, desc in unsupported:
        opts = req.settings.column_options(req.rel_path, table, name)
        if not (opts.null or opts.exclude):
            bad.append((name, desc))
    if not bad:
        return
    cols = ", ".join(f"{name!r} ({desc})" for name, desc in bad)
    fix = ("; ".join(f"--null {name} or --exclude {name}" for name, _ in bad)
           + " (--null keeps the column and its type with every value missing; --exclude drops the column)")
    raise UnsupportedColumnError(
        f"{len(bad)} column(s) cannot be generated (binary, nested and other unsupported values are never copied "
        f"or stringified): {cols}", fix=fix)


def table_result(table: Any, req: "ReadRequest", fmt: str, extra: dict[str, Any],
                 messages: list[Message] | None = None) -> ReadResult:
    """ReadResult of one Arrow table (Parquet, Feather, SAS input)."""
    import pyarrow as pa
    import pyarrow.types as pt

    if any(pt.is_dictionary(f.type) for f in table.schema):
        table = table.unify_dictionaries()
    columns: list[RawColumn] = []
    unsupported: list[tuple[str, str]] = []
    for i, field in enumerate(table.schema):
        arr = table.column(i).combine_chunks()
        kind = arrow_kind(field.type)
        native: dict[str, Any] = {"family": "arrow", "kind": kind, "arrow_type": type_text(field.type),
                                  "field": field}
        values = np.empty(len(arr), dtype=object)
        desc = type_text(field.type)
        if kind != "unsupported":
            try:
                values, info = render_array(arr, kind)
                native.update(info)
            except (_CannotRender, OverflowError, ValueError, pa.ArrowException) as e:
                native["kind"] = kind = "unsupported"
                desc += f"; values cannot be rendered ({type(e).__name__})"
        if kind == "unsupported":
            values = np.empty(len(arr), dtype=object)  # never stringified or copied
            if arr.null_count < len(arr):
                unsupported.append((field.name, desc))
        columns.append(RawColumn(field.name, values, native))
    check_unsupported(req, None, unsupported)
    extra = dict(extra)
    extra[SCHEMA_KEY] = table.schema  # private: the writer keeps only its structural metadata
    extra.setdefault("arrow_types", [type_text(f.type) for f in table.schema])
    layout = FileLayout(format=fmt, extra=extra)
    raw = RawTable(None, list(table.schema.names), columns, table.num_rows)
    messages = list(messages or [])
    if req.purpose == "source":
        dropped = metadata_message(table.schema)
        if dropped is not None:
            messages.append(dropped)
    return ReadResult(layout, [raw], messages)


def _n_leaves(t: Any) -> int:
    import pyarrow.types as pt

    if pt.is_struct(t):
        return sum(_n_leaves(t.field(i).type) for i in range(t.num_fields)) or 1
    if pt.is_map(t):
        return _n_leaves(t.key_type) + _n_leaves(t.item_type)
    if pt.is_list(t) or pt.is_large_list(t) or pt.is_fixed_size_list(t) or (
            hasattr(pt, "is_list_view") and (pt.is_list_view(t) or pt.is_large_list_view(t))):
        return _n_leaves(t.value_type)
    if pt.is_dictionary(t):
        return _n_leaves(t.value_type)
    return 1


def _hive_folder(req: "ReadRequest") -> str | None:
    if req.purpose != "source":
        return None
    for part in PurePosixPath(req.rel_path).parts[:-1]:
        if _HIVE_RE.match(part):
            return part
    parent = Path(req.path).parent.name
    return parent if _HIVE_RE.match(parent) else None


def _parquet_details(pf: Any, schema: Any) -> tuple[dict[str, Any], list[Message]]:
    md = pf.metadata
    messages: list[Message] = []
    raw_kv = md.metadata or {}
    n_fields = len(schema)
    codecs: list[str | None] = [None] * n_fields
    use_dict: list[bool] = [True] * n_fields
    stats: list[bool] = [True] * n_fields
    rg_rows = [md.row_group(i).num_rows for i in range(md.num_row_groups)]
    if md.num_row_groups:
        rg = md.row_group(0)
        leaf = 0
        for i, f in enumerate(schema):
            if leaf >= rg.num_columns:
                break
            cc = rg.column(leaf)
            codecs[i] = str(cc.compression).upper()
            use_dict[i] = any("DICTIONARY" in str(e) for e in (cc.encodings or ()))
            stats[i] = bool(cc.is_stats_set)
            leaf += _n_leaves(f.type)
    known = [c for c in codecs if c]
    for c in sorted(set(known)):
        if c not in _PARQUET_CODEC_OUT:
            messages.append(Message("warning", "parquet_codec",
                                    f"the source uses the {c} codec, which pyarrow cannot write; the output uses "
                                    "SNAPPY"))
    uniform = sorted(set(known))
    try:  # legacy INT96 timestamps (Spark, Hive, Impala): read as timestamp[ns], written back as INT96
        pschema = pf.schema
        int96 = any(pschema.column(j).physical_type == "INT96" for j in range(len(pschema)))
    except Exception:  # noqa: BLE001 - a schema pyarrow cannot describe: no INT96 mirror
        int96 = False
    extra: dict[str, Any] = {
        "parquet_version": str(md.format_version),
        "codec": (uniform[0].lower() if len(uniform) == 1 else ("mixed" if uniform else None)),
        "row_groups": md.num_row_groups,
        "row_group_rows": max(rg_rows) if rg_rows else None,
        "row_group_bytes": md.row_group(0).total_byte_size if md.num_row_groups else None,
        "arrow_schema_stored": b"ARROW:schema" in raw_kv,
        "int96_timestamps": int96,
        "_codecs": codecs,
        "_use_dictionary": use_dict,
        "_write_statistics": stats,
    }
    return extra, messages


def read_parquet(req: "ReadRequest") -> ReadResult:
    """Read a Parquet file keeping its Arrow schema, codecs and row-group size."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    hive = _hive_folder(req)
    if hive is not None:
        raise ReadError(
            f"this file is one partition of a hive-partitioned Parquet dataset (folder {hive!r}); partitioned "
            "folders are not supported in v1 (each .parquet file must be a whole table)",
            fix="combine the partitions into one .parquet file (e.g. pandas: pd.read_parquet(FOLDER)"
                ".to_parquet('table.parquet')) and run datanonymous on that file")
    data = Path(req.path).read_bytes()
    try:
        pf = pq.ParquetFile(pa.BufferReader(data))
        table = pf.read()
    except (pa.ArrowException, OSError, ValueError) as e:
        raise ReadError(f"not a readable Parquet file ({type(e).__name__})",
                        fix="check the file (it may be truncated, encrypted or not Parquet)") from None
    extra, messages = _parquet_details(pf, table.schema)
    return table_result(table, req, "parquet", extra, messages)


def _u32(b: bytes, o: int) -> int:
    return struct.unpack_from("<I", b, o)[0]


def _fb_field(b: bytes, table: int, index: int) -> int | None:
    """Absolute position of field ``index`` of the flatbuffer table at ``table`` (None if absent)."""
    vtable = table - struct.unpack_from("<i", b, table)[0]
    vt_size = struct.unpack_from("<H", b, vtable)[0]
    slot = 4 + 2 * index
    if slot + 2 > vt_size:
        return None
    off = struct.unpack_from("<H", b, vtable + slot)[0]
    return table + off if off else None


def _ipc_codec(data: bytes) -> str:
    """Body codec of the first record batch of an Arrow IPC file ('lz4', 'zstd' or 'uncompressed')."""
    import pyarrow as pa

    try:
        reader = pa.ipc.MessageReader.open_stream(pa.BufferReader(data[8:]))
        while True:
            msg = reader.read_next_message()
            if msg is None:
                return "uncompressed"
            if msg.type != "record batch":
                continue
            b = msg.metadata.to_pybytes()
            header = _fb_field(b, _u32(b, 0), 2)  # Message.header
            if header is None:
                return "uncompressed"
            rb = header + _u32(b, header)
            comp = _fb_field(b, rb, 3)  # RecordBatch.compression
            if comp is None:
                return "uncompressed"
            ctab = comp + _u32(b, comp)
            codec = _fb_field(b, ctab, 0)  # BodyCompression.codec (default LZ4_FRAME)
            return {0: "lz4", 1: "zstd"}.get(b[codec] if codec is not None else 0, "uncompressed")
    except (StopIteration, pa.ArrowException, OSError, struct.error, IndexError):
        return "uncompressed"


def read_feather(req: "ReadRequest") -> ReadResult:
    """Read a Feather (V1) / Arrow IPC (V2) file keeping its schema, codec and batch size."""
    import pyarrow as pa
    import pyarrow.feather as feather

    data = Path(req.path).read_bytes()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)  # Feather V1
            table = feather.read_table(pa.BufferReader(data), memory_map=False)
    except (pa.ArrowException, OSError, ValueError) as e:
        raise ReadError(f"not a readable Feather/Arrow file ({type(e).__name__})",
                        fix="check the file (it may be truncated or not Feather)") from None
    extra: dict[str, Any] = {"feather_version": 1 if data[:4] == b"FEA1" else 2}
    if extra["feather_version"] == 2:
        extra["codec"] = _ipc_codec(data)
        try:
            r = pa.ipc.open_file(pa.BufferReader(data))
            rows = [r.get_batch(i).num_rows for i in range(r.num_record_batches)]
        except pa.ArrowException:
            rows = []
        extra["record_batches"] = len(rows)
        extra["batch_rows"] = max(rows) if rows else None
    else:
        extra["codec"] = "uncompressed"
    return table_result(table, req, "feather", extra)


# --------------------------------------------------------------------------- writing
def value_fix(name: str, profile: Any = None) -> str:
    """The fix for a synthetic value that the column's storage type cannot hold (never quotes a value).

    Mentions ``--force-type`` only when the column has one; an ID column whose synthetic IDs do
    not fit the storage type (too few non-real IDs of the real shape, D9) says so.
    """
    if profile is not None and getattr(profile, "source", None) == "force-type":
        return f"remove --force-type for {name!r}, or --null {name} / --exclude {name}"
    if profile is not None and getattr(profile, "type", None) is ColumnType.IDENTIFIER:
        return (f"--exclude {name} or --null {name} (the synthetic IDs, which never reuse a real ID (D9), do not "
                "fit the column's storage type)")
    return f"--null {name} or --exclude {name}"


def _type_error(name: str, t: Any, profile: Any = None) -> WriteError:
    what = f"synthetic {profile.type.value} value" if getattr(profile, "type", None) is not None else "synthetic value"
    return WriteError(f"column {name!r}: a {what} cannot be stored as Arrow {t}", fix=value_fix(name, profile))


def _cell_list(values: np.ndarray, kind: str) -> list[Any]:
    """Python list with ``None`` for missing; '' is missing except in string columns."""
    out = values.tolist() if isinstance(values, np.ndarray) else list(values)
    if kind == "string":
        return out
    return [None if (v is None or (isinstance(v, str) and v.strip() == "")) else v for v in out]


def _parse_temporal(vals: list[Any], kind: str, t: Any, name: str, profile: Any = None) -> list[int | None]:
    """Integers in the Arrow storage unit for canonical temporal texts."""
    cache: dict[str, int] = {}
    unit = _unit_of(t)
    per = UNIT_PER.get(unit, 1)
    tz = getattr(t, "tz", None)
    needs_zone: list[str] = []  # wall times without an offset in a zoned column
    for v in vals:
        if v is None:
            continue
        s = str(v)
        if s in cache:
            continue
        if kind == "date":
            r = parse_datetime(s)
            if r is None or (r[3] and (r[0] % _DAY or r[1])):
                raise _type_error(name, t, profile)
            days = r[0] // _DAY
            val = days * _DAY * 1000 if str(t) == "date64[ms]" else days
        elif kind == "timestamp":
            r = parse_datetime(s)
            if r is None:
                raise _type_error(name, t, profile)
            wall, frac_ns, off, _ = r
            if tz is not None and off is None:
                needs_zone.append(s)
            secs = wall - off if (tz is not None and off is not None) else wall
            val = secs * per + frac_ns * per // 10**9
        else:  # time / duration
            ns = parse_clock(s)
            if ns is None or (kind == "time" and not 0 <= ns < _DAY * 10**9):
                raise _type_error(name, t, profile)
            val = ns * per // 10**9 if ns >= 0 else -((-ns) * per // 10**9)
        cache[s] = val
    if needs_zone:  # localise those wall times in the column's zone
        import pandas as pd

        walls = np.array([cache[s] for s in needs_zone], dtype=np.int64)
        local = pd.DatetimeIndex(walls.astype(f"datetime64[{unit}]")).tz_localize(
            str(tz), ambiguous=True, nonexistent="shift_forward")
        utc = local.tz_convert("UTC").tz_localize(None).as_unit(unit).asi8
        for s, u in zip(needs_zone, utc.tolist()):
            cache[s] = int(u)
    return [None if v is None else cache[str(v)] for v in vals]


def values_to_arrow(values: np.ndarray, field: Any, native: dict[str, Any] | None, profile: Any = None) -> Any:
    """Convert one rendered output column back to the source field's Arrow type."""
    import pyarrow as pa

    t = field.type
    name = field.name
    kind = arrow_kind(t)
    n = len(values)
    if kind == "dictionary":
        return _dictionary_array(values, field, native or {}, profile)
    vals = _cell_list(values, kind)
    if all(v is None for v in vals):
        return pa.nulls(n, type=t)
    if kind in ("unsupported", "null"):
        raise WriteError(f"column {name!r}: values present in a column of Arrow type {t}",
                         fix=f"--null {name} or --exclude {name}")
    try:
        if kind == "string":
            return pa.array([None if v is None else str(v) for v in vals], type=pa.string()).cast(t)
        if kind == "bool":
            conv = []
            for v in vals:
                if v is None:
                    conv.append(None)
                    continue
                s = str(v).strip().casefold()
                if s not in ("true", "false"):
                    raise _type_error(name, t, profile)
                conv.append(s == "true")
            return pa.array(conv, type=t)
        if kind == "int":
            return pa.array([None if v is None else int(str(v).strip()) for v in vals], type=t)
        if kind == "float":
            return pa.array([None if v is None else float(str(v).strip()) for v in vals], type=pa.float64()).cast(
                t, safe=False)
        if kind == "decimal":
            # a context wide enough for decimal256 (the default 28 digits cannot quantize decimal(38, s) values)
            q = Decimal(1).scaleb(-t.scale, _DEC_CONTEXT)
            return pa.array([None if v is None else Decimal(str(v).strip()).quantize(
                q, rounding=ROUND_HALF_EVEN, context=_DEC_CONTEXT) for v in vals], type=t)
        if kind in TEMPORAL_KINDS:
            ints = _parse_temporal(vals, kind, t, name, profile)
            storage = pa.int32() if (kind == "date" and str(t) == "date32[day]") or (
                kind == "time" and str(t).startswith("time32")) else pa.int64()
            return pa.array(ints, type=storage).cast(t)
    except WriteError:
        raise
    except (ValueError, TypeError, InvalidOperation, OverflowError, pa.ArrowException):
        raise _type_error(name, t, profile) from None
    raise _type_error(name, t, profile)


def vocabulary_label(text: Any) -> bool:
    """True for a label that is a word of the tool's vocabularies (``detect.vocab.is_vocabulary``, D35): the
    standard education vocabulary (race/ethnicity, gender, grade levels, programs, statuses ...), a level of an
    ordinal scale (Likert, performance levels, class standing ...), a boolean word, an NA or suppression token,
    a school year, or blank.

    Such a label quotes nobody, so a dictionary entry or Stata value label that **no row holds** is still
    written when it is one (a grade dictionary keeps ``06`` in a school without 6th graders); any other
    unused label was never checked (it can quote anything, e.g. an email) and is never written (R2-03).
    """
    from ..detect.vocab import is_vocabulary

    return is_vocabulary(text)


def _level_map(profile: Any) -> dict[str, str]:
    """Real -> written value of a column's levels the D35 allowlist did not keep (``generate.pseudonyms``)."""
    from ..detect.text import PSEUDO_MAP_KEY

    model = getattr(profile, "model", None)
    extra = getattr(model, "extra", None) or {}
    return extra.get(PSEUDO_MAP_KEY) or {}


def _dictionary_array(values: np.ndarray, field: Any, native: dict[str, Any], profile: Any) -> Any:
    import pyarrow as pa

    t = field.type
    if profile is not None and getattr(profile, "role", "") == "null":
        # --null: the same dictionary type with every value null and an EMPTY dictionary. The source
        # dictionary is never copied: the column was not detected, so its labels are not known to be
        # label-safe (they may be names), and the dictionary's contents are not part of the Arrow type.
        return pa.DictionaryArray.from_arrays(pa.nulls(len(values), t.index_type), pa.array([], t.value_type),
                                              ordered=t.ordered)
    vkind = arrow_kind(t.value_type)
    vals = _cell_list(values, vkind)
    non_null = [str(v) for v in vals if v is not None]
    src_dict = native.get("dictionary")
    rendered = native.get("dictionary_values")
    label_safe = profile is None or profile.type in LABEL_SAFE_TYPES
    names = _level_map(profile)
    if names and src_dict is not None and rendered is not None:
        # D35: an entry the allowlist did not keep is written as its pseudonym, in the entry's place (an ordered
        # dictionary keeps its order); the real entry is never written
        rendered = [names.get(s, s) if s is not None else None for s in rendered]
        try:
            src_dict = pa.array(rendered, type=src_dict.type)
        except (pa.ArrowException, TypeError, ValueError):
            src_dict = None
    if label_safe and src_dict is not None and rendered is not None and src_dict.type == t.value_type:
        pos: dict[str, int] = {}
        for i, s in enumerate(rendered):
            if s is not None and s not in pos:
                pos[s] = i
        if all(v in pos for v in non_null):
            # the source entries the synthetic rows hold, in the source order (ordered dictionaries keep their
            # order); an entry no row holds was never checked, so it is written only when it is a word of the
            # tool's own vocabularies (a grade level no row holds, R2-03)
            keep = sorted({pos[v] for v in non_null} | {i for s, i in pos.items() if vocabulary_label(s)})
            new_pos = {old: k for k, old in enumerate(keep)}
            idx = [None if v is None else new_pos[pos[str(v)]] for v in vals]
            try:
                return pa.DictionaryArray.from_arrays(pa.array(idx, type=t.index_type),
                                                      src_dict.take(pa.array(keep, type=pa.int64())),
                                                      ordered=t.ordered)
            except pa.ArrowException:
                raise _type_error(field.name, t, profile) from None
    # rebuilt on the synthetic values: real labels of non-label types never enter the dictionary
    uniq = sorted(set(non_null))
    dvals = np.empty(len(uniq), dtype=object)
    dvals[:] = uniq
    try:
        dict_arr = values_to_arrow(dvals, pa.field(field.name, t.value_type), {"kind": vkind}, None)
    except WriteError:
        raise _type_error(field.name, t, profile) from None
    pos2 = {s: i for i, s in enumerate(uniq)}
    idx2 = [None if v is None else pos2[str(v)] for v in vals]
    try:
        return pa.DictionaryArray.from_arrays(pa.array(idx2, type=t.index_type), dict_arr, ordered=t.ordered)
    except pa.ArrowException:
        raise _type_error(field.name, t, profile) from None


def build_table(req: "WriteRequest") -> tuple[Any, list[int]]:
    """The output Arrow table on the source fields (structural metadata only), and each output column's source
    index."""
    import pyarrow as pa

    if len(req.tables) != 1:
        raise WriteError("the Arrow writer needs exactly one table", fix="report this as a bug")
    schema = req.layout.extra.get(SCHEMA_KEY)
    if schema is None:
        raise WriteError("the source Arrow schema is missing", fix="report this as a bug")
    t = req.tables[0]
    fields = []
    arrays = []
    for k, si in enumerate(t.source_index):
        field = schema.field(si)
        prof = t.profile.columns[si] if t.profile is not None else None
        native = (prof.native if prof is not None else None) or {}
        arr = values_to_arrow(t.columns[k], field, native, prof)
        if not field.nullable and arr.null_count:
            raise WriteError(f"column {field.name!r} is NOT NULL in the source schema but has missing values",
                             fix=f"--exclude {field.name} (a NOT NULL column cannot be written all-missing)")
        fields.append(_with_metadata(field, safe_field_metadata(field.metadata)[0]))
        arrays.append(arr)
    # free-form metadata (descriptions, owners, DataFrame.attrs ...) is never copied: structural parts only
    md = safe_schema_metadata(schema.metadata, [f.name for f in fields])[0]
    sizes = {f.name: len(a.dictionary) for f, a in zip(fields, arrays) if isinstance(a, pa.DictionaryArray)}
    out_schema = pa.schema(fields, metadata=_with_category_counts(md, sizes))
    table = pa.Table.from_arrays(arrays, schema=out_schema)
    return table.cast(out_schema), list(t.source_index)


def _with_category_counts(md: dict[bytes, bytes] | None, sizes: dict[str, int]) -> dict[bytes, bytes] | None:
    """The pandas metadata's ``num_categories`` set to the written dictionary sizes (unused source entries are
    not written, so the source count would describe categories the file does not hold)."""
    if not md or PANDAS_KEY not in md or not sizes:
        return md
    try:
        meta = json.loads(md[PANDAS_KEY])
    except (ValueError, UnicodeDecodeError):
        return md
    changed = False
    for e in meta.get("columns") or []:
        m = e.get("metadata") if isinstance(e, dict) else None
        n = sizes.get(e.get("field_name")) if isinstance(e, dict) else None
        if isinstance(m, dict) and "num_categories" in m and n is not None and m["num_categories"] != n:
            m["num_categories"] = n
            changed = True
    if not changed:
        return md
    return {**md, PANDAS_KEY: json.dumps(meta).encode("utf-8")}


def _comparable_metadata(md: dict[bytes, bytes] | None) -> Any:
    """Schema metadata for the self-check comparison: the pandas part parsed, without ``num_categories``
    (rebuilt from the written dictionaries by :func:`_with_category_counts`)."""
    if not md:
        return {}
    out: dict[Any, Any] = dict(md)
    if PANDAS_KEY in md:
        try:
            meta = json.loads(md[PANDAS_KEY])
        except (ValueError, UnicodeDecodeError):
            return out
        columns = meta.get("columns") if isinstance(meta, dict) else None
        for e in columns if isinstance(columns, list) else []:
            if isinstance(e, dict) and isinstance(e.get("metadata"), dict):
                e["metadata"].pop("num_categories", None)
        out[PANDAS_KEY] = meta
    return out


def _per_column(values: list[Any], index: list[int], default: Any) -> list[Any]:
    return [values[i] if i < len(values) and values[i] is not None else default for i in index]


def write_parquet(req: "WriteRequest") -> None:
    """Write Parquet with the source schema, codec(s), row-group size and writer settings."""
    import pyarrow.parquet as pq

    table, index = build_table(req)
    ex = req.layout.extra
    names = table.schema.names
    unique_names = len(set(names)) == len(names)
    codecs = [_PARQUET_CODEC_OUT.get(str(c).upper(), "snappy") for c in
              _per_column(ex.get("_codecs") or [], index, "SNAPPY")]
    compression: Any = "snappy" if not codecs else codecs[0] if len(set(codecs)) == 1 else (
        dict(zip(names, codecs)) if unique_names else codecs[0])
    use_dict = _per_column(ex.get("_use_dictionary") or [], index, True)
    use_dictionary: Any = (use_dict[0] if len(set(use_dict)) == 1 else
                           ([n for n, u in zip(names, use_dict) if u] if unique_names else True)) if use_dict else True
    stats = _per_column(ex.get("_write_statistics") or [], index, True)
    write_statistics: Any = (stats[0] if len(set(stats)) == 1 else
                             ([n for n, s in zip(names, stats) if s] if unique_names else True)) if stats else True
    version = str(ex.get("parquet_version") or "2.6")
    if version not in ("1.0", "2.4", "2.6"):
        version = "2.6"
    try:
        pq.write_table(table, str(req.path), compression=compression, use_dictionary=use_dictionary,
                       write_statistics=write_statistics, version=version,
                       row_group_size=ex.get("row_group_rows") or None,
                       store_schema=bool(ex.get("arrow_schema_stored", True)),
                       use_deprecated_int96_timestamps=bool(ex.get("int96_timestamps", False)))
    except Exception as e:  # pyarrow raises several error types
        raise WriteError(f"Parquet writer error ({type(e).__name__})", fix="report this as a bug") from None


def write_feather(req: "WriteRequest") -> None:
    """Write Feather with the source schema, IPC version, codec and record-batch size."""
    import pyarrow.feather as feather

    table, _ = build_table(req)
    ex = req.layout.extra
    version = int(ex.get("feather_version") or 2)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)  # Feather V1 is deprecated but still mirrored
            if version == 1:
                feather.write_feather(table, str(req.path), version=1)
            else:
                feather.write_feather(table, str(req.path), compression=str(ex.get("codec") or "uncompressed"),
                                      version=2, chunksize=ex.get("batch_rows") or None)
    except Exception as e:  # pyarrow raises several error types
        raise WriteError(f"Feather writer error ({type(e).__name__})", fix="report this as a bug") from None


# --------------------------------------------------------------------------- detection
def native_date_detection(ctx: Any, target: ColumnType, what: str) -> Any:
    """Ask AREA A's ``detect_dates`` for a natively typed date/time column (``ctx.forced_type = target``).

    The values are the canonical texts of this module. Raises
    :class:`UnsupportedColumnError` (fix ``--null`` / ``--exclude``) when no
    date-family detection is returned (the file FAILs instead of writing text
    that the typed column cannot hold).
    """
    from ..detect import dates

    prev = ctx.forced_type
    ctx.forced_type = target
    try:
        det = dates.detect_dates(ctx)
    finally:
        ctx.forced_type = prev
    if det is None or TYPE_INFO[det.type].family != "date":
        raise UnsupportedColumnError(
            f"column {ctx.name!r}: {what} values cannot be generated in this build (no date/time generator "
            f"accepted the column as {target.value})",
            fix=f"--null {ctx.name} (keeps the column and its type, every value missing) or --exclude {ctx.name}")
    det.source = "native"
    det.params.setdefault("native_type", what)
    det.reason = f"native {what}; " + (det.reason or det.type.value)
    return det


_TEMPORAL_TARGET = {"date": ColumnType.DATE, "time": ColumnType.TIME, "duration": ColumnType.DURATION}


def _dictionary_detection(ctx: Any, native: dict[str, Any]) -> Any:
    from ..detect.context import Detection

    if not ctx.text_values:
        return None
    t = native.get("field").type if native.get("field") is not None else None
    from ..detect.text import GUARDED_PII_TEXT

    entries = [v for v in (native.get("dictionary_values") or []) if v is not None]
    # the levels some row holds, plus unused tool-vocabulary levels (grade levels ...): any other entry no row
    # holds was never checked (its text can be anything, e.g. an email) and is never written
    # (``_dictionary_array``), nor quoted in the params, docs or report (R2-03)
    levels = list(dict.fromkeys(v for v in entries if v in ctx.raw_counts or vocabulary_label(v)))
    unused = len({v for v in entries if v not in ctx.raw_counts})
    dropped = len(set(entries) - set(levels))
    # an ordered dictionary under a PII/free-text header (ordered names) is offered as FACTOR so the
    # D12 guard (which never rewrites ORDINAL vocabularies) replaces its real labels
    pii_header = ctx.hints.pii in GUARDED_PII_TEXT or ctx.hints.free_text is not None
    if t is not None and t.ordered and not pii_header:
        det = Detection(ColumnType.ORDINAL, params={
            "vocabulary": None, "levels": levels, "order_source": "Arrow ordered dictionary",
            "dictionary_size": len(entries), "unused_levels": unused},
            source="native", reason=f"Arrow ordered dictionary ({len(levels)} levels): real levels kept in order")
    else:
        variants: dict[str, set[str]] = {}
        for raw in ctx.raw_counts:
            if raw is None or raw.strip() == "":
                continue
            variants.setdefault(raw.strip().casefold(), set()).add(raw)
        det = Detection(ColumnType.FACTOR, params={
            "n_levels": len(variants), "variant_groups": sum(1 for v in variants.values() if len(v) > 1),
            "dictionary_size": len(entries), "unused_levels": unused},
            source="native", reason=f"Arrow dictionary column ({len(variants)} levels used): real levels kept")
    if dropped:  # never written: an entry no row holds was never checked (it can quote anything)
        det.messages.append(Message(
            "info", "dictionary_unused_levels",
            f"{dropped} dictionary level(s) are held by no row; they are not written (a label that no row holds "
            "is never checked, so only tool-vocabulary labels such as grade levels are kept unused)",
            column=ctx.name))
    return det


def detect_native(ctx: Any) -> Any:
    """Type implied by the Arrow type (dictionary/ordered, date/time types); None = value rules.

    Numbers (int, float, decimal) are left to the value rules like any number, so the header rules come first
    (a ``decimal128(10, 0)`` ``school_id`` is a D10 org code, R4-05); the numeric rule types a native decimal
    as DECIMAL (``detect.numeric.detect_numeric_extended``) and the writer casts every value back.
    """
    native = ctx.native or {}
    if native.get("family") != "arrow":
        return None
    kind = native.get("kind")
    if kind == "unsupported":
        if not ctx.text_values:
            return None  # every value null: an EMPTY column, written back as nulls of its type
        raise UnsupportedColumnError(
            f"column {ctx.name!r}: Arrow {native.get('arrow_type')} values are never copied or stringified",
            fix=f"--null {ctx.name} (keeps the column and its type, every value missing) or --exclude {ctx.name}")
    if kind == "dictionary":
        return _dictionary_detection(ctx, native)
    if not ctx.typed_values:
        return None
    if kind in TEMPORAL_KINDS:
        t = native["field"].type
        target = (ColumnType.DATETIME_TZ if t.tz is not None else ColumnType.DATETIME) if kind == "timestamp" \
            else _TEMPORAL_TARGET[kind]
        return native_date_detection(ctx, target, f"Arrow {t}")
    return None


# --------------------------------------------------------------------------- self-check
def _matching_fields(src: Any, out_names: list[str]) -> list[int] | None:
    """Indices of the source fields matching the output columns in order (excluded columns skipped)."""
    found = []
    j = 0
    for name in out_names:
        while j < len(src) and src.field(j).name != name:
            j += 1
        if j >= len(src):
            return None
        found.append(j)
        j += 1
    return found


def schema_diff(src: Any, out: Any) -> list[str]:
    """Field-by-field Arrow schema differences (type, nullability, field metadata) plus schema metadata.

    Metadata is compared with the source's *structural* metadata (:func:`safe_field_metadata`,
    :func:`safe_schema_metadata`): free-form source metadata is expected to be absent, and any
    other metadata in the output is a difference.
    """
    if src is None or out is None:
        return ["Arrow schema missing"]
    issues: list[str] = []
    expected = _matching_fields(src, list(out.names))
    if expected is None:
        issues.append("column names/order differ from the source Arrow schema")
    else:
        for sf, of in zip((src.field(i) for i in expected), out):
            if not sf.type.equals(of.type):
                issues.append(f"column {of.name!r}: Arrow type {sf.type} -> {of.type}")
            if sf.nullable != of.nullable:
                issues.append(f"column {of.name!r}: nullability {sf.nullable} -> {of.nullable}")
            if (safe_field_metadata(sf.metadata)[0] or {}) != (of.metadata or {}):
                issues.append(f"column {of.name!r}: field metadata differs from the source's structural metadata")
    if _comparable_metadata(safe_schema_metadata(src.metadata, list(out.names))[0]) != \
            _comparable_metadata(out.metadata):
        issues.append("schema (file) metadata differs from the source's structural metadata")
    return issues


def unused_dictionary_entries(tables: list[Any]) -> list[tuple[str, int]]:
    """(column, number of entries) of dictionary columns whose dictionary holds an entry no row holds that is
    not a tool-vocabulary word (:func:`vocabulary_label`).

    ``tables``: re-read RawTables. The writer only writes the entries the synthetic rows hold (an unused
    source entry was never checked and can quote anything), so any other entry in an output is a defect.
    """
    out: list[tuple[str, int]] = []
    for t in tables:
        for col in getattr(t, "columns", []) or []:
            nat = col.native or {}
            if nat.get("kind") != "dictionary":
                continue
            entries = [v for v in (nat.get("dictionary_values") or []) if v is not None]
            held = {v for v in col.values.tolist() if v is not None}
            n = sum(1 for v in set(entries) if v not in held and not vocabulary_label(v))
            if n:
                out.append((col.name, n))
    return out


def _norm_codec(c: Any) -> str | None:
    if c is None:
        return None
    c = str(c).upper()
    return {"LZ4_RAW": "LZ4", "NONE": "UNCOMPRESSED"}.get(c, c)


def layout_diff(source: FileLayout, output: FileLayout, source_tables: list[dict[str, Any]],
                output_tables: list[Any]) -> list[str]:
    """Arrow schema and writer-setting differences for the self-check (Parquet, Feather, SAS-as-Parquet)."""
    s, o = source.extra or {}, output.extra or {}
    issues = schema_diff(s.get(SCHEMA_KEY), o.get(SCHEMA_KEY))
    for name, n in unused_dictionary_entries(output_tables):
        issues.append(f"column {name!r}: {n} dictionary entr{'y' if n == 1 else 'ies'} held by no row (a label no "
                      "row holds is never written)")
    if output.format == "parquet":
        sv = s.get("parquet_version")
        if sv in ("1.0", "2.4", "2.6") and sv != o.get("parquet_version"):
            issues.append(f"Parquet format version {sv} -> {o.get('parquet_version')}")
        if s.get("arrow_schema_stored", True) != o.get("arrow_schema_stored", True):
            issues.append("stored Arrow schema (ARROW:schema) differs from the source")
        if bool(s.get("int96_timestamps", False)) != bool(o.get("int96_timestamps", False)):
            issues.append("legacy INT96 timestamp storage differs from the source")
        src_schema, out_schema = s.get(SCHEMA_KEY), o.get(SCHEMA_KEY)
        if src_schema is not None and out_schema is not None and o.get("_codecs"):
            s_codecs, o_codecs = s.get("_codecs") or [], o["_codecs"]
            writable = {_norm_codec(x) for x in _PARQUET_CODEC_OUT}
            for k, i in enumerate(_matching_fields(src_schema, list(out_schema.names)) or []):
                sc = _norm_codec(s_codecs[i]) if i < len(s_codecs) else None
                oc = _norm_codec(o_codecs[k]) if k < len(o_codecs) else None
                if sc and oc and sc in writable and sc != oc:
                    issues.append(f"column {out_schema.field(k).name!r}: codec {sc} -> {oc}")
        srows, orows = s.get("row_group_rows"), o.get("row_group_rows")
        if srows and orows and orows > srows:
            issues.append(f"row-group size: source up to {srows} rows, output {orows}")
    elif output.format == "feather":
        if s.get("feather_version") != o.get("feather_version"):
            issues.append(f"Feather version {s.get('feather_version')} -> {o.get('feather_version')}")
        if s.get("codec") != o.get("codec") and o.get("feather_version") == 2:
            issues.append(f"Feather codec {s.get('codec')} -> {o.get('codec')}")
        sb, ob = s.get("batch_rows"), o.get("batch_rows")
        if sb and ob and ob > sb:
            issues.append(f"record-batch size: source up to {sb} rows, output {ob}")
    return issues


def inferred_dtypes(path: Path, layout: FileLayout, table: str | None) -> list[str] | None:
    """dtypes pandas infers with default ``read_parquet`` / ``read_feather`` (None if it cannot read the file)."""
    import pandas as pd

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = pd.read_feather(path) if layout.format == "feather" else pd.read_parquet(path)
    except Exception:
        return None
    return [str(t) for t in df.dtypes]
